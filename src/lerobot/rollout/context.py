# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rollout context: shared state created once before strategy dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Event

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.robots import Robot, make_robot_from_config
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

from .configs import BaseStrategyConfig, RolloutConfig
from .robot_wrapper import ThreadSafeRobot

logger = logging.getLogger(__name__)


@dataclass
class RolloutContext:
    """Bundle of shared resources passed to every rollout strategy.

    Built once by :func:`build_rollout_context` before strategy dispatch.
    """

    cfg: RolloutConfig
    robot: Robot
    robot_wrapper: ThreadSafeRobot
    teleop: Teleoperator | None
    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation]
    dataset: LeRobotDataset | None
    shutdown_event: Event = field(default_factory=Event)
    dataset_features: dict = field(default_factory=dict)
    action_keys: list[str] = field(default_factory=list)
    hw_features: dict = field(default_factory=dict)


def build_rollout_context(cfg: RolloutConfig, shutdown_event: Event) -> RolloutContext:
    """Wire up hardware, policy, processors, and dataset from config.

    This function performs all the one-time setup that every strategy
    needs, keeping the strategy implementations lean.
    """
    # --- Hardware ---
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = ThreadSafeRobot(robot)

    teleop = None
    if cfg.teleop is not None:
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()

    # --- Processors ---
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # --- Policy ---
    use_rtc = cfg.rtc.enabled
    policy_class = get_policy_class(cfg.policy.type)
    policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    # Set compile_model for pi0/pi05
    if hasattr(policy_config, "compile_model"):
        policy_config.compile_model = cfg.use_torch_compile

    # Handle PEFT models
    if policy_config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_path = cfg.policy.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=policy_config
        )
        policy = PeftModel.from_pretrained(policy, peft_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=policy_config)

    # Enable RTC on the policy
    if use_rtc:
        policy.config.rtc_config = cfg.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply torch.compile if requested (skip pi0/pi05 — they handle their own)
    if cfg.use_torch_compile and policy.type not in ("pi0", "pi05"):
        try:
            if hasattr(torch, "compile"):
                compile_kwargs = {
                    "backend": cfg.torch_compile_backend,
                    "mode": cfg.torch_compile_mode,
                    "options": {"triton.cudagraphs": False},
                }
                policy.predict_action_chunk = torch.compile(policy.predict_action_chunk, **compile_kwargs)
                logger.info("torch.compile applied to predict_action_chunk")
        except Exception as e:
            logger.warning("Failed to apply torch.compile: %s", e)

    # --- Observation features (filter to .pos joints + camera streams) ---
    all_obs_features = robot.observation_features
    observation_features_hw = {
        k: v for k, v in all_obs_features.items() if k.endswith(".pos") or isinstance(v, tuple)
    }

    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}

    # Build dataset features
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=cfg.dataset.video if cfg.dataset else True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features_hw),
            use_videos=cfg.dataset.video if cfg.dataset else True,
        ),
    )

    hw_features = hw_to_dataset_features(observation_features_hw, "observation")

    # Action keys
    action_keys = [k for k in robot.action_features if k.endswith(".pos")]

    # --- Dataset ---
    dataset = None
    if cfg.dataset is not None and not isinstance(cfg.strategy, BaseStrategyConfig):
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
            )
        else:
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

    # --- Pre/post processors ---
    dataset_stats = None
    if dataset is not None:
        dataset_stats = rename_stats(
            dataset.meta.stats,
            cfg.dataset.rename_map if cfg.dataset else {},
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device or cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.dataset.rename_map if cfg.dataset else {}},
        },
    )

    return RolloutContext(
        cfg=cfg,
        robot=robot,
        robot_wrapper=robot_wrapper,
        teleop=teleop,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        dataset=dataset,
        shutdown_event=shutdown_event,
        dataset_features=dataset_features,
        action_keys=action_keys,
        hw_features=hw_features,
    )
