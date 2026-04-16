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

"""Rollout context: shared state created once before strategy dispatch.

Grouped into five topical sub-contexts — :class:`RuntimeContext`,
:class:`HardwareContext`, :class:`PolicyContext`, :class:`ProcessorContext`,
and :class:`DatasetContext` — assembled into :class:`RolloutContext`.
"""

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
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

from .configs import BaseStrategyConfig, DAggerStrategyConfig, RolloutConfig
from .inference import (
    InferenceEngine,
    RTCInferenceConfig,
    create_inference_engine,
)
from .robot_wrapper import ThreadSafeRobot

logger = logging.getLogger(__name__)


def _resolve_action_key_order(
    policy_action_names: list[str] | None, dataset_action_names: list[str]
) -> list[str]:
    """Choose action name ordering for mapping policy tensor outputs to robot action dicts."""
    if not policy_action_names:
        return dataset_action_names
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(dataset_action_names):
        logger.warning(
            "policy.action_feature_names length (%d) != dataset action dim (%d); using dataset order",
            len(policy_action_names),
            len(dataset_action_names),
        )
        return dataset_action_names
    if set(dataset_action_names) != set(policy_action_names):
        logger.warning("policy.action_feature_names keys don't match dataset; using dataset order")
        return dataset_action_names
    return policy_action_names


# ---------------------------------------------------------------------------
# Sub-contexts
# ---------------------------------------------------------------------------


@dataclass
class RuntimeContext:
    """Runtime knobs shared with every strategy."""

    cfg: RolloutConfig
    shutdown_event: Event


@dataclass
class HardwareContext:
    """Connected hardware.

    The raw robot is available via ``robot_wrapper.inner`` when needed
    (e.g. for disconnect); strategies should otherwise go through the
    thread-safe wrapper.
    """

    robot_wrapper: ThreadSafeRobot
    teleop: Teleoperator | None


@dataclass
class PolicyContext:
    """Loaded policy and its inference engine."""

    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    inference: InferenceEngine


@dataclass
class ProcessorContext:
    """Robot-side pipelines (run outside the policy)."""

    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation]


@dataclass
class DatasetContext:
    """Dataset and feature bookkeeping."""

    dataset: LeRobotDataset | None
    dataset_features: dict = field(default_factory=dict)
    hw_features: dict = field(default_factory=dict)
    ordered_action_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutContext:
    """Bundle of sub-contexts passed to every rollout strategy.

    Built once by :func:`build_rollout_context` before strategy dispatch.
    """

    runtime: RuntimeContext
    hardware: HardwareContext
    policy: PolicyContext
    processors: ProcessorContext
    data: DatasetContext


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> RolloutContext:
    """Wire up policy, processors, hardware, dataset, and inference engine.

    The order is policy-first / hardware-last so a bad ``--policy.path``
    fails fast without touching the robot.
    """
    is_rtc = isinstance(cfg.inference, RTCInferenceConfig)

    # --- 1. Policy (heavy I/O, but no hardware yet) -------------------
    policy_config = cfg.policy
    policy_class = get_policy_class(policy_config.type)

    full_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
    for attr in ("device", "use_amp"):
        if hasattr(cfg.policy, attr) and hasattr(full_config, attr):
            cli_val = getattr(cfg.policy, attr)
            if cli_val is not None:
                setattr(full_config, attr, cli_val)

    if hasattr(full_config, "compile_model"):
        full_config.compile_model = cfg.use_torch_compile

    if full_config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_path = cfg.policy.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=full_config
        )
        policy = PeftModel.from_pretrained(policy, peft_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=full_config)

    if is_rtc:
        policy.config.rtc_config = cfg.inference.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

    policy = policy.to(cfg.device)
    policy.eval()

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

    # --- 2. Robot-side processors (user-supplied or defaults) --------
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    # --- 3. Hardware (heaviest side-effect, deferred) -----------------
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = ThreadSafeRobot(robot)

    teleop = None
    if cfg.teleop is not None:
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()

    # --- 4. Features + action-key reconciliation ---------------------
    all_obs_features = robot.observation_features
    observation_features_hw = {
        k: v for k, v in all_obs_features.items() if v is float or isinstance(v, tuple)
    }
    action_features_hw = robot.action_features

    # The action side is always needed: sync inference reads action names from
    # ``dataset_features[ACTION]`` to map policy tensors back to robot actions.
    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=cfg.dataset.video if cfg.dataset else True,
    )
    # Observation-side aggregation is needed because of build_dataset_frame
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=cfg.dataset.video if cfg.dataset else True,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    raw_action_keys = list(robot.action_features.keys())
    policy_action_names = getattr(policy_config, "action_feature_names", None)
    ordered_action_keys = _resolve_action_key_order(
        list(policy_action_names) if policy_action_names else None,
        raw_action_keys,
    )

    # --- 5. Dataset -------------
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
            if isinstance(cfg.strategy, DAggerStrategyConfig):
                dataset_features["intervention"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                }

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

    # --- 6. Policy pre/post processors (needs dataset stats if any) ---
    dataset_stats = None
    if dataset is not None:
        dataset_stats = rename_stats(
            dataset.meta.stats,
            cfg.dataset.rename_map if cfg.dataset else {},
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device or getattr(policy_config, "device", "cpu")},
            "rename_observations_processor": {"rename_map": cfg.dataset.rename_map if cfg.dataset else {}},
        },
    )

    # --- 7. Inference strategy (needs policy + pre/post + hardware) --
    task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
    inference_strategy = create_inference_engine(
        cfg.inference,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=task_str,
        fps=cfg.fps,
        device=cfg.device,
        use_torch_compile=cfg.use_torch_compile,
        compile_warmup_inferences=cfg.compile_warmup_inferences,
        shutdown_event=shutdown_event,
    )

    # --- 8. Assemble ---------------------------------------------------
    return RolloutContext(
        runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
        hardware=HardwareContext(robot_wrapper=robot_wrapper, teleop=teleop),
        policy=PolicyContext(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            inference=inference_strategy,
        ),
        processors=ProcessorContext(
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        ),
        data=DatasetContext(
            dataset=dataset,
            dataset_features=dataset_features,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
        ),
    )
