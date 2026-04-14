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

"""Sentry rollout strategy: continuous autonomous recording with auto-upload."""

from __future__ import annotations

import logging
import time
from threading import Thread

from lerobot.datasets import VideoEncodingManager
from lerobot.policies.rtc import ActionInterpolator
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..configs import SentryStrategyConfig
from ..context import RolloutContext
from ..inference import InferenceEngine, _resolve_action_key_order
from . import RolloutStrategy

logger = logging.getLogger(__name__)


class SentryStrategy(RolloutStrategy):
    """Continuous autonomous rollout with always-on recording.

    Episodes are auto-rotated every ``episode_duration_s`` seconds.
    The dataset is pushed to Hub in the background every
    ``upload_every_n_episodes`` episodes.

    Requires ``streaming_encoding=True`` (enforced in config validation)
    to prevent disk I/O from blocking the control loop.

    All actions flow through ``robot_observation_processor`` (observations)
    and ``robot_action_processor`` (actions) before reaching the robot,
    supporting EE-space recording with joint-space robots.
    """

    config: SentryStrategyConfig

    def __init__(self, config: SentryStrategyConfig):
        super().__init__(config)
        self._engine: InferenceEngine | None = None
        self._push_thread: Thread | None = None

    def setup(self, ctx: RolloutContext) -> None:
        interpolator = ActionInterpolator(multiplier=ctx.cfg.interpolation_multiplier)

        self._engine = InferenceEngine(
            policy=ctx.policy,
            preprocessor=ctx.preprocessor,
            postprocessor=ctx.postprocessor,
            robot_wrapper=ctx.robot_wrapper,
            rtc_config=ctx.cfg.rtc,
            hw_features=ctx.hw_features,
            action_keys=ctx.action_keys,
            task=ctx.cfg.task,
            fps=ctx.cfg.fps,
            device=ctx.cfg.device,
            interpolator=interpolator,
            use_torch_compile=ctx.cfg.use_torch_compile,
            compile_warmup_inferences=ctx.cfg.compile_warmup_inferences,
        )
        self._engine.start()
        logger.info(
            "Sentry strategy ready (episode_duration=%.0fs, upload_every=%d eps)",
            self.config.episode_duration_s,
            self.config.upload_every_n_episodes,
        )

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.cfg
        robot = ctx.robot_wrapper
        dataset = ctx.dataset
        action_keys = ctx.action_keys

        interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
        control_interval = interpolator.get_control_interval(cfg.fps)

        policy_action_names = getattr(cfg.policy, "action_feature_names", None)
        ordered_keys = _resolve_action_key_order(
            list(policy_action_names) if policy_action_names else None,
            action_keys,
        )

        if engine.is_rtc:
            engine.resume()

        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        warmup_flushed = False
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task

        with VideoEncodingManager(dataset):
            try:
                while not ctx.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        break

                    obs = robot.get_observation()
                    obs_processed = ctx.robot_observation_processor(obs)
                    action_dict = None

                    if engine.is_rtc:
                        engine.update_observation(obs_processed)

                        if cfg.use_torch_compile and not engine.compile_warmup_done.is_set():
                            dt = time.perf_counter() - loop_start
                            if (sleep_t := control_interval - dt) > 0:
                                precise_sleep(sleep_t)
                            continue

                        if cfg.use_torch_compile and not warmup_flushed:
                            engine.reset()
                            interpolator.reset()
                            warmup_flushed = True

                        if interpolator.needs_new_action():
                            action_tensor = engine.consume_rtc_action()
                            if action_tensor is not None:
                                interpolator.add(action_tensor.cpu())

                        interp = interpolator.get()
                        if interp is not None:
                            action_dict = {
                                k: interp[i].item() for i, k in enumerate(ordered_keys) if i < len(interp)
                            }
                            processed = ctx.robot_action_processor((action_dict, obs))
                            robot.send_action(processed)
                    else:
                        obs_frame = build_dataset_frame(ctx.dataset_features, obs_processed, prefix=OBS_STR)
                        action_tensor = engine.get_action_sync(obs_frame)
                        action_dict = make_robot_action(action_tensor, ctx.dataset_features)
                        processed = ctx.robot_action_processor((action_dict, obs))
                        robot.send_action(processed)

                    # Record frame
                    if action_dict is not None:
                        obs_frame = build_dataset_frame(ctx.dataset_features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(ctx.dataset_features, action_dict, prefix=ACTION)
                        frame = {**obs_frame, **action_frame, "task": task_str}
                        dataset.add_frame(frame)

                    # Auto-rotate episodes
                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= self.config.episode_duration_s:
                        dataset.save_episode()
                        episodes_since_push += 1
                        logger.info("Episode saved (total: %d)", dataset.num_episodes)

                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0

                        episode_start = time.perf_counter()
                        engine.reset()
                        interpolator.reset()
                        if engine.is_rtc:
                            engine.resume()

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)

            finally:
                try:
                    dataset.save_episode()
                except Exception:
                    pass

    def teardown(self, ctx: RolloutContext) -> None:
        if self._engine is not None:
            self._engine.stop()

        if ctx.dataset is not None:
            ctx.dataset.finalize()
            if ctx.cfg.dataset and ctx.cfg.dataset.push_to_hub:
                ctx.dataset.push_to_hub(
                    tags=ctx.cfg.dataset.tags,
                    private=ctx.cfg.dataset.private,
                )

        if self._push_thread is not None and self._push_thread.is_alive():
            self._push_thread.join(timeout=60)

        if ctx.robot.is_connected:
            ctx.robot.disconnect()
        if ctx.teleop is not None and ctx.teleop.is_connected:
            ctx.teleop.disconnect()
        logger.info("Sentry strategy teardown complete")

    def _background_push(self, dataset, cfg) -> None:
        """Push dataset to hub in a background thread (non-blocking)."""
        if self._push_thread is not None and self._push_thread.is_alive():
            logger.info("Previous push still in progress, skipping")
            return

        def _push():
            try:
                dataset.push_to_hub(
                    tags=cfg.dataset.tags if cfg.dataset else None,
                    private=cfg.dataset.private if cfg.dataset else False,
                )
                logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._push_thread = Thread(target=_push, daemon=True)
        self._push_thread.start()
