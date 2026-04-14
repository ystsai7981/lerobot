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

"""Highlight Reel strategy: on-demand recording via ring buffer."""

from __future__ import annotations

import logging
import time

from lerobot.datasets import VideoEncodingManager
from lerobot.policies.rtc import ActionInterpolator
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..configs import HighlightStrategyConfig
from ..context import RolloutContext
from ..inference import InferenceEngine, _resolve_action_key_order
from ..ring_buffer import RolloutRingBuffer
from . import RolloutStrategy

logger = logging.getLogger(__name__)


class HighlightStrategy(RolloutStrategy):
    """Autonomous rollout with on-demand recording via ring buffer.

    The robot runs autonomously while a memory-bounded ring buffer
    captures continuous telemetry.  When the user presses the save key:

    1. The ring buffer is flushed to the dataset (last *Z* seconds).
    2. Live recording continues until the save key is pressed again.
    3. The episode is saved and the ring buffer resumes capturing.

    All actions flow through ``robot_action_processor`` before reaching
    the robot, supporting EE-space recording with joint-space robots.
    """

    config: HighlightStrategyConfig

    def __init__(self, config: HighlightStrategyConfig):
        super().__init__(config)
        self._engine: InferenceEngine | None = None
        self._ring: RolloutRingBuffer | None = None
        self._listener = None
        self._save_requested = False
        self._recording_live = False

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

        self._ring = RolloutRingBuffer(
            max_seconds=self.config.ring_buffer_seconds,
            max_memory_mb=self.config.ring_buffer_max_memory_mb,
            fps=ctx.cfg.fps,
        )

        self._setup_keyboard()
        logger.info(
            "Highlight strategy ready (buffer=%.0fs, key='%s')",
            self.config.ring_buffer_seconds,
            self.config.save_key,
        )

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.cfg
        robot = ctx.robot_wrapper
        dataset = ctx.dataset
        action_keys = ctx.action_keys
        ring = self._ring

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

                    # Build frame for ring buffer / live recording
                    if action_dict is not None:
                        obs_frame = build_dataset_frame(ctx.dataset_features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(ctx.dataset_features, action_dict, prefix=ACTION)
                        frame = {**obs_frame, **action_frame, "task": task_str}

                        # Handle save key toggle
                        if self._save_requested:
                            self._save_requested = False
                            if not self._recording_live:
                                logger.info(
                                    "Flushing ring buffer (%d frames) + starting live recording", len(ring)
                                )
                                for buffered_frame in ring.drain():
                                    dataset.add_frame(buffered_frame)
                                self._recording_live = True
                            else:
                                dataset.add_frame(frame)
                                dataset.save_episode()
                                logger.info("Episode saved")
                                self._recording_live = False
                                engine.reset()
                                interpolator.reset()
                                if engine.is_rtc:
                                    engine.resume()

                        if self._recording_live:
                            dataset.add_frame(frame)
                        else:
                            ring.append(frame)

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)

            finally:
                if self._recording_live:
                    try:
                        dataset.save_episode()
                    except Exception:
                        pass

    def teardown(self, ctx: RolloutContext) -> None:
        if self._engine is not None:
            self._engine.stop()
        if self._listener is not None:
            self._listener.stop()

        if ctx.dataset is not None:
            ctx.dataset.finalize()
            if ctx.cfg.dataset and ctx.cfg.dataset.push_to_hub:
                ctx.dataset.push_to_hub(
                    tags=ctx.cfg.dataset.tags,
                    private=ctx.cfg.dataset.private,
                )

        if ctx.robot.is_connected:
            ctx.robot.disconnect()
        if ctx.teleop is not None and ctx.teleop.is_connected:
            ctx.teleop.disconnect()
        logger.info("Highlight strategy teardown complete")

    def _setup_keyboard(self) -> None:
        """Set up keyboard listener for the save key."""
        from lerobot.common.control_utils import is_headless

        if is_headless():
            logger.warning("Headless environment — highlight save key unavailable")
            return

        try:
            from pynput import keyboard

            save_key = self.config.save_key

            def on_press(key):
                try:
                    if hasattr(key, "char") and key.char == save_key:
                        self._save_requested = True
                    elif key == keyboard.Key.esc:
                        self._save_requested = False
                except Exception:
                    pass

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
        except ImportError:
            logger.warning("pynput not available — keyboard listener disabled")
