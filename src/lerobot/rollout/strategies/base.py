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

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import time

from lerobot.policies.rtc import ActionInterpolator
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from ..inference import InferenceEngine, _resolve_action_key_order
from . import RolloutStrategy

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    Supports both synchronous and RTC inference backends via the
    :class:`InferenceEngine`.  All actions flow through the
    ``robot_action_processor`` pipeline before reaching the robot.
    """

    def __init__(self, config):
        super().__init__(config)
        self._engine: InferenceEngine | None = None

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
        logger.info("Base strategy ready (rtc=%s)", self._engine.is_rtc)

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.cfg
        robot = ctx.robot_wrapper
        action_keys = ctx.action_keys

        interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
        control_interval = interpolator.get_control_interval(cfg.fps)

        policy_action_names = getattr(cfg.policy, "action_feature_names", None)
        ordered_keys = _resolve_action_key_order(
            list(policy_action_names) if policy_action_names else None,
            action_keys,
        )

        start_time = time.perf_counter()
        warmup_flushed = False

        if engine.is_rtc:
            engine.resume()

        while not ctx.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                break

            obs = robot.get_observation()
            obs_processed = ctx.robot_observation_processor(obs)

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
                    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys) if i < len(interp)}
                    processed = ctx.robot_action_processor((action_dict, obs))
                    robot.send_action(processed)

            else:
                obs_frame = build_dataset_frame(ctx.dataset_features, obs_processed, prefix=OBS_STR)
                action_tensor = engine.get_action_sync(obs_frame)
                action_dict = make_robot_action(action_tensor, ctx.dataset_features)
                processed = ctx.robot_action_processor((action_dict, obs))
                robot.send_action(processed)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

    def teardown(self, ctx: RolloutContext) -> None:
        if self._engine is not None:
            self._engine.stop()
        if ctx.robot.is_connected:
            ctx.robot.disconnect()
        if ctx.teleop is not None and ctx.teleop.is_connected:
            ctx.teleop.disconnect()
        logger.info("Base strategy teardown complete")
