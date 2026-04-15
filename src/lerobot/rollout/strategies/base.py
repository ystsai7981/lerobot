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

from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .core import RolloutStrategy, infer_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    Supports both synchronous and RTC inference backends via the
    :class:`InferenceEngine`.  All actions flow through the
    ``robot_action_processor`` pipeline before reaching the robot.
    """

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)
        logger.info("Base strategy ready (rtc=%s)", self._engine.is_rtc)

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.cfg
        robot = ctx.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        ordered_keys = ctx.ordered_action_keys

        start_time = time.perf_counter()

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

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            infer_action(engine, obs_processed, obs, ctx, interpolator, ordered_keys, ctx.dataset_features)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

    def teardown(self, ctx: RolloutContext) -> None:
        self._teardown_hardware(ctx)
        logger.info("Base strategy teardown complete")
