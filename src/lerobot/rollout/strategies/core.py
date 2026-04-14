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

"""Rollout strategy ABC and shared inference helper."""

from __future__ import annotations

import abc
import time
from typing import TYPE_CHECKING

import torch

from lerobot.policies.rtc import ActionInterpolator
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

if TYPE_CHECKING:
    from lerobot.rollout.configs import RolloutStrategyConfig
    from lerobot.rollout.context import RolloutContext
    from lerobot.rollout.inference import InferenceEngine


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Create and start the inference engine and action interpolator.

        Call this from ``setup()`` to avoid duplicating the engine
        construction across every strategy.
        """
        from lerobot.rollout.inference import InferenceEngine

        self._interpolator = ActionInterpolator(multiplier=ctx.cfg.interpolation_multiplier)
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
            use_torch_compile=ctx.cfg.use_torch_compile,
            compile_warmup_inferences=ctx.cfg.compile_warmup_inferences,
            shutdown_event=ctx.shutdown_event,
        )
        self._engine.start()
        self._warmup_flushed = False

    def _handle_warmup(self, use_torch_compile: bool, loop_start: float, control_interval: float) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.compile_warmup_done.is_set():
            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            return True
        if not self._warmup_flushed:
            engine.reset()
            interpolator.reset()
            self._warmup_flushed = True
            if engine.is_rtc:
                engine.resume()
        return False

    def _teardown_hardware(self, ctx: RolloutContext) -> None:
        """Stop the inference engine and disconnect hardware."""
        if self._engine is not None:
            self._engine.stop()
        if ctx.robot.is_connected:
            ctx.robot.disconnect()
        if ctx.teleop is not None and ctx.teleop.is_connected:
            ctx.teleop.disconnect()

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires."""

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: save dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared inference helper
# ---------------------------------------------------------------------------


def infer_action(
    engine: InferenceEngine,
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
    ordered_keys: list[str],
    features: dict,
) -> dict | None:
    """Run one policy inference step and send the resulting action to the robot.

    Handles both sync and RTC backends.  Uses the interpolator for smooth
    control at higher-than-inference rates (works with any multiplier,
    including 1 where it acts as a pass-through).

    Parameters
    ----------
    engine:
        The inference engine (sync or RTC).
    obs_processed:
        Observation dict after ``robot_observation_processor``.
    obs_raw:
        Raw observation dict (needed by ``robot_action_processor``).
    ctx:
        Rollout context.
    interpolator:
        Action interpolator for Nx control rate.
    ordered_keys:
        Ordered action feature names (policy-to-robot mapping).
    features:
        Feature specification dict for ``build_dataset_frame`` /
        ``make_robot_action``.  Use ``dataset.features`` when recording,
        ``ctx.dataset_features`` otherwise.

    Returns
    -------
    Action dict sent to the robot, or ``None`` if no action was
    available (empty RTC queue, interpolator buffer not ready).
    """
    if engine.is_rtc:
        if interpolator.needs_new_action():
            action_tensor = engine.consume_rtc_action()
            if action_tensor is not None:
                interpolator.add(action_tensor.cpu())
    else:
        if interpolator.needs_new_action():
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            action_tensor = engine.get_action_sync(obs_frame)
            action_dict = make_robot_action(action_tensor, features)
            action_t = torch.tensor([action_dict[k] for k in ordered_keys])
            interpolator.add(action_t)

    interp = interpolator.get()
    if interp is not None:
        action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys) if i < len(interp)}
        processed = ctx.robot_action_processor((action_dict, obs_raw))
        ctx.robot_wrapper.send_action(processed)
        return action_dict
    return None
