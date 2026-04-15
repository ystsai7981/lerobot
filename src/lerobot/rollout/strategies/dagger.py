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

"""DAgger rollout strategy: Human-in-the-Loop data collection.

Implements the RaC paradigm (Recovery and Correction) for interactive
imitation learning.  Alternates between autonomous policy execution and
human intervention via teleoperator.

Keyboard Controls:
    SPACE  - Pause policy (robot holds position, no recording)
    c      - Take control (start correction, recording resumes)
    p      - Resume policy after pause/correction
    ->     - End episode (save and continue)
    <-     - Re-record episode
    ESC    - Stop recording and push to hub
"""

from __future__ import annotations

import contextlib
import enum
import logging
import time
from threading import Event, Lock
from typing import Any

import numpy as np

from lerobot.common.control_utils import is_headless
from lerobot.datasets import VideoEncodingManager
from lerobot.processor import RobotProcessorPipeline
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.pedal import start_pedal_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import DAggerStrategyConfig
from ..context import RolloutContext
from ..robot_wrapper import ThreadSafeRobot
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


class DAggerPhase(enum.Enum):
    """Observable phases of a DAgger episode."""

    AUTONOMOUS = "autonomous"  # Policy driving, recording autonomous frames
    PAUSED = "paused"  # Engine paused, teleop aligned, awaiting takeover/resume
    CORRECTING = "correcting"  # Human driving via teleop, recording interventions


# Valid (current_phase, event) → next_phase
_DAGGER_TRANSITIONS: dict[tuple[DAggerPhase, str], DAggerPhase] = {
    (DAggerPhase.AUTONOMOUS, "pause"): DAggerPhase.PAUSED,
    (DAggerPhase.PAUSED, "takeover"): DAggerPhase.CORRECTING,
    (DAggerPhase.PAUSED, "resume"): DAggerPhase.AUTONOMOUS,
    (DAggerPhase.CORRECTING, "resume"): DAggerPhase.AUTONOMOUS,
}


class DAggerEvents:
    """Thread-safe container for DAgger keyboard/pedal events.

    The keyboard/pedal threads write transition requests; the main loop
    consumes them.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = DAggerPhase.AUTONOMOUS
        self._pending_transition: str | None = None

        # Episode-level flags written by keyboard/pedal threads, consumed by
        # the main loop.  ``threading.Event`` gives us atomic set/clear/check
        # semantics without taking ``self._lock``.
        self.exit_early = Event()
        self.rerecord_episode = Event()
        self.stop_recording = Event()

        # Reset-phase flags (simpler lifecycle, shared between threads).
        self.in_reset = Event()
        self.start_next_episode = Event()

    # -- Thread-safe phase access ------------------------------------------

    @property
    def phase(self) -> DAggerPhase:
        with self._lock:
            return self._phase

    @phase.setter
    def phase(self, value: DAggerPhase) -> None:
        with self._lock:
            self._phase = value

    def request_transition(self, event: str) -> None:
        """Request a phase transition (called from keyboard/pedal threads).

        Only enqueues the request if it corresponds to a valid transition
        from the current phase, preventing impossible state changes.
        """
        with self._lock:
            if (self._phase, event) in _DAGGER_TRANSITIONS:
                self._pending_transition = event

    def consume_transition(self) -> tuple[DAggerPhase, DAggerPhase] | None:
        """Consume a pending transition (called from main loop)."""
        with self._lock:
            if self._pending_transition is None:
                return None
            key = (self._phase, self._pending_transition)
            self._pending_transition = None
            new_phase = _DAGGER_TRANSITIONS.get(key)
            if new_phase is None:
                return None
            old_phase = self._phase
            self._phase = new_phase
            return old_phase, new_phase

    def reset_for_episode(self) -> None:
        """Reset all transient state at the start of an episode."""
        with self._lock:
            self._phase = DAggerPhase.AUTONOMOUS
            self._pending_transition = None
        self.exit_early.clear()
        self.rerecord_episode.clear()


# ---------------------------------------------------------------------------
# Teleoperator helpers
# ---------------------------------------------------------------------------


def _teleop_has_motor_control(teleop: Teleoperator) -> bool:
    return all(hasattr(teleop, attr) for attr in ("enable_torque", "disable_torque", "write_goal_positions"))


def _teleop_disable_torque(teleop: Teleoperator) -> None:
    if hasattr(teleop, "disable_torque"):
        teleop.disable_torque()


def _teleop_enable_torque(teleop: Teleoperator) -> None:
    if hasattr(teleop, "enable_torque"):
        teleop.enable_torque()


def _teleop_smooth_move_to(
    teleop: Teleoperator, target_pos: dict, duration_s: float = 2.0, fps: int = 50
) -> None:
    """Smoothly move teleop to target position if motor control is available."""
    if not _teleop_has_motor_control(teleop):
        logger.warning("Teleop does not support motor control — cannot mirror robot position")
        return

    _teleop_enable_torque(teleop)
    current = teleop.get_action()
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {}
        for k in current:
            if k in target_pos:
                interp[k] = current[k] * (1 - t) + target_pos[k] * t
            else:
                interp[k] = current[k]
        teleop.write_goal_positions(interp)
        time.sleep(1 / fps)


def _reset_loop(
    robot: ThreadSafeRobot,
    teleop: Teleoperator,
    events: DAggerEvents,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
) -> None:
    """Reset period where the human repositions the environment."""
    logger.info("RESET — press any key to enable teleoperation")

    events.in_reset.set()
    events.start_next_episode.clear()

    obs = robot.get_observation()
    robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features}
    _teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)

    while not events.start_next_episode.is_set() and not events.stop_recording.is_set():
        precise_sleep(0.05)

    if events.stop_recording.is_set():
        return

    events.start_next_episode.clear()
    _teleop_disable_torque(teleop)
    logger.info("Teleop enabled — press any key to start episode")

    while not events.start_next_episode.is_set() and not events.stop_recording.is_set():
        loop_start = time.perf_counter()
        obs = robot.get_observation()
        action = teleop.get_action()
        processed_teleop = teleop_action_processor((action, obs))
        robot_action_to_send = robot_action_processor((processed_teleop, obs))
        robot.send_action(robot_action_to_send)
        precise_sleep(1 / fps - (time.perf_counter() - loop_start))

    events.in_reset.clear()
    events.start_next_episode.clear()
    events.reset_for_episode()


def _init_dagger_keyboard(events: DAggerEvents):
    """Initialise keyboard listener with DAgger/HIL controls.

    Returns the pynput Listener (or ``None`` in headless mode).
    """
    if is_headless():
        logger.warning("Headless environment — keyboard controls unavailable")
        return None

    from pynput import keyboard

    def on_press(key):
        try:
            if events.in_reset.is_set():
                if (
                    key in [keyboard.Key.space, keyboard.Key.right]
                    or hasattr(key, "char")
                    and key.char == "c"
                ):
                    events.start_next_episode.set()
                elif key == keyboard.Key.esc:
                    events.stop_recording.set()
                    events.start_next_episode.set()
                return

            phase = events.phase
            if key == keyboard.Key.space and phase == DAggerPhase.AUTONOMOUS:
                logger.info("PAUSED — press 'c' to take control or 'p' to resume policy")
                events.request_transition("pause")
            elif hasattr(key, "char") and key.char == "c" and phase == DAggerPhase.PAUSED:
                logger.info("Taking control...")
                events.request_transition("takeover")
            elif (
                hasattr(key, "char")
                and key.char == "p"
                and phase
                in (
                    DAggerPhase.PAUSED,
                    DAggerPhase.CORRECTING,
                )
            ):
                logger.info("Resuming policy...")
                events.request_transition("resume")

            elif key == keyboard.Key.right:
                logger.info("End episode")
                events.exit_early.set()
            elif key == keyboard.Key.left:
                logger.info("Re-record episode")
                events.rerecord_episode.set()
                events.exit_early.set()
            elif key == keyboard.Key.esc:
                logger.info("Stop recording...")
                events.stop_recording.set()
                events.exit_early.set()
        except Exception as e:
            logger.debug("Key error: %s", e)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


_DAGGER_PEDAL_KEYS = ("KEY_A", "KEY_C")


def _dagger_pedal_callback(events: DAggerEvents):
    """Build the pedal key-press handler for DAgger's state machine."""

    def on_press(code: str) -> None:
        if code not in _DAGGER_PEDAL_KEYS:
            return
        if events.in_reset.is_set():
            events.start_next_episode.set()
            return
        phase = events.phase
        if phase == DAggerPhase.CORRECTING:
            events.request_transition("resume")
        elif phase == DAggerPhase.PAUSED:
            events.request_transition("takeover")
        elif phase == DAggerPhase.AUTONOMOUS:
            events.request_transition("pause")

    return on_press


# ---------------------------------------------------------------------------
# DAgger Strategy
# ---------------------------------------------------------------------------


class DAggerStrategy(RolloutStrategy):
    """Human-in-the-Loop data collection with intervention tagging.

    State machine::

        AUTONOMOUS --(SPACE)--> PAUSED --(c)--> CORRECTING --(p)--> AUTONOMOUS
                                       --(p)--> AUTONOMOUS

    Intervention frames are tagged with ``intervention=True`` (bool) in
    the dataset; autonomous frames with ``intervention=False``.  When
    ``record_autonomous=False`` only corrections are recorded.
    """

    config: DAggerStrategyConfig

    def __init__(self, config: DAggerStrategyConfig):
        super().__init__(config)
        self._listener = None
        self._events = DAggerEvents()

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)

        self._listener = _init_dagger_keyboard(self._events)
        start_pedal_listener(_dagger_pedal_callback(self._events))

        logger.info(
            "DAgger strategy ready (episodes=%d, episode_time=%.0fs, record_autonomous=%s)",
            self.config.num_episodes,
            self.config.episode_time_s,
            self.config.record_autonomous,
        )
        logger.info("Controls: SPACE=pause, c=take control, p=resume, ->=end, <-=redo, ESC=stop")

    def run(self, ctx: RolloutContext) -> None:
        dataset = ctx.data.dataset
        events = self._events
        teleop = ctx.hardware.teleop

        with VideoEncodingManager(dataset):
            try:
                recorded = 0
                while recorded < self.config.num_episodes and not events.stop_recording.is_set():
                    log_say(f"Episode {dataset.num_episodes}", self.config.play_sounds)

                    self._run_episode(ctx)

                    if events.rerecord_episode.is_set():
                        log_say("Re-recording", self.config.play_sounds)
                        events.rerecord_episode.clear()
                        events.exit_early.clear()
                        dataset.clear_episode_buffer()
                        continue

                    dataset.save_episode()
                    recorded += 1

                    if recorded < self.config.num_episodes and not events.stop_recording.is_set():
                        _reset_loop(
                            ctx.hardware.robot_wrapper,
                            teleop,
                            events,
                            int(ctx.runtime.cfg.fps),
                            ctx.processors.teleop_action_processor,
                            ctx.processors.robot_action_processor,
                        )

            finally:
                with contextlib.suppress(Exception):
                    dataset.save_episode()

    def teardown(self, ctx: RolloutContext) -> None:
        log_say("Stop recording", self.config.play_sounds, blocking=True)

        if self._listener is not None and not is_headless():
            self._listener.stop()

        if ctx.data.dataset is not None:
            ctx.data.dataset.finalize()
            if ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                ctx.data.dataset.push_to_hub(
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                )

        self._teardown_hardware(ctx)
        logger.info("DAgger strategy teardown complete")

    # ------------------------------------------------------------------
    # Episode rollout (state machine)
    # ------------------------------------------------------------------

    def _run_episode(self, ctx: RolloutContext) -> None:
        """Run a single DAgger episode with the HIL state machine."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        stream_online = bool(cfg.dataset.streaming_encoding) if cfg.dataset else False
        record_stride = max(1, cfg.interpolation_multiplier)
        record_autonomous = self.config.record_autonomous

        ordered_keys = ctx.data.ordered_action_keys
        features = dataset.features

        engine.reset()
        interpolator.reset()
        events.reset_for_episode()
        _teleop_disable_torque(teleop)

        last_action: dict[str, Any] | None = None
        frame_buffer: list[dict] = []
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task

        timestamp = 0.0
        record_tick = 0
        start_t = time.perf_counter()

        engine.resume()

        while timestamp < self.config.episode_time_s:
            loop_start = time.perf_counter()

            if events.exit_early.is_set():
                events.exit_early.clear()
                break

            transition = events.consume_transition()
            if transition is not None:
                old_phase, new_phase = transition
                self._apply_transition(old_phase, new_phase, engine, interpolator, robot, teleop)
                last_action = None

            phase = events.phase

            obs = robot.get_observation()
            obs_processed = ctx.processors.robot_observation_processor(obs)
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)

            # --- CORRECTING: human teleop control ---
            if phase == DAggerPhase.CORRECTING:
                teleop_action = teleop.get_action()
                processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                robot.send_action(robot_action_to_send)
                action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                if record_tick % record_stride == 0:
                    frame = {
                        **obs_frame,
                        **action_frame,
                        "task": task_str,
                        "intervention": np.array([True], dtype=bool),
                    }
                    if stream_online:
                        dataset.add_frame(frame)
                    else:
                        frame_buffer.append(frame)
                record_tick += 1

            # --- PAUSED: hold position ---
            elif phase == DAggerPhase.PAUSED:
                if last_action:
                    robot.send_action(last_action)

            # --- AUTONOMOUS: policy control ---
            else:
                engine.notify_observation(obs_processed)

                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    timestamp = time.perf_counter() - start_t
                    continue

                action_dict = send_next_action(
                    engine, obs_processed, obs, ctx, interpolator, ordered_keys, features
                )

                if action_dict is not None:
                    last_action = ctx.processors.robot_action_processor((action_dict, obs))
                    action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                    if record_autonomous and record_tick % record_stride == 0:
                        frame = {
                            **obs_frame,
                            **action_frame,
                            "task": task_str,
                            "intervention": np.array([False], dtype=bool),
                        }
                        if stream_online:
                            dataset.add_frame(frame)
                        else:
                            frame_buffer.append(frame)
                    record_tick += 1

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            timestamp = time.perf_counter() - start_t

        # End of episode: pause engine, disable teleop, flush buffer
        engine.pause()
        _teleop_disable_torque(teleop)

        if not stream_online:
            for frame in frame_buffer:
                dataset.add_frame(frame)

    # ------------------------------------------------------------------
    # State-machine transition side-effects
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transition(
        old_phase: DAggerPhase,
        new_phase: DAggerPhase,
        engine,
        interpolator,
        robot: ThreadSafeRobot,
        teleop: Teleoperator,
    ) -> None:
        """Execute side-effects for a validated phase transition."""
        if old_phase == DAggerPhase.AUTONOMOUS and new_phase == DAggerPhase.PAUSED:
            engine.pause()
            obs = robot.get_observation()
            robot_pos = {
                k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features
            }
            _teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)
            interpolator.reset()

        elif new_phase == DAggerPhase.CORRECTING:
            _teleop_disable_torque(teleop)
            engine.reset()

        elif new_phase == DAggerPhase.AUTONOMOUS:
            interpolator.reset()
            engine.reset()
            engine.resume()
