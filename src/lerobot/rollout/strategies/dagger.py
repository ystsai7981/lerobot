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
from threading import Lock
from typing import Any

import numpy as np

from lerobot.common.control_utils import is_headless
from lerobot.datasets import VideoEncodingManager
from lerobot.processor import RobotProcessorPipeline
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import DAggerStrategyConfig
from ..context import RolloutContext
from ..robot_wrapper import ThreadSafeRobot
from . import RolloutStrategy, infer_action

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

    Replaces the previous plain dict with a lock-protected phase enum
    and edge-triggered transition requests.  The keyboard/pedal threads
    write transition requests; the main loop consumes them.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = DAggerPhase.AUTONOMOUS
        self._pending_transition: str | None = None

        # Episode-level flags (written by keyboard, consumed by main loop)
        self.exit_early: bool = False
        self.rerecord_episode: bool = False
        self.stop_recording: bool = False

        # Reset-phase flags (simpler lifecycle, shared between threads)
        self.in_reset: bool = False
        self.start_next_episode: bool = False

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
        """Consume a pending transition (called from main loop).

        Returns ``(old_phase, new_phase)`` if a valid transition was
        pending, or ``None`` if there is nothing to process.
        """
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
        self.exit_early = False
        self.rerecord_episode = False


# ---------------------------------------------------------------------------
# Teleoperator helpers (extracted from examples/hil/hil_utils.py)
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
    """Reset period where the human repositions the environment.

    All teleop actions flow through the processor pipelines to ensure
    correct behavior for EE-space robots.
    """
    logger.info("RESET — press any key to enable teleoperation")

    events.in_reset = True
    events.start_next_episode = False

    obs = robot.get_observation()
    robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features}
    _teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)

    while not events.start_next_episode and not events.stop_recording:
        precise_sleep(0.05)

    if events.stop_recording:
        return

    events.start_next_episode = False
    _teleop_disable_torque(teleop)
    logger.info("Teleop enabled — press any key to start episode")

    while not events.start_next_episode and not events.stop_recording:
        loop_start = time.perf_counter()
        obs = robot.get_observation()
        action = teleop.get_action()
        processed_teleop = teleop_action_processor((action, obs))
        robot_action_to_send = robot_action_processor((processed_teleop, obs))
        robot.send_action(robot_action_to_send)
        precise_sleep(1 / fps - (time.perf_counter() - loop_start))

    events.in_reset = False
    events.start_next_episode = False
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
            # During the reset phase, only accept episode-start or stop
            if events.in_reset:
                if (
                    key in [keyboard.Key.space, keyboard.Key.right]
                    or hasattr(key, "char")
                    and key.char == "c"
                ):
                    events.start_next_episode = True
                elif key == keyboard.Key.esc:
                    events.stop_recording = True
                    events.start_next_episode = True
                return

            # Phase-aware transition requests
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

            # Episode-level controls (valid in any phase)
            elif key == keyboard.Key.right:
                logger.info("End episode")
                events.exit_early = True
            elif key == keyboard.Key.left:
                logger.info("Re-record episode")
                events.rerecord_episode = True
                events.exit_early = True
            elif key == keyboard.Key.esc:
                logger.info("Stop recording...")
                events.stop_recording = True
                events.exit_early = True
        except Exception as e:
            logger.debug("Key error: %s", e)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _start_pedal_listener(events: DAggerEvents) -> None:
    """Start foot pedal listener thread if evdev is available."""
    import threading

    try:
        from evdev import InputDevice, categorize, ecodes
    except ImportError:
        return

    pedal_device = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"

    def pedal_reader():
        try:
            dev = InputDevice(pedal_device)
            logger.info("Pedal connected: %s", dev.name)
            for ev in dev.read_loop():
                if ev.type != ecodes.EV_KEY:
                    continue
                key = categorize(ev)
                code = key.keycode
                if isinstance(code, (list, tuple)):
                    code = code[0]
                if key.keystate != 1:
                    continue
                if events.in_reset:
                    if code in ["KEY_A", "KEY_C"]:
                        events.start_next_episode = True
                else:
                    if code not in ["KEY_A", "KEY_C"]:
                        continue
                    phase = events.phase
                    if phase == DAggerPhase.CORRECTING:
                        events.request_transition("resume")
                    elif phase == DAggerPhase.PAUSED:
                        events.request_transition("takeover")
                    elif phase == DAggerPhase.AUTONOMOUS:
                        events.request_transition("pause")
        except (FileNotFoundError, PermissionError):
            pass
        except Exception as e:
            logger.warning("Pedal error: %s", e)

    threading.Thread(target=pedal_reader, daemon=True).start()


# ---------------------------------------------------------------------------
# DAgger Strategy
# ---------------------------------------------------------------------------


class DAggerStrategy(RolloutStrategy):
    """Human-in-the-Loop data collection with intervention tagging.

    Uses a formal state machine (see :class:`DAggerPhase`) for phase
    transitions, eliminating impossible states::

        AUTONOMOUS --(SPACE)--> PAUSED --(c)--> CORRECTING --(p)--> AUTONOMOUS
                                       --(p)--> AUTONOMOUS

    Supports both synchronous and RTC inference backends.
    All actions (policy and teleop) flow through the appropriate
    processor pipelines, supporting EE-space recording.

    Intervention frames are tagged with ``intervention=1`` (int64) in
    the dataset to allow downstream BC training to distinguish
    autonomous from human-corrected data.
    """

    config: DAggerStrategyConfig

    def __init__(self, config: DAggerStrategyConfig):
        super().__init__(config)
        self._listener = None
        self._events = DAggerEvents()

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)

        self._listener = _init_dagger_keyboard(self._events)
        _start_pedal_listener(self._events)

        logger.info(
            "DAgger strategy ready (rtc=%s, episodes=%d, episode_time=%.0fs)",
            self._engine.is_rtc,
            self.config.num_episodes,
            self.config.episode_time_s,
        )
        logger.info("Controls: SPACE=pause, c=take control, p=resume, ->=end, <-=redo, ESC=stop")

    def run(self, ctx: RolloutContext) -> None:
        dataset = ctx.dataset
        events = self._events
        teleop = ctx.teleop

        with VideoEncodingManager(dataset):
            try:
                recorded = 0
                while recorded < self.config.num_episodes and not events.stop_recording:
                    log_say(f"Episode {dataset.num_episodes}", self.config.play_sounds)

                    self._run_episode(ctx)

                    if events.rerecord_episode:
                        log_say("Re-recording", self.config.play_sounds)
                        events.rerecord_episode = False
                        events.exit_early = False
                        dataset.clear_episode_buffer()
                        continue

                    dataset.save_episode()
                    recorded += 1

                    if recorded < self.config.num_episodes and not events.stop_recording:
                        _reset_loop(
                            ctx.robot_wrapper,
                            teleop,
                            events,
                            int(ctx.cfg.fps),
                            ctx.teleop_action_processor,
                            ctx.robot_action_processor,
                        )

            finally:
                with contextlib.suppress(Exception):
                    dataset.save_episode()

    def teardown(self, ctx: RolloutContext) -> None:
        log_say("Stop recording", self.config.play_sounds, blocking=True)

        if self._listener is not None and not is_headless():
            self._listener.stop()

        if ctx.dataset is not None:
            ctx.dataset.finalize()
            if ctx.cfg.dataset and ctx.cfg.dataset.push_to_hub:
                ctx.dataset.push_to_hub(
                    tags=ctx.cfg.dataset.tags,
                    private=ctx.cfg.dataset.private,
                )

        self._teardown_hardware(ctx)
        logger.info("DAgger strategy teardown complete")

    # ------------------------------------------------------------------
    # Episode rollout (state machine)
    # ------------------------------------------------------------------

    def _run_episode(self, ctx: RolloutContext) -> None:
        """Run a single DAgger episode with the HIL state machine."""
        engine = self._engine
        cfg = ctx.cfg
        robot = ctx.robot_wrapper
        teleop = ctx.teleop
        dataset = ctx.dataset
        events = self._events
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        stream_online = bool(cfg.dataset.streaming_encoding) if cfg.dataset else False
        record_stride = max(1, cfg.interpolation_multiplier)

        ordered_keys = ctx.ordered_action_keys
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

        if engine.is_rtc:
            engine.resume()

        while timestamp < self.config.episode_time_s:
            loop_start = time.perf_counter()

            if events.exit_early:
                events.exit_early = False
                break

            # --- Process pending phase transition ---
            transition = events.consume_transition()
            if transition is not None:
                old_phase, new_phase = transition
                self._apply_transition(old_phase, new_phase, engine, interpolator, robot, teleop)
                last_action = None

            phase = events.phase

            # --- Get observation ---
            obs = robot.get_observation()
            obs_processed = ctx.robot_observation_processor(obs)
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)

            # --- CORRECTING: human teleop control ---
            if phase == DAggerPhase.CORRECTING:
                teleop_action = teleop.get_action()
                processed_teleop = ctx.teleop_action_processor((teleop_action, obs))
                robot_action_to_send = ctx.robot_action_processor((processed_teleop, obs))
                robot.send_action(robot_action_to_send)
                action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                if record_tick % record_stride == 0:
                    frame = {
                        **obs_frame,
                        **action_frame,
                        "task": task_str,
                        "intervention": np.array([1], dtype=np.int64),
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
                if engine.is_rtc:
                    engine.update_observation(obs_processed)

                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    timestamp = time.perf_counter() - start_t
                    continue

                action_dict = infer_action(
                    engine, obs_processed, obs, ctx, interpolator, ordered_keys, features
                )

                if action_dict is not None:
                    last_action = ctx.robot_action_processor((action_dict, obs))
                    action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                    if record_tick % record_stride == 0:
                        frame = {
                            **obs_frame,
                            **action_frame,
                            "task": task_str,
                            "intervention": np.array([0], dtype=np.int64),
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

        # End of episode: flush any buffered frames
        if engine.is_rtc:
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
            # Pause engine + align teleop to robot position
            if engine.is_rtc:
                engine.pause()
            obs = robot.get_observation()
            robot_pos = {
                k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features
            }
            _teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)
            interpolator.reset()

        elif new_phase == DAggerPhase.CORRECTING:
            # Enable human teleop control
            _teleop_disable_torque(teleop)
            if engine.is_rtc:
                engine.reset()

        elif new_phase == DAggerPhase.AUTONOMOUS:
            # Resume policy from pause or correction
            interpolator.reset()
            engine.reset()
            if engine.is_rtc:
                engine.resume()
