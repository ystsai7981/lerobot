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

import logging
import time
from typing import Any

import torch

from lerobot.common.control_utils import is_headless, predict_action
from lerobot.datasets import VideoEncodingManager
from lerobot.policies.rtc import ActionInterpolator
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import DAggerStrategyConfig
from ..context import RolloutContext
from ..inference import InferenceEngine, _resolve_action_key_order
from . import RolloutStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teleoperator helpers (extracted from examples/hil/hil_utils.py)
# ---------------------------------------------------------------------------


def _teleop_has_motor_control(teleop) -> bool:
    return all(hasattr(teleop, attr) for attr in ("enable_torque", "disable_torque", "write_goal_positions"))


def _teleop_disable_torque(teleop) -> None:
    if hasattr(teleop, "disable_torque"):
        teleop.disable_torque()


def _teleop_enable_torque(teleop) -> None:
    if hasattr(teleop, "enable_torque"):
        teleop.enable_torque()


def _teleop_smooth_move_to(teleop, target_pos: dict, duration_s: float = 2.0, fps: int = 50) -> None:
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


def _reset_loop(robot, teleop, events: dict, fps: int) -> None:
    """Reset period where the human repositions the environment."""
    logger.info("RESET — press any key to enable teleoperation")

    events["in_reset"] = True
    events["start_next_episode"] = False

    obs = robot.get_observation()
    robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features}
    _teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)

    while not events["start_next_episode"] and not events["stop_recording"]:
        precise_sleep(0.05)

    if events["stop_recording"]:
        return

    events["start_next_episode"] = False
    _teleop_disable_torque(teleop)
    logger.info("Teleop enabled — press any key to start episode")

    while not events["start_next_episode"] and not events["stop_recording"]:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        robot.send_action(action)
        precise_sleep(1 / fps - (time.perf_counter() - loop_start))

    events["in_reset"] = False
    events["start_next_episode"] = False
    events["exit_early"] = False
    events["policy_paused"] = False
    events["correction_active"] = False
    events["resume_policy"] = False


def _init_dagger_keyboard():
    """Initialise keyboard listener with DAgger/HIL controls."""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "policy_paused": False,
        "correction_active": False,
        "resume_policy": False,
        "in_reset": False,
        "start_next_episode": False,
    }

    if is_headless():
        logger.warning("Headless environment — keyboard controls unavailable")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if events["in_reset"]:
                if key in [keyboard.Key.space, keyboard.Key.right] or hasattr(key, "char") and key.char == "c":
                    events["start_next_episode"] = True
                elif key == keyboard.Key.esc:
                    events["stop_recording"] = True
                    events["start_next_episode"] = True
            else:
                if key == keyboard.Key.space:
                    if not events["policy_paused"] and not events["correction_active"]:
                        logger.info("PAUSED — press 'c' to take control or 'p' to resume policy")
                        events["policy_paused"] = True
                elif hasattr(key, "char") and key.char == "c":
                    if events["policy_paused"] and not events["correction_active"]:
                        logger.info("Taking control...")
                        events["start_next_episode"] = True
                elif hasattr(key, "char") and key.char == "p":
                    if events["policy_paused"] or events["correction_active"]:
                        logger.info("Resuming policy...")
                        events["resume_policy"] = True
                elif key == keyboard.Key.right:
                    logger.info("End episode")
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    logger.info("Re-record episode")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    logger.info("Stop recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
        except Exception as e:
            logger.debug("Key error: %s", e)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def _start_pedal_listener(events: dict) -> None:
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
                if events["in_reset"]:
                    if code in ["KEY_A", "KEY_C"]:
                        events["start_next_episode"] = True
                else:
                    if code not in ["KEY_A", "KEY_C"]:
                        continue
                    if events["correction_active"]:
                        events["resume_policy"] = True
                    elif events["policy_paused"]:
                        events["start_next_episode"] = True
                    else:
                        events["policy_paused"] = True
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

    State machine:
        AUTONOMOUS -> (SPACE) -> PAUSED -> ('c') -> TAKEOVER -> ('p') -> AUTONOMOUS
                                                             -> (->)  -> save episode

    Supports both synchronous and RTC inference backends.
    All actions (policy and teleop) flow through the appropriate
    processor pipelines, supporting EE-space recording.
    """

    config: DAggerStrategyConfig

    def __init__(self, config: DAggerStrategyConfig):
        super().__init__(config)
        self._engine: InferenceEngine | None = None
        self._listener = None
        self._events: dict[str, Any] = {}

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

        self._listener, self._events = _init_dagger_keyboard()
        _start_pedal_listener(self._events)

        logger.info(
            "DAgger strategy ready (rtc=%s, episodes=%d, episode_time=%.0fs)",
            self._engine.is_rtc,
            self.config.num_episodes,
            self.config.episode_time_s,
        )
        logger.info("Controls: SPACE=pause, c=take control, p=resume, ->=end, <-=redo, ESC=stop")

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        dataset = ctx.dataset
        events = self._events
        teleop = ctx.teleop

        with VideoEncodingManager(dataset):
            try:
                recorded = 0
                while recorded < self.config.num_episodes and not events["stop_recording"]:
                    log_say(f"Episode {dataset.num_episodes}", self.config.play_sounds)

                    self._run_episode(ctx)

                    if events["rerecord_episode"]:
                        log_say("Re-recording", self.config.play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        dataset.clear_episode_buffer()
                        continue

                    dataset.save_episode()
                    recorded += 1

                    if recorded < self.config.num_episodes and not events["stop_recording"]:
                        _reset_loop(ctx.robot_wrapper, teleop, events, int(ctx.cfg.fps))

            finally:
                try:
                    dataset.save_episode()
                except Exception:
                    pass

    def teardown(self, ctx: RolloutContext) -> None:
        log_say("Stop recording", self.config.play_sounds, blocking=True)

        if self._engine is not None:
            self._engine.stop()

        if self._listener is not None and not is_headless():
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

        interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
        control_interval = interpolator.get_control_interval(cfg.fps)
        stream_online = bool(cfg.dataset.streaming_encoding) if cfg.dataset else False
        record_stride = max(1, cfg.interpolation_multiplier)

        policy_action_names = getattr(cfg.policy, "action_feature_names", None)
        ordered_keys = _resolve_action_key_order(
            list(policy_action_names) if policy_action_names else None,
            ctx.action_keys,
        )

        dataset_action_keys = list(dataset.features.get(ACTION, {}).get("names", ctx.action_keys))

        engine.reset()
        _teleop_disable_torque(teleop)

        was_paused = False
        waiting_for_takeover = False
        last_action: dict[str, Any] | None = None
        robot_action: dict[str, Any] = {}
        frame_buffer: list[dict] = []
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task

        timestamp = 0.0
        record_tick = 0
        start_t = time.perf_counter()
        warmup_flushed = False

        if engine.is_rtc:
            engine.resume()

        while timestamp < self.config.episode_time_s:
            loop_start = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                events["policy_paused"] = False
                events["correction_active"] = False
                events["resume_policy"] = False
                break

            # --- Resume from pause/correction ---
            if events["resume_policy"] and (
                events["policy_paused"] or events["correction_active"] or waiting_for_takeover
            ):
                events["resume_policy"] = False
                events["start_next_episode"] = False
                events["policy_paused"] = False
                events["correction_active"] = False
                waiting_for_takeover = False
                was_paused = False
                last_action = None
                interpolator.reset()
                engine.reset()
                if engine.is_rtc:
                    engine.resume()

            # --- Pause: align teleop to robot position ---
            if events["policy_paused"] and not was_paused:
                if engine.is_rtc:
                    engine.pause()
                obs = robot.get_observation()
                robot_pos = {
                    k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features
                }
                _teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)
                events["start_next_episode"] = False
                waiting_for_takeover = True
                was_paused = True
                interpolator.reset()

            # --- Takeover: enable teleop control ---
            if waiting_for_takeover and events["start_next_episode"]:
                _teleop_disable_torque(teleop)
                events["start_next_episode"] = False
                events["correction_active"] = True
                waiting_for_takeover = False
                if engine.is_rtc:
                    engine.reset()

            # --- Get observation ---
            obs = robot.get_observation()
            obs_processed = ctx.robot_observation_processor(obs)
            obs_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

            # --- CORRECTION: human teleop control ---
            if events["correction_active"]:
                teleop_action = teleop.get_action()
                processed_teleop = ctx.teleop_action_processor((teleop_action, obs))
                robot_action_to_send = ctx.robot_action_processor((processed_teleop, obs))
                robot.send_action(robot_action_to_send)
                action_frame = build_dataset_frame(dataset.features, processed_teleop, prefix=ACTION)
                if record_tick % record_stride == 0:
                    frame = {**obs_frame, **action_frame, "task": task_str}
                    if stream_online:
                        dataset.add_frame(frame)
                    else:
                        frame_buffer.append(frame)
                record_tick += 1

            # --- PAUSED: hold position ---
            elif waiting_for_takeover or events["policy_paused"]:
                if last_action:
                    robot.send_action(last_action)

            # --- AUTONOMOUS: policy control ---
            else:
                if engine.is_rtc:
                    engine.update_observation(obs_processed)

                    if cfg.use_torch_compile and not engine.compile_warmup_done.is_set():
                        dt = time.perf_counter() - loop_start
                        if (sleep_t := control_interval - dt) > 0:
                            precise_sleep(sleep_t)
                        timestamp = time.perf_counter() - start_t
                        continue

                    if cfg.use_torch_compile and not warmup_flushed:
                        engine.reset()
                        interpolator.reset()
                        warmup_flushed = True
                        if engine.is_rtc:
                            engine.resume()

                    if interpolator.needs_new_action():
                        action_tensor = engine.consume_rtc_action()
                        if action_tensor is not None:
                            interpolator.add(action_tensor.cpu())

                    interp = interpolator.get()
                    if interp is not None:
                        robot_action = {
                            k: interp[i].item() for i, k in enumerate(ordered_keys) if i < len(interp)
                        }
                        processed = ctx.robot_action_processor((robot_action, obs))
                        robot.send_action(processed)
                        last_action = processed
                        action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
                        if record_tick % record_stride == 0:
                            frame = {**obs_frame, **action_frame, "task": task_str}
                            if stream_online:
                                dataset.add_frame(frame)
                            else:
                                frame_buffer.append(frame)
                        record_tick += 1
                else:
                    # Sync inference
                    if interpolator.needs_new_action():
                        device = get_safe_torch_device(cfg.device)
                        action_tensor = predict_action(
                            observation=obs_frame,
                            policy=ctx.policy,
                            device=device,
                            preprocessor=ctx.preprocessor,
                            postprocessor=ctx.postprocessor,
                            use_amp=ctx.policy.config.use_amp,
                            task=task_str,
                            robot_type=robot.robot_type,
                        )
                        robot_action = make_robot_action(action_tensor, dataset.features)
                        action_t = torch.tensor([robot_action[k] for k in dataset_action_keys])
                        interpolator.add(action_t)

                    interp = interpolator.get()
                    if interp is not None:
                        robot_action = {k: interp[i].item() for i, k in enumerate(dataset_action_keys)}
                        processed = ctx.robot_action_processor((robot_action, obs))
                        robot.send_action(processed)
                        last_action = processed
                        action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
                        if record_tick % record_stride == 0:
                            frame = {**obs_frame, **action_frame, "task": task_str}
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
