#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
SO Leader teleoperator extended with HIL-SERL intervention events and haptic follow.

This wrapper around :class:`SOLeader` keeps the underlying joint-reading behaviour
intact (so :func:`get_action` returns ``{"<motor>.pos": float}``) while adding:

* A pynput keyboard listener that toggles intervention with SPACE and emits
  ``success`` / ``rerecord`` / ``fail`` signals via S / R / Q keys, mirroring
  :class:`KeyboardEndEffectorTeleop`.
* A :func:`get_teleop_events` method satisfying the
  :class:`HasTeleopEvents` protocol consumed by ``AddTeleopEventsAsInfoStep``.
* An :func:`action_features` override that announces the 4-D
  ``[delta_x, delta_y, delta_z, gripper]`` space the leader will project into
  via :class:`LeaderArmInterventionStep` -- this is what ends up recorded by
  ``LeRobotDataset`` in HIL-SERL ``record`` mode.
* :func:`send_action` for **haptic follow**: when the human is not intervening,
  the leader is torque-enabled and tracks the follower's joint positions so the
  user can grab it at any time and seamlessly take over (mirrors the design from
  https://github.com/huggingface/lerobot/pull/2596). When intervention is
  toggled on, leader torque is disabled so the user can move it freely.
* Lower position-loop gains on :func:`connect` (``P=16, I=0, D=16``) so the
  haptic follow does not jerk the user's hand when grabbing the leader.
* Bus-control primitives (:func:`enable_torque`, :func:`disable_torque`,
  :func:`write_goal_positions`) and a :func:`smooth_move_to` helper. These
  satisfy the ``teleop_has_motor_control`` capability gate in
  ``examples/hil/hil_utils.py``, so the BC-style HIL data collector
  (``examples/hil/hil_data_collection.py``) can drive an SO leader the same way
  it drives the OpenArm Mini -- pause / smooth-mirror to follower / take over.

The joint-to-EE-delta conversion does **not** happen here; it is performed by
:class:`LeaderArmInterventionStep` in the action processor pipeline so the
leader stays a pure I/O device.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available

from ..utils import TeleopEvents
from .config_so_leader import SOLeaderTeleopConfig
from .so_leader import SOLeader

LEADER_FOLLOWER_P_GAIN = 16
LEADER_FOLLOWER_I_GAIN = 0
LEADER_FOLLOWER_D_GAIN = 16

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = _pynput_available
keyboard: Any = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logger.info("No DISPLAY set. Skipping pynput import for SOLeaderFollower.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard  # type: ignore[no-redef]
    except Exception as e:  # pragma: no cover - hardware path
        PYNPUT_AVAILABLE = False
        logger.info(f"Could not import pynput: {e}")


class SOLeaderFollower(SOLeader):
    """SO leader teleop with intervention/success/rerecord keyboard signals."""

    config_class = SOLeaderTeleopConfig
    name = "so_leader_follower"

    def __init__(self, config: SOLeaderTeleopConfig):
        super().__init__(config)

        self._is_intervention: bool = False
        self._success: bool = False
        self._rerecord: bool = False
        self._terminate_episode: bool = False
        self._listener: Any = None

        # Haptic follow state (mirrors `is_intervening` / `leader_torque_enabled`
        # in https://github.com/huggingface/lerobot/pull/2596 SO101LeaderFollower).
        self._leader_torque_enabled: bool = True
        self._last_follower_pos: np.ndarray | None = None

    @property
    def action_features(self) -> dict[str, Any]:
        """Announce the 4-D EE-delta action space recorded by the dataset.

        The leader still produces raw joints from :func:`get_action`; this
        property describes what downstream processors emit and what the dataset
        layer should reserve in the ``action`` column for HIL-SERL recordings.
        """
        names: dict[str, int] = {"delta_x": 0, "delta_y": 1, "delta_z": 2}
        shape = (3,)
        if getattr(self.config, "use_gripper", True):
            names["gripper"] = 3
            shape = (4,)
        return {"dtype": "float32", "shape": shape, "names": names}

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        self._configure_leader_follower_gains()
        self._start_keyboard_listener()
        logger.info(
            "[SOLeaderFollower] connected. Press SPACE to toggle intervention, "
            "'s' for success, 'r' for re-record, 'q' to terminate."
        )

    def _configure_leader_follower_gains(self) -> None:
        """Lower position-loop gains so haptic follow does not yank the user.

        Mirrors the gains used by the SO101LeaderFollower in PR #2596 — high
        default gains make the leader fight the user's hand when they grab it
        between interventions.
        """
        for motor in self.bus.motors:
            try:
                self.bus.write("P_Coefficient", motor, LEADER_FOLLOWER_P_GAIN)
                self.bus.write("I_Coefficient", motor, LEADER_FOLLOWER_I_GAIN)
                self.bus.write("D_Coefficient", motor, LEADER_FOLLOWER_D_GAIN)
            except Exception as e:  # pragma: no cover - hardware path
                logger.warning(f"[SOLeaderFollower] could not set PID gains for '{motor}': {e}")

    def _start_keyboard_listener(self) -> None:
        if not PYNPUT_AVAILABLE:
            logger.info("pynput unavailable; SOLeaderFollower keyboard events disabled.")
            return

        def on_press(key: Any) -> None:
            try:
                if key == keyboard.Key.space:
                    self._is_intervention = not self._is_intervention
                    logger.info(f"[SOLeaderFollower] intervention -> {self._is_intervention}")
                    return
                char = getattr(key, "char", None)
                if char is None:
                    return
                if char == "s":
                    self._success = True
                    self._terminate_episode = True
                elif char == "r":
                    self._rerecord = True
                    self._terminate_episode = True
                elif char == "q":
                    self._terminate_episode = True
            except Exception:  # nosec B110
                # Never let the listener thread crash on weird keys.
                pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def enable_torque(self) -> None:
        """Enable position-loop torque on every motor.

        Exposed alongside :func:`disable_torque` and :func:`write_goal_positions`
        so this teleop satisfies the ``teleop_has_motor_control`` gate used by
        ``examples/hil/hil_data_collection.py`` (mirrors the OpenArm Mini API).
        Errors are logged and swallowed -- the loop must keep ticking even if a
        single haptic update fails.
        """
        if not self.is_connected:
            return
        try:
            self.bus.sync_write("Torque_Enable", 1)
            self._leader_torque_enabled = True
        except Exception as e:  # pragma: no cover - hardware path
            logger.warning(f"[SOLeaderFollower] could not enable leader torque: {e}")

    def disable_torque(self) -> None:
        """Disable position-loop torque so the user can move the leader freely."""
        if not self.is_connected:
            return
        try:
            self.bus.sync_write("Torque_Enable", 0)
            self._leader_torque_enabled = False
        except Exception as e:  # pragma: no cover - hardware path
            logger.warning(f"[SOLeaderFollower] could not disable leader torque: {e}")

    def write_goal_positions(self, positions: dict[str, float]) -> None:
        """Push goal positions to the leader bus (no torque toggling).

        Accepts dataset-style keys ``{"<motor>.pos": deg}`` (matches what
        :func:`get_action` produces and what :func:`smooth_move_to` and
        :func:`send_action` consume) -- bare motor names are also tolerated
        for parity with :class:`OpenArmMini.write_goal_positions`.
        """
        if not self.is_connected:
            return
        goal_pos: dict[str, float] = {}
        for motor in self.bus.motors:
            if f"{motor}.pos" in positions:
                goal_pos[motor] = float(positions[f"{motor}.pos"])
            elif motor in positions:
                goal_pos[motor] = float(positions[motor])
        if not goal_pos:
            return
        try:
            self.bus.sync_write("Goal_Position", goal_pos)
        except Exception as e:  # pragma: no cover - hardware path
            logger.warning(f"[SOLeaderFollower] could not push goal position to leader: {e}")

    def smooth_move_to(
        self,
        target_pos: dict[str, float],
        duration_s: float = 2.0,
        fps: int = 50,
    ) -> None:
        """Linearly ramp the leader from its current pose to ``target_pos``.

        Mirrors the ``teleop_smooth_move_to`` helper from
        ``examples/hil/hil_utils.py`` so the leader can be safely re-engaged
        without yanking the user's hand -- typical use is right after
        :func:`connect` (or whenever the leader and follower drift apart, e.g.
        on episode reset). Blocks for ``duration_s`` seconds.
        """
        if not self.is_connected:
            return

        steps = max(int(duration_s * fps), 1)
        try:
            current = self.get_action()
        except Exception as e:  # pragma: no cover - hardware path
            logger.warning(f"[SOLeaderFollower] smooth_move_to could not read current pose: {e}")
            return

        self.enable_torque()
        if not self._leader_torque_enabled:
            return

        for step in range(steps + 1):
            t = step / steps
            interp = {}
            for key, current_val in current.items():
                if key in target_pos:
                    interp[key] = current_val * (1.0 - t) + float(target_pos[key]) * t
                else:
                    interp[key] = current_val
            self.write_goal_positions(interp)
            time.sleep(1.0 / fps)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # When the user has just toggled into intervention, make sure leader
        # torque is OFF so they can move it without fighting the position loop.
        if self._is_intervention and self._leader_torque_enabled:
            self.disable_torque()
        return super().get_action()

    def send_action(self, action: dict[str, float]) -> None:  # type: ignore[override]
        """Mirror the follower's joint positions on the leader (haptic follow).

        This is called every step from the action pipeline (typically by
        :class:`LeaderArmInterventionStep`) with the follower's raw joint
        positions ``{"<motor>.pos": float}``. While the user is **not**
        intervening the leader is torque-enabled and tracks the follower so the
        operator can grab it at any time and continue motion smoothly. As soon
        as the user toggles intervention on (SPACE), torque is dropped in
        :func:`get_action` so the human can move the leader freely.

        Args:
            action: Dictionary of follower motor positions, ``{motor.pos: deg}``.
        """
        if not self.is_connected:
            return

        try:
            self._last_follower_pos = np.array(
                [float(action.get(f"{m}.pos", 0.0)) for m in self.bus.motors],
                dtype=float,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"[SOLeaderFollower] could not extract follower joints: {e}")
            return

        if self._is_intervention:
            return

        if not self._leader_torque_enabled:
            self.enable_torque()
            if not self._leader_torque_enabled:
                return

        self.write_goal_positions(action)

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        events = {
            TeleopEvents.IS_INTERVENTION: self._is_intervention,
            TeleopEvents.TERMINATE_EPISODE: self._terminate_episode,
            TeleopEvents.SUCCESS: self._success,
            TeleopEvents.RERECORD_EPISODE: self._rerecord,
        }
        # Edge-trigger the episode-control flags so the next read does not
        # re-fire the same event for several frames.
        self._success = False
        self._rerecord = False
        self._terminate_episode = False
        return events

    def reset(self) -> None:
        self._is_intervention = False
        self._success = False
        self._rerecord = False
        self._terminate_episode = False
        self._leader_torque_enabled = True
        self._last_follower_pos = None

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._listener is not None:
            with contextlib.suppress(Exception):
                self._listener.stop()
            self._listener = None
        super().disconnect()
