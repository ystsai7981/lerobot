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

import logging
from typing import Any

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.import_utils import require_package

from ..so_leader.config_so_leader import SO101LeaderConfig
from ..so_leader.so_leader import SO101Leader
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_so_leader_hil import SOLeaderHILTeleopConfig

logger = logging.getLogger(__name__)

# Joint order matching SO101 follower / leader URDF
_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class _GripperAction:
    """Discrete gripper command codes — must match teleop_gamepad.GripperAction."""

    CLOSE = 0
    STAY = 1
    OPEN = 2


class SO101LeaderHIL(Teleoperator):
    """SO101 leader teleop with keyboard event listener — drop-in replacement for
    GamepadTeleop in HIL-SERL.

    Action shape (matches GamepadTeleop):
        - delta_x, delta_y, delta_z : float in [-1, 1]
            Per-frame leader-EE delta, normalised by ``end_effector_step_sizes``.
            HIL-SERL's ``EEReferenceAndDelta`` will multiply back by step_size,
            so the follower mirrors the leader's motion.
        - gripper : int (CLOSE=0 / STAY=1 / OPEN=2)
            Derived from leader gripper movement between frames.

    Keyboard events (pynput, captured globally):
        [s]      mark episode SUCCESS  (reward=1, terminate)
        [esc]    mark episode FAILURE  (reward=0, terminate)
        [r]      RERECORD episode (discard, run again)
        [space]  toggle INTERVENTION on/off (let leader override policy)

    Why this exists: the upstream HIL-SERL pipeline only accepts teleops that
    expose ``get_teleop_events()`` (currently GamepadTeleop and
    KeyboardEndEffectorTeleop). The plain SO101 leader does not, so users with a
    leader arm could not drive HIL-SERL out of the box. This class composes a
    leader read + a keyboard listener and reports them through both contracts.
    """

    config_class = SOLeaderHILTeleopConfig
    name = "so101_leader_hil"

    def __init__(self, config: SOLeaderHILTeleopConfig):
        require_package("placo", extra="placo-dep")
        super().__init__(config)
        self.config = config

        # Internal SO101 leader for joint reads (we wrap, not subclass, to keep
        # calibration and registration independent).
        leader_config = SO101LeaderConfig(
            port=config.port,
            use_degrees=config.use_degrees,
            id=config.id,
            calibration_dir=config.calibration_dir,
        )
        self.leader = SO101Leader(leader_config)

        # Step sizes used to normalise leader-EE delta -> [-1, 1] action range.
        self._step_sizes = np.array(
            [
                float(config.end_effector_step_sizes.get("x", 0.025)),
                float(config.end_effector_step_sizes.get("y", 0.025)),
                float(config.end_effector_step_sizes.get("z", 0.025)),
            ],
            dtype=float,
        )

        # Lazy: only initialised on connect()
        self._kinematics = None
        self._kb_listener = None

        # Runtime state
        self._last_ee_pos: np.ndarray | None = None
        self._last_gripper_pos: float = 0.0
        self._intervention_active: bool = False
        self._was_intervention_active: bool = False
        self._success_pending: bool = False
        self._failure_pending: bool = False
        self._rerecord_pending: bool = False

    # ── Teleoperator contract ────────────────────────────────────────────────

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.leader.is_connected and self._kb_listener is not None and self._kinematics is not None

    @property
    def is_calibrated(self) -> bool:
        return self.leader.is_calibrated

    def calibrate(self) -> None:
        self.leader.calibrate()

    def configure(self) -> None:
        self.leader.configure()

    def connect(self, calibrate: bool = True) -> None:
        self.leader.connect(calibrate=calibrate)
        # Lazy-import RobotKinematics to keep placo as a soft dep at module-load time.
        from lerobot.model.kinematics import RobotKinematics

        self._kinematics = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
            joint_names=_MOTOR_NAMES,
        )
        self._start_keyboard_listener()
        logger.info(
            "%s connected — leader on %s, FK ready (target_frame=%s), keyboard armed.",
            self,
            self.config.port,
            self.config.target_frame_name,
        )

    def disconnect(self) -> None:
        if self._kb_listener is not None:
            try:
                self._kb_listener.stop()
            except Exception:
                logger.exception("Error stopping keyboard listener")
            self._kb_listener = None
        if self.leader.is_connected:
            self.leader.disconnect()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # No haptic feedback on SO101 leader.
        pass

    def get_action(self) -> RobotAction:
        # 1) Read the 6 joint positions from the leader.
        leader_action = self.leader.get_action()
        joints = np.array(
            [leader_action[f"{m}.pos"] for m in _MOTOR_NAMES],
            dtype=float,
        )

        # 2) Forward kinematics to get the leader EE pose in robot-base frame.
        T = self._kinematics.forward_kinematics(joints)
        current_ee = np.array(T[:3, 3], dtype=float)
        current_gripper = float(joints[5])

        # 3) Reset reference whenever we are NOT actively intervening, or at the
        #    rising edge of intervention, so that re-engaging the leader never
        #    produces a sudden jump on the follower.
        rising_edge = self._intervention_active and not self._was_intervention_active
        if rising_edge or self._last_ee_pos is None or not self._intervention_active:
            self._last_ee_pos = current_ee.copy()
            self._last_gripper_pos = current_gripper

        # 4) Normalise per-frame EE delta into [-1, 1] using the configured step sizes,
        #    matching the policy's output range.
        delta_xyz = (current_ee - self._last_ee_pos) / self._step_sizes
        delta_xyz = np.clip(delta_xyz, -1.0, 1.0)

        # 5) Map leader gripper movement to a discrete OPEN/CLOSE/STAY command.
        gripper_cmd = _GripperAction.STAY
        if self.config.use_gripper:
            d = current_gripper - self._last_gripper_pos
            if d >= self.config.gripper_open_threshold_deg:
                gripper_cmd = _GripperAction.OPEN
            elif d <= self.config.gripper_close_threshold_deg:
                gripper_cmd = _GripperAction.CLOSE

        # 6) Update state for next iteration.
        self._last_ee_pos = current_ee.copy()
        self._last_gripper_pos = current_gripper
        self._was_intervention_active = self._intervention_active

        result: dict[str, Any] = {
            "delta_x": float(delta_xyz[0]),
            "delta_y": float(delta_xyz[1]),
            "delta_z": float(delta_xyz[2]),
        }
        if self.config.use_gripper:
            result["gripper"] = int(gripper_cmd)
        return result

    # ── HasTeleopEvents contract (consumed by AddTeleopEventsAsInfoStep) ────

    def get_teleop_events(self) -> dict[str, Any]:
        success = self._success_pending
        self._success_pending = False
        failure = self._failure_pending
        self._failure_pending = False
        rerecord = self._rerecord_pending
        self._rerecord_pending = False

        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_active,
            TeleopEvents.TERMINATE_EPISODE: bool(success or failure or rerecord),
            TeleopEvents.SUCCESS: bool(success),
            TeleopEvents.RERECORD_EPISODE: bool(rerecord),
        }

    # ── Internals ───────────────────────────────────────────────────────────

    def _start_keyboard_listener(self) -> None:
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.KeyCode.from_char("s"):
                    self._success_pending = True
                elif key == keyboard.KeyCode.from_char("r"):
                    self._rerecord_pending = True
                elif key == keyboard.Key.esc:
                    self._failure_pending = True
                elif key == keyboard.Key.space:
                    self._intervention_active = not self._intervention_active
                    logger.info(
                        "Intervention %s",
                        "ENABLED (leader overrides policy)"
                        if self._intervention_active
                        else "DISABLED (policy back in control)",
                    )
            except AttributeError:
                # Special keys without a char attribute fall through here.
                pass

        self._kb_listener = keyboard.Listener(on_press=on_press)
        self._kb_listener.start()

        bar = "─" * 60
        print(bar)
        print("SO101 Leader + Keyboard teleop ready.")
        print("  Move the LEADER arm to drive the follower (only when intervening).")
        print("Keyboard:")
        print("  [s]      mark episode SUCCESS  (reward=1, terminate)")
        print("  [esc]    mark episode FAILURE  (reward=0, terminate)")
        print("  [r]      RERECORD episode (discard, run again)")
        print("  [space]  toggle INTERVENTION (leader overrides policy)")
        print(bar)
