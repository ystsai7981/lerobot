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
Processor step for using a leader arm as the HIL-SERL intervention device.

Position-only port of the leader/follower control mode (no rotation): the leader
arm acts as a 4-D end-effector delta source ``[dx, dy, dz, gripper]`` for the
existing ``InterventionActionProcessorStep`` overriding pipeline.

The teleop_action returned by the leader is a flat dictionary of joint angles
(degrees) like ``{"shoulder_pan.pos": ..., ..., "gripper.pos": ...}``. This step
converts that into a normalised EE-delta dictionary by:

1. Running forward kinematics on the leader joints -> ``p_leader`` (xyz, m).
2. Running forward kinematics on the follower joints (read from the env
   transition's observation / complementary data) -> ``p_follower`` (xyz, m).
3. Normalising ``p_leader - p_follower`` by ``end_effector_step_sizes`` and
   clipping to ``[-1, 1]`` (matches the gamepad / keyboard EE convention).
4. Mapping the leader gripper position ``[0, 100]`` to the discrete
   ``{0=close, 1=stay, 2=open}`` action used by the SO follower.

The output is written back to ``complementary_data["teleop_action"]`` so the
rest of the action pipeline (``InterventionActionProcessorStep`` ->
``MapTensorToDeltaActionDictStep`` -> IK) is unchanged.

Additionally, when an optional ``teleop_device`` reference is provided, this
step also pushes the follower's raw joint positions back to the leader via
``teleop_device.send_action(follower_joints)`` every tick. Combined with
:class:`SOLeaderFollower.send_action`, this implements the **haptic follow**
behaviour from https://github.com/huggingface/lerobot/pull/2596: the leader
mimics the follower while the human is hands-off, then drops torque the
moment intervention is toggled so the user can grab and steer it.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.model import RobotKinematics
from lerobot.types import EnvTransition, TransitionKey

from .pipeline import ProcessorStep, ProcessorStepRegistry

logger = logging.getLogger(__name__)

TELEOP_ACTION_KEY = "teleop_action"
RAW_JOINT_POSITIONS_KEY = "raw_joint_positions"
GRIPPER_KEY = "gripper"

# Leader gripper is in [0, 100] when calibrated.
LEADER_GRIPPER_OPEN_DEFAULT = 60.0
LEADER_GRIPPER_CLOSE_DEFAULT = 30.0

# Discrete gripper command convention (matches GripperVelocityToJoint).
GRIPPER_CLOSE = 0.0
GRIPPER_STAY = 1.0
GRIPPER_OPEN = 2.0


def _joint_dict_to_array(joint_dict: dict[str, float], motor_names: list[str]) -> np.ndarray | None:
    """Pull joint positions in ``motor_names`` order from a ``"<motor>.pos"`` dict.

    Returns ``None`` if any motor is missing.
    """
    out = np.zeros(len(motor_names), dtype=float)
    for i, name in enumerate(motor_names):
        v = joint_dict.get(f"{name}.pos")
        if v is None:
            return None
        out[i] = float(v)
    return out


@ProcessorStepRegistry.register("leader_arm_intervention")
@dataclass
class LeaderArmInterventionStep(ProcessorStep):
    """Convert leader joint positions in ``teleop_action`` into a 4-D EE-delta dict.

    This step is intended to run **between** ``AddTeleopActionAsComplimentaryDataStep``
    (which populates ``complementary_data["teleop_action"]`` with raw leader joint
    angles) and ``InterventionActionProcessorStep`` (which expects a delta dict).

    Attributes:
        kinematics: Robot kinematic model shared with the follower; used for FK
            on both the leader arm and the follower arm. Both arms must use the
            same URDF joint order.
        motor_names: Ordered joint names matching ``kinematics.joint_names``,
            used to slice joint dicts.
        end_effector_step_sizes: Per-axis normalisation in metres, e.g.
            ``{"x": 0.025, "y": 0.025, "z": 0.025}``. The clamped delta is
            ``(p_leader - p_follower) / step_size``.
        use_gripper: When ``True``, append a discrete gripper command derived from
            the leader gripper joint to the output dict.
        leader_gripper_open: Threshold (>= ) above which the leader gripper is
            considered ``open`` -> command ``2``.
        leader_gripper_close: Threshold (<= ) below which the leader gripper is
            considered ``closed`` -> command ``0``.
        teleop_device: Optional reference to the leader teleoperator. When set
            and the device implements ``send_action(action_dict)``, this step
            pushes the follower's raw joints to it every tick to drive haptic
            follow. The teleop is responsible for gating actual motor writes on
            its own intervention state (see :class:`SOLeaderFollower`).
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    end_effector_step_sizes: dict[str, float]
    use_gripper: bool = True
    leader_gripper_open: float = LEADER_GRIPPER_OPEN_DEFAULT
    leader_gripper_close: float = LEADER_GRIPPER_CLOSE_DEFAULT
    teleop_device: Any = None

    _initial_follower_joints: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})

        # Haptic follow: push follower joints to the leader every step (whether
        # or not we have a usable leader action this tick). The leader's own
        # send_action gates writes on its intervention state.
        follower_joints_dict = self._read_follower_joints_dict(transition, complementary_data)
        if follower_joints_dict is not None:
            self._push_haptic_follow(follower_joints_dict)

        leader_joints_dict = complementary_data.get(TELEOP_ACTION_KEY)
        if not isinstance(leader_joints_dict, dict):
            # Nothing to convert (e.g. teleop disconnected). Leave transition untouched.
            return new_transition

        if not any(k.endswith(".pos") for k in leader_joints_dict):
            # Already in EE-delta form (or unrecognised); skip.
            return new_transition

        follower_joints = (
            _joint_dict_to_array(follower_joints_dict, self.motor_names)
            if follower_joints_dict is not None
            else None
        )
        leader_joints = _joint_dict_to_array(leader_joints_dict, self.motor_names)

        if follower_joints is None or leader_joints is None:
            # Cannot compute delta this step; expose a zero-action so downstream
            # InterventionActionProcessorStep does not propagate stale joints.
            complementary_data[TELEOP_ACTION_KEY] = self._zero_action()
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
            return new_transition

        p_leader = self.kinematics.forward_kinematics(leader_joints)[:3, 3]
        p_follower = self.kinematics.forward_kinematics(follower_joints)[:3, 3]

        delta = p_leader - p_follower
        delta_norm = np.array(
            [
                delta[0] / max(self.end_effector_step_sizes.get("x", 1.0), 1e-6),
                delta[1] / max(self.end_effector_step_sizes.get("y", 1.0), 1e-6),
                delta[2] / max(self.end_effector_step_sizes.get("z", 1.0), 1e-6),
            ],
            dtype=float,
        )
        delta_norm = np.clip(delta_norm, -1.0, 1.0)

        teleop_action: dict[str, float] = {
            "delta_x": float(delta_norm[0]),
            "delta_y": float(delta_norm[1]),
            "delta_z": float(delta_norm[2]),
        }

        if self.use_gripper:
            leader_gripper = float(leader_joints_dict.get(f"{GRIPPER_KEY}.pos", 50.0))
            teleop_action[GRIPPER_KEY] = self._discretise_gripper(leader_gripper)

        complementary_data[TELEOP_ACTION_KEY] = teleop_action
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _read_follower_joints_dict(
        self, transition: EnvTransition, complementary_data: dict[str, Any]
    ) -> dict[str, float] | None:
        """Best-effort read of the follower joints from the transition.

        Tries (in order):
        1. ``complementary_data["raw_joint_positions"]`` (set after env.step).
        2. ``transition[OBSERVATION]`` if it is a flat ``"<motor>.pos"`` dict
           (this is the convention used by ``step_env_and_process_transition``
           when staging an action transition).

        Returns the source dict if all expected motors are present, else
        ``None``. We return the *dict* (not the array) because we want to feed
        it back to ``teleop_device.send_action`` for haptic follow.
        """
        raw = complementary_data.get(RAW_JOINT_POSITIONS_KEY)
        if isinstance(raw, dict) and all(f"{m}.pos" in raw for m in self.motor_names):
            return raw  # type: ignore[return-value]

        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict) and all(f"{m}.pos" in observation for m in self.motor_names):
            return observation  # type: ignore[return-value]

        return None

    def _push_haptic_follow(self, follower_joints_dict: dict[str, float]) -> None:
        """Send the follower's joints back to the leader for haptic follow.

        Errors are logged once and swallowed -- a failed haptic update must
        never break the policy / learner loop.
        """
        if self.teleop_device is None:
            return
        send_action = getattr(self.teleop_device, "send_action", None)
        if send_action is None:
            return
        try:
            send_action(follower_joints_dict)
        except NotImplementedError:
            # Plain SOLeader / unsupported teleop -- silently disable haptic follow.
            self.teleop_device = None
        except Exception as e:  # pragma: no cover - hardware path
            logger.warning(f"[LeaderArmInterventionStep] haptic follow failed: {e}")

    def _discretise_gripper(self, leader_gripper_pos: float) -> float:
        """Map a leader gripper position in ``[0, 100]`` to ``{0, 1, 2}``."""
        if leader_gripper_pos >= self.leader_gripper_open:
            return GRIPPER_OPEN
        if leader_gripper_pos <= self.leader_gripper_close:
            return GRIPPER_CLOSE
        return GRIPPER_STAY

    def _zero_action(self) -> dict[str, float]:
        out: dict[str, float] = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}
        if self.use_gripper:
            out[GRIPPER_KEY] = GRIPPER_STAY
        return out

    def get_config(self) -> dict[str, Any]:
        # `kinematics` and `teleop_device` are runtime objects (not JSON-serializable)
        # and are re-injected by `gym_manipulator.make_processors`, so they are
        # intentionally omitted from the saved config.
        return {
            "motor_names": list(self.motor_names),
            "end_effector_step_sizes": dict(self.end_effector_step_sizes),
            "use_gripper": self.use_gripper,
            "leader_gripper_open": self.leader_gripper_open,
            "leader_gripper_close": self.leader_gripper_close,
        }

    def reset(self) -> None:
        self._initial_follower_joints = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
