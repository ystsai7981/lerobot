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

"""SO100 leader / follower teleop with HIL-SERL-style intervention toggle.

Position-only standalone demo of the leader-arm intervention pattern used by
PR #2596's HIL-SERL training stack (see
``lerobot.processor.LeaderFollowerProcessor`` and
``lerobot.teleoperators.so_leader.SO101LeaderFollower``). Compared to the
verbatim PR #2596 example (which builds the full ``EEReferenceAndDelta`` ->
``EEBoundsAndSafety`` -> ``GripperVelocityToJoint`` -> ``InverseKinematicsRLStep``
pipeline), this version computes the EE delta and the IK target inline against
the follower's *measured* pose every tick. That removes the latched-reference
feedback loop and produces noticeably smoother haptic teleop.

Behaviour:
    * **Following mode** (default): the follower is idle, the leader is
      torque-enabled with low PID gains and haptically tracks the follower.
      The user can grab the leader at any time without fighting the position
      loop.
    * **Intervention mode** (toggled by pressing SPACE): the leader's torque
      is released, the user moves the leader freely, and the follower mirrors
      the leader's end-effector position via ``[delta_x, delta_y, delta_z]``
      deltas, plus a direct gripper passthrough. This matches the action
      space recorded by ``LeaderFollowerProcessor`` during HIL-SERL recording.

Keyboard:
    * ``SPACE`` -- toggle intervention on/off.
    * ``ESC``   -- terminate (treated as failure event).
    * ``s``     -- mark current intervention as success.
    * ``r``     -- request re-record of current episode.
"""

from __future__ import annotations

import time

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101LeaderFollower
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

# Per-axis EE-delta normalisation (metres). The clamped delta is
# ``clip((p_leader - p_follower) / step, -1, 1) * step``, so a single tick is
# bounded by ``step`` in metres. Keep small for safe motion.
EE_STEP_SIZES = {"x": 0.010, "y": 0.010, "z": 0.010}

# Workspace bounds (metres) - tight box around the rest pose to keep the
# follower from running into joint limits during the demo. Adjust to your
# workspace.
EE_BOUNDS = {
    "min": np.array([-0.20, -0.30, 0.02]),
    "max": np.array([0.30, 0.30, 0.40]),
}

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
# https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
URDF_PATH = "./SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"

# Set these to the actual ports on your machine.
FOLLOWER_PORT = "/dev/usb_follower_arm_a"
LEADER_PORT = "/dev/usb_leader_arm_a"


def _joints_dict_to_array(joints: dict[str, float], motor_names: list[str]) -> np.ndarray:
    return np.array([joints[f"{m}.pos"] for m in motor_names], dtype=float)


def _array_to_joints_dict(arr: np.ndarray, motor_names: list[str]) -> dict[str, float]:
    return {f"{m}.pos": float(v) for m, v in zip(motor_names, arr, strict=True)}


def main() -> None:
    follower_config = SO100FollowerConfig(port=FOLLOWER_PORT, id="my_follower_arm", use_degrees=True)
    leader_config = SO101LeaderConfig(
        port=LEADER_PORT,
        id="my_leader_arm",
        use_degrees=True,
        leader_follower_mode=True,
        use_gripper=True,
    )

    follower = SO100Follower(follower_config)
    leader = SO101LeaderFollower(leader_config)

    follower_motor_names = list(follower.bus.motors.keys())
    leader_motor_names = list(leader.bus.motors.keys())

    follower_kinematics = RobotKinematics(
        urdf_path=URDF_PATH, target_frame_name=TARGET_FRAME, joint_names=follower_motor_names
    )
    leader_kinematics = RobotKinematics(
        urdf_path=URDF_PATH, target_frame_name=TARGET_FRAME, joint_names=leader_motor_names
    )

    follower.connect()
    leader.connect()

    print("Starting leader-follower intervention demo...")
    print("  - Press SPACE to toggle intervention.")
    print("  - Press ESC to terminate, 's' for success, 'r' to re-record.")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Read both arms.
            follower_obs = follower.get_observation()
            follower_joints_dict = {f"{m}.pos": float(follower_obs[f"{m}.pos"]) for m in follower_motor_names}
            leader_joints_dict = leader.get_action()

            # 2. Haptic follow: push follower joints back to the leader. The
            # leader's ``send_action`` gates motor writes on its intervention
            # state internally (torque on while following, off while
            # intervening), so this call is safe in both modes.
            leader.send_action(follower_joints_dict)

            # 3. Pull teleop events from the leader's keyboard listener.
            events = leader.get_teleop_events()
            if events.get(TeleopEvents.TERMINATE_EPISODE):
                print("Termination requested -- exiting.")
                break

            is_intervention = events.get(TeleopEvents.IS_INTERVENTION, False)

            if is_intervention:
                # 4a. INTERVENTION: take normalised position-only delta against
                # the follower's *measured* pose every tick (no latched
                # reference, no compounding lag), integrate onto the follower's
                # current EE pose, clip to the workspace, then IK.
                leader_arr = _joints_dict_to_array(leader_joints_dict, leader_motor_names)
                follower_arr = _joints_dict_to_array(follower_joints_dict, follower_motor_names)

                p_leader = leader_kinematics.forward_kinematics(leader_arr)[:3, 3]
                p_follower_mat = follower_kinematics.forward_kinematics(follower_arr)
                p_follower = p_follower_mat[:3, 3]

                step_vec = np.array([EE_STEP_SIZES["x"], EE_STEP_SIZES["y"], EE_STEP_SIZES["z"]], dtype=float)
                raw_delta = p_leader - p_follower
                delta_norm = np.clip(raw_delta / step_vec, -1.0, 1.0)
                delta_m = delta_norm * step_vec

                target_pose = p_follower_mat.copy()
                target_pose[:3, 3] = np.clip(p_follower + delta_m, EE_BOUNDS["min"], EE_BOUNDS["max"])

                # IK -> joint-space goal for the follower's arm chain. Position
                # only (orientation_weight=0.0) keeps it stable under the
                # rotation-noise that would otherwise come from leader FK.
                target_joints = follower_kinematics.inverse_kinematics(
                    current_joint_pos=follower_arr,
                    desired_ee_pose=target_pose,
                    orientation_weight=0.0,
                )
                follower_action = _array_to_joints_dict(target_joints, follower_motor_names)
                # Gripper passthrough: leader gripper position drives follower
                # gripper directly (no IK).
                follower_action["gripper.pos"] = float(leader_joints_dict.get("gripper.pos", 50.0))
                follower.send_action(follower_action)

            # 4b. FOLLOWING: leave the follower alone -- the leader haptically
            # tracks it via the ``leader.send_action`` call above. In real
            # HIL-SERL training this is where the policy would step the
            # follower forward.

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
