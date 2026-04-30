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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class SOLeaderHILBase:
    """Composition config: SO101 leader as the action source + keyboard for HIL-SERL events."""

    # Leader USB port (e.g. /dev/ttyACM1)
    port: str
    use_degrees: bool = True

    # Whether to expose a gripper command in the action (must match env.processor.gripper.use_gripper)
    use_gripper: bool = True

    # Forward kinematics for the leader. SO101 leader and follower have the same URDF, so the
    # follower's URDF works fine here.
    urdf_path: str = "assets/so101/so101_new_calib.urdf"
    target_frame_name: str = "gripper_frame_link"

    # Per-axis EE step size used to normalise the leader-EE delta into the [-1, 1] action range.
    # MUST match env.processor.inverse_kinematics.end_effector_step_sizes — otherwise the human
    # intervention scale will not match the policy's scale.
    end_effector_step_sizes: dict = field(
        default_factory=lambda: {"x": 0.025, "y": 0.025, "z": 0.025}
    )

    # Discrete gripper command thresholds, in degrees of leader-gripper change between two
    # consecutive frames. Above OPEN-threshold -> command OPEN; below CLOSE-threshold -> CLOSE;
    # otherwise STAY. Tune if your leader gripper is twitchy or sluggish.
    gripper_open_threshold_deg: float = 1.0
    gripper_close_threshold_deg: float = -1.0


@TeleoperatorConfig.register_subclass("so101_leader_hil")
@dataclass
class SOLeaderHILTeleopConfig(TeleoperatorConfig, SOLeaderHILBase):
    pass


SO101LeaderHILConfig = SOLeaderHILTeleopConfig
