#!/usr/bin/env python

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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class SOLeaderConfig:
    """Base configuration class for SO Leader teleoperators."""

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True

    # When True, the SO leader is wrapped in `SOLeaderFollower`, which adds a
    # keyboard listener for HIL-SERL intervention events (SPACE toggles
    # intervention, S/R/Q signal success/rerecord/fail) and reports a 4-D
    # EE-delta action space via `action_features`. The raw leader joints are
    # still returned by `get_action()` and converted downstream by
    # `LeaderArmInterventionStep`.
    leader_follower_mode: bool = False

    # When `leader_follower_mode` is enabled, include the gripper in the
    # 4-D delta action space announced by `action_features`.
    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    pass


SO100LeaderConfig = SOLeaderTeleopConfig
SO101LeaderConfig = SOLeaderTeleopConfig
