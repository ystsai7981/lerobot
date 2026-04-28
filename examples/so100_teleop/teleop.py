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

"""
Simple SO100/SO101 leader-follower teleoperation with spacebar intervention toggle.

Modes:
  - Default (not intervening): leader arm tracks follower position (torque ON on leader)
  - Intervention (SPACE pressed): follower arm tracks leader position (torque OFF on leader)

Usage:
    uv run python examples/so100_teleop/teleop.py

Controls:
    SPACE  — toggle intervention on/off
    Ctrl+C — exit
"""

import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderFollower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

# ── Configure ports ──────────────────────────────────────────────────────────
follower_config = SO101FollowerConfig(
    port="/dev/ttyUSB0",  # adjust to your follower port, e.g. /dev/usb_follower_arm
    id="follower_arm",
    use_degrees=True,
)

leader_config = SOLeaderTeleopConfig(
    port="/dev/ttyUSB1",  # adjust to your leader port, e.g. /dev/usb_leader_arm
    id="leader_arm",
    use_degrees=True,
)

# ── Connect ──────────────────────────────────────────────────────────────────
follower = SO101Follower(follower_config)
leader = SO101LeaderFollower(leader_config)

follower.connect()
leader.connect()

print("\nTeleoperation started.")
print("  Not intervening → leader mirrors follower (torque ON)")
print("  SPACE           → toggle intervention: follower mirrors leader (torque OFF)")
print("  Ctrl+C          → exit\n")

try:
    while True:
        t0 = time.perf_counter()

        # 1. Read current follower joint positions
        follower_obs = follower.get_observation()

        # 2. Send follower positions to leader so it tracks the follower when not intervening
        leader.send_action(follower_obs)

        # 3. Read current leader joint positions
        leader_action = leader.get_action()

        # 4. When intervening, forward leader positions to follower
        if leader.is_intervening:
            follower.send_action(leader_action)
        # else: follower holds its current position (no new command sent)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    follower.disconnect()
    leader.disconnect()
