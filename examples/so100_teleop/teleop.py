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
  - Default (not intervening): follower holds its current position.
    The leader arm has torque ENABLED and mirrors the follower so there is no
    large position jump when intervention starts.
  - Intervention (SPACE pressed): leader torque DISABLED, human moves the leader
    freely, and the follower mirrors the leader joint-by-joint.

Usage:
    uv run python examples/so100_teleop/teleop.py

Controls:
    SPACE  — toggle intervention on/off
    Ctrl+C — exit
"""

import logging
import os
import sys
import time
from threading import Event, Thread

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── pynput keyboard listener ─────────────────────────────────────────────────
PYNPUT_AVAILABLE = True
try:
    if "DISPLAY" not in os.environ and "linux" in sys.platform:
        raise ImportError("No DISPLAY set, pynput skipped.")
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False

# ── Configure ports ──────────────────────────────────────────────────────────
FOLLOWER_PORT = "/dev/ttyUSB0"  # ← change to your follower port
LEADER_PORT = "/dev/ttyUSB1"  # ← change to your leader port
FPS = 30


def hold_position(robot) -> dict:
    """Read current joint positions and write them back as the goal.

    This prevents the motors from snapping to a stale Goal_Position register
    value (which can happen when torque is re-enabled after calibration).
    Returns the current position dict for reuse.
    """
    current = robot.bus.sync_read("Present_Position")
    robot.bus.sync_write("Goal_Position", current)
    return {f"{motor}.pos": val for motor, val in current.items()}


# ── Connect ───────────────────────────────────────────────────────────────────
follower_config = SO101FollowerConfig(
    port=FOLLOWER_PORT,
    id="follower_arm",
    use_degrees=True,
)
leader_config = SOLeaderTeleopConfig(
    port=LEADER_PORT,
    id="leader_arm",
    use_degrees=True,
)

follower = SO101Follower(follower_config)
leader = SO101Leader(leader_config)

follower.connect()
leader.connect()

# ── CRITICAL: hold both arms at their current position before doing anything ─
# configure() enables follower torque, and the Goal_Position register may contain
# a stale value from a previous session. Writing current→goal prevents sudden motion.
follower_current = hold_position(follower)
leader_current = hold_position(leader)  # leader torque is still off here, but sets the register

# ── Intervention state + keyboard listener ───────────────────────────────────
is_intervening = False
stop_event = Event()


def _start_keyboard_listener():
    if not PYNPUT_AVAILABLE:
        logger.warning("pynput not available — spacebar toggle disabled.")
        return None

    def on_press(key):
        global is_intervening
        if key == pynput_keyboard.Key.space:
            is_intervening = not is_intervening
            state = "INTERVENTION  (leader → follower)" if is_intervening else "IDLE  (follower holds)"
            print(f"\n[SPACE] {state}\n")

    def listen():
        with pynput_keyboard.Listener(on_press=on_press) as listener:
            while not stop_event.is_set():
                time.sleep(0.05)
            listener.stop()

    t = Thread(target=listen, daemon=True)
    t.start()
    return t


kbd_thread = _start_keyboard_listener()

# Enable leader torque AFTER writing its goal to current position, so it holds in place.
leader.bus.sync_write("Torque_Enable", 1)
leader_torque_on = True

print("\nTeleoperation ready.")
print("  SPACE  → toggle intervention (leader controls follower)")
print("  Ctrl+C → exit\n")

try:
    while True:
        t0 = time.perf_counter()

        if is_intervening:
            # ── Intervention: leader torque OFF, follower mirrors leader ──────
            if leader_torque_on:
                leader.bus.sync_write("Torque_Enable", 0)
                leader_torque_on = False

            leader_action = leader.get_action()  # reads present leader joints
            follower.send_action(leader_action)  # follower tracks leader

        else:
            # ── Idle: leader torque ON, leader mirrors follower, follower holds
            if not leader_torque_on:
                # Before re-enabling torque, set the leader's goal to its current
                # position so it doesn't snap to the follower position suddenly.
                hold_position(leader)
                leader.bus.sync_write("Torque_Enable", 1)
                leader_torque_on = True

            follower_obs = follower.get_observation()
            # Command leader to match follower (so next intervention has no jump)
            goal_pos = {motor: follower_obs[f"{motor}.pos"] for motor in leader.bus.motors}
            leader.bus.sync_write("Goal_Position", goal_pos)
            # Follower holds — no send_action call

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    stop_event.set()
    leader.bus.sync_write("Torque_Enable", 0)
    follower.disconnect()
    leader.disconnect()
