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
SO Leader teleoperator extended with HIL-SERL intervention events.

This thin wrapper around :class:`SOLeader` keeps the underlying joint-reading
behaviour intact (so :func:`get_action` returns ``{"<motor>.pos": float}``)
while adding:

* A pynput keyboard listener that toggles intervention with SPACE and emits
  ``success`` / ``rerecord`` / ``fail`` signals via S / R / Q keys, mirroring
  :class:`KeyboardEndEffectorTeleop`.
* A :func:`get_teleop_events` method satisfying the
  :class:`HasTeleopEvents` protocol consumed by ``AddTeleopEventsAsInfoStep``.
* An :func:`action_features` override that announces the 4-D
  ``[delta_x, delta_y, delta_z, gripper]`` space the leader will project into
  via :class:`LeaderArmInterventionStep` -- this is what ends up recorded by
  ``LeRobotDataset`` in HIL-SERL ``record`` mode.

The actual joint-to-EE-delta conversion does **not** happen here; it is
performed by :class:`LeaderArmInterventionStep` in the action processor
pipeline so the leader stays a pure I/O device.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available

from ..utils import TeleopEvents
from .config_so_leader import SOLeaderTeleopConfig
from .so_leader import SOLeader

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
        self._start_keyboard_listener()

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

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # Reuse the SOLeader joint read so we still expose the leader pose.
        return super().get_action()

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

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._listener is not None:
            with contextlib.suppress(Exception):
                self._listener.stop()
            self._listener = None
        super().disconnect()
