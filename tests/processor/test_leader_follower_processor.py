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
"""Tests for LeaderArmInterventionStep (placo-free, FK is mocked)."""

from typing import Any

import numpy as np
import pytest

from lerobot.processor.converters import create_transition
from lerobot.processor.leader_follower_processor import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    GRIPPER_STAY,
    LeaderArmInterventionStep,
)
from lerobot.types import TransitionKey

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
STEP_SIZES = {"x": 0.025, "y": 0.025, "z": 0.025}


class _FakeKinematics:
    """Minimal stand-in for `RobotKinematics.forward_kinematics`.

    Maps a joint vector deterministically to a 4x4 transform whose translation
    is `(j[0] * 0.001, j[1] * 0.001, j[2] * 0.001)`. This lets the test drive
    arbitrary EE positions by choosing leader / follower joint values without
    depending on placo / a URDF.
    """

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        t = np.eye(4, dtype=float)
        t[:3, 3] = np.asarray(joints[:3], dtype=float) * 1e-3
        return t


def _joint_dict(values: list[float]) -> dict[str, float]:
    return {f"{name}.pos": v for name, v in zip(MOTOR_NAMES, values, strict=False)}


def _make_step(use_gripper: bool = True) -> LeaderArmInterventionStep:
    return LeaderArmInterventionStep(
        kinematics=_FakeKinematics(),  # type: ignore[arg-type]
        motor_names=MOTOR_NAMES,
        end_effector_step_sizes=STEP_SIZES,
        use_gripper=use_gripper,
    )


def _build_transition(
    leader_joints: dict[str, float] | None,
    follower_joints: dict[str, float] | None,
    extra_complementary: dict[str, Any] | None = None,
) -> Any:
    complementary: dict[str, Any] = dict(extra_complementary or {})
    if leader_joints is not None:
        complementary["teleop_action"] = leader_joints
    if follower_joints is not None:
        complementary["raw_joint_positions"] = follower_joints
    return create_transition(complementary_data=complementary)


def test_replaces_teleop_action_with_normalised_ee_delta():
    leader = _joint_dict([25.0, 0.0, 0.0, 0.0, 0.0])
    leader["gripper.pos"] = 80.0
    follower = _joint_dict([0.0, 0.0, 0.0, 0.0, 0.0])
    follower["gripper.pos"] = 30.0

    transition = _build_transition(leader, follower)
    step = _make_step()
    out = step(transition)

    teleop_action = out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
    assert set(teleop_action) == {"delta_x", "delta_y", "delta_z", "gripper"}
    # joint 0 differs by +25 -> 0.025 m -> normalised by 0.025 step -> 1.0
    assert teleop_action["delta_x"] == pytest.approx(1.0)
    assert teleop_action["delta_y"] == pytest.approx(0.0)
    assert teleop_action["delta_z"] == pytest.approx(0.0)
    # leader gripper 80 >= open threshold 60 -> open command
    assert teleop_action["gripper"] == GRIPPER_OPEN


def test_clips_delta_to_unit_box():
    leader = _joint_dict([1000.0, -1000.0, 1000.0, 0.0, 0.0])
    follower = _joint_dict([0.0, 0.0, 0.0, 0.0, 0.0])
    transition = _build_transition(leader, follower)

    out = _make_step(use_gripper=False)(transition)

    teleop_action = out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
    assert "gripper" not in teleop_action
    assert teleop_action["delta_x"] == pytest.approx(1.0)
    assert teleop_action["delta_y"] == pytest.approx(-1.0)
    assert teleop_action["delta_z"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("leader_gripper", "expected"),
    [
        (10.0, GRIPPER_CLOSE),
        (45.0, GRIPPER_STAY),
        (90.0, GRIPPER_OPEN),
    ],
)
def test_gripper_quantisation(leader_gripper: float, expected: float):
    leader = _joint_dict([0.0, 0.0, 0.0, 0.0, 0.0])
    leader["gripper.pos"] = leader_gripper
    follower = _joint_dict([0.0, 0.0, 0.0, 0.0, 0.0])
    follower["gripper.pos"] = 50.0

    out = _make_step(use_gripper=True)(_build_transition(leader, follower))
    teleop_action = out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
    assert teleop_action["gripper"] == expected


def test_zero_action_when_follower_joints_missing():
    leader = _joint_dict([10.0, 10.0, 10.0, 0.0, 0.0])
    leader["gripper.pos"] = 50.0
    transition = _build_transition(leader, follower_joints=None)

    out = _make_step()(transition)

    teleop_action = out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
    assert teleop_action == {
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_z": 0.0,
        "gripper": GRIPPER_STAY,
    }


def test_passthrough_when_teleop_action_missing():
    transition = _build_transition(leader_joints=None, follower_joints=None)
    out = _make_step()(transition)
    assert "teleop_action" not in out[TransitionKey.COMPLEMENTARY_DATA]


def test_passthrough_when_teleop_action_is_already_delta_dict():
    """Idempotent on dicts that don't look like raw joint reads."""
    delta = {"delta_x": 0.5, "delta_y": 0.0, "delta_z": -0.3, "gripper": GRIPPER_OPEN}
    follower = _joint_dict([0.0, 0.0, 0.0, 0.0, 0.0])
    transition = _build_transition(delta, follower)
    out = _make_step()(transition)
    assert out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == delta


def test_reads_follower_from_observation_when_complementary_missing():
    leader = _joint_dict([20.0, 0.0, 0.0, 0.0, 0.0])
    leader["gripper.pos"] = 50.0
    follower = _joint_dict([10.0, 0.0, 0.0, 0.0, 0.0])

    transition = create_transition(
        observation=follower,
        complementary_data={"teleop_action": leader},
    )
    out = _make_step()(transition)

    teleop_action = out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
    # delta = (20 - 10) * 1e-3 = 0.01, normalised by 0.025 -> 0.4
    assert teleop_action["delta_x"] == pytest.approx(0.4)
