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
"""Module 1/2/3 unit tests with stubbed VLMs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.annotations.steerable_pipeline.config import (
    Module1Config,
    Module2Config,
    Module3Config,
)
from lerobot.annotations.steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

from ._helpers import make_canned_responder


@dataclass
class _StubFrameProvider:
    """Returns one sentinel object per requested timestamp."""

    sentinel: Any = field(default_factory=lambda: object())
    calls: list[tuple[int, tuple[float, ...]]] = field(default_factory=list)

    def frames_at(self, record, timestamps):
        self.calls.append((record.episode_index, tuple(timestamps)))
        return [self.sentinel] * len(timestamps)


def _spy_responder(captured: list[list[dict[str, Any]]], reply: Any):
    def responder(messages):
        captured.append(list(messages))
        return reply

    return StubVlmClient(responder=responder)


def test_module1_plan_memory_subtask_smoke(fixture_dataset_root: Path, tmp_path: Path) -> None:
    vlm = make_canned_responder(
        {
            "Decompose the demonstration": {
                "subtasks": [
                    {"text": "grasp the handle of the sponge", "start": 0.0, "end": 0.4},
                    {"text": "wipe the counter from left to right", "start": 0.4, "end": 0.8},
                    {"text": "place the sponge into the sink", "start": 0.8, "end": 1.1},
                ]
            },
            "write a concise hierarchical PLAN": {"plan": "1. grasp\n2. wipe\n3. place"},
            "Update the memory": {"memory": "wiped the counter once"},
        },
    )
    module = PlanSubtasksMemoryModule(vlm=vlm, config=Module1Config())
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("module_1")

    styles = {r["style"] for r in rows}
    assert {"subtask", "plan", "memory"}.issubset(styles)
    # subtask timestamps must be exact frame timestamps
    frame_set = set(record.frame_timestamps)
    for row in rows:
        assert row["timestamp"] in frame_set
    # exactly one plan row at t0
    plan_rows = [r for r in rows if r["style"] == "plan"]
    assert len(plan_rows) == 1
    assert plan_rows[0]["timestamp"] == record.frame_timestamps[0]


def test_module2_at_t0_emits_speech_only_no_interjection(fixture_dataset_root: Path, tmp_path: Path) -> None:
    vlm = make_canned_responder(
        {"acknowledgement the robot": {"text": "Sure, on it."}},
    )
    module = InterjectionsAndSpeechModule(
        vlm=vlm,
        config=Module2Config(max_interjections_per_episode=0),
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("module_2")
    assert len(rows) == 1
    only = rows[0]
    assert only["role"] == "assistant"
    assert only["style"] is None
    assert only["content"] is None
    assert only["timestamp"] == record.frame_timestamps[0]
    assert only["tool_calls"][0]["function"]["name"] == "say"


def test_module2_mid_episode_emits_paired_interjection_and_speech(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    vlm = make_canned_responder(
        {
            "acknowledgement the robot": {"text": "OK."},
            "ONE realistic interruption": {
                "interjection": "actually skip the dishes",
                "speech": "Skipping the dishes.",
            },
        },
    )
    module = InterjectionsAndSpeechModule(
        vlm=vlm,
        config=Module2Config(max_interjections_per_episode=1, interjection_min_t=0.2),
        seed=7,
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("module_2")

    interjections = [r for r in rows if r["style"] == "interjection"]
    speeches = [r for r in rows if r["style"] is None and r["role"] == "assistant"]
    assert len(interjections) == 1
    assert len(speeches) >= 2  # initial t=0 + one paired with the interjection
    inter_t = interjections[0]["timestamp"]
    assert any(abs(s["timestamp"] - inter_t) < 1e-9 for s in speeches)


def test_module3_vqa_unique_per_frame(single_episode_root: Path, tmp_path: Path) -> None:
    payload = {
        "question": "How many cups?",
        "answer": {"label": "cup", "count": 2, "note": "white & blue"},
    }
    vlm = make_canned_responder({"frame-grounded visual question": payload})
    module = GeneralVqaModule(
        vlm=vlm,
        config=Module3Config(vqa_emission_hz=1.0, K=3),
        seed=1,
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("module_3")
    user_ts = [r["timestamp"] for r in rows if r["role"] == "user" and r["style"] == "vqa"]
    assistant_ts = [r["timestamp"] for r in rows if r["role"] == "assistant" and r["style"] == "vqa"]
    # at most one user (vqa) per frame; same for assistant
    assert len(user_ts) == len(set(user_ts))
    assert len(assistant_ts) == len(set(assistant_ts))
    # every emitted timestamp must be an exact source frame timestamp
    frame_set = set(record.frame_timestamps)
    for ts in user_ts + assistant_ts:
        assert ts in frame_set


def test_module3_attaches_frame_image_block_to_prompt(single_episode_root: Path, tmp_path: Path) -> None:
    """Each VQA prompt must carry a single image block at the emission frame."""
    captured: list[list[dict[str, Any]]] = []
    payload = {
        "question": "How many cups?",
        "answer": {"label": "cup", "count": 1},
    }
    provider = _StubFrameProvider()
    module = GeneralVqaModule(
        vlm=_spy_responder(captured, payload),
        config=Module3Config(vqa_emission_hz=1.0, K=1),
        seed=0,
        frame_provider=provider,
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    assert captured, "no VLM calls made"
    for messages in captured:
        content = messages[0]["content"]
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        assert len(image_blocks) == 1, f"expected 1 image block per VQA prompt, got {content}"
        assert image_blocks[0]["image"] is provider.sentinel
        assert len(text_blocks) == 1
    # provider was called once per emission with the exact emission timestamp
    for ep_idx, ts_tuple in provider.calls:
        assert ep_idx == record.episode_index
        assert len(ts_tuple) == 1
        assert ts_tuple[0] in record.frame_timestamps


def test_module3_assistant_content_is_valid_json(single_episode_root: Path, tmp_path: Path) -> None:
    payload = {
        "question": "Where is the cup?",
        "answer": {"detections": [{"label": "cup", "bbox_format": "xyxy", "bbox": [10, 20, 50, 80]}]},
    }
    vlm = make_canned_responder({"frame-grounded visual question": payload})
    module = GeneralVqaModule(
        vlm=vlm,
        config=Module3Config(vqa_emission_hz=1.0, K=2),
        seed=2,
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("module_3")
    for row in rows:
        if row["role"] == "assistant" and row["style"] == "vqa":
            decoded = json.loads(row["content"])
            assert "detections" in decoded
