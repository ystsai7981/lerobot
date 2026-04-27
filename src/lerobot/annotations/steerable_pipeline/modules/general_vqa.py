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
"""Module 3: general VQA at a timed cadence.

Anchors ``K`` (question, answer) pairs to ``K`` consecutive frames per
emission so each frame gets at most one ``(vqa, user)`` and one
``(vqa, assistant)`` pair — keeps the resolver contract scalar.

Question types covered (per the plan's Module 3 table): bbox, keypoint,
count, attribute, spatial. The assistant's ``content`` is a JSON string
whose schema depends on the question type. Malformed JSON triggers one
retry inside :meth:`VlmClient.generate_json`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import Module3Config
from ..frames import FrameProvider, null_provider, to_image_blocks
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord
from ..staging import EpisodeStaging
from ..validator import classify_vqa_answer
from ..vlm_client import VlmClient


def _emission_anchor_indices(frame_timestamps: Sequence[float], hz: float, k: int) -> list[int]:
    """Return the relative frame indices to anchor VQA emissions to.

    For each emission tick (every ``1/hz`` seconds), we anchor ``k``
    consecutive frames starting at the tick. Ticks fall on the nearest
    available source frame timestamp.
    """
    if hz <= 0 or k <= 0 or not frame_timestamps:
        return []
    t0 = frame_timestamps[0]
    t_last = frame_timestamps[-1]
    period = 1.0 / hz
    indices: list[int] = []
    t = t0
    while t <= t_last + 1e-9:
        # find the index of the nearest frame to t
        nearest_i = min(range(len(frame_timestamps)), key=lambda i: abs(frame_timestamps[i] - t))
        for offset in range(k):
            j = nearest_i + offset
            if j >= len(frame_timestamps):
                break
            if not indices or indices[-1] != j:
                indices.append(j)
        t += period
    # dedupe while preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        deduped.append(i)
    return deduped


@dataclass
class GeneralVqaModule:
    """Emit grounded VQA pairs at a timed cadence."""

    vlm: VlmClient
    config: Module3Config
    seed: int = 1729
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        if not record.frame_timestamps:
            staging.write("module_3", [])
            return
        rng = random.Random(f"{self.seed}:{record.episode_index}:vqa")
        anchor_idx = _emission_anchor_indices(
            record.frame_timestamps, self.config.vqa_emission_hz, self.config.K
        )
        rows: list[dict[str, Any]] = []
        for idx in anchor_idx:
            ts = float(record.frame_timestamps[idx])
            qtype = rng.choice(self.config.question_types)
            qa = self._generate_one(record, qtype, ts)
            if qa is None:
                continue
            question, answer = qa
            rows.append(
                {
                    "role": "user",
                    "content": question,
                    "style": "vqa",
                    "timestamp": ts,
                    "tool_calls": None,
                }
            )
            rows.append(
                {
                    "role": "assistant",
                    "content": json.dumps(answer, sort_keys=True),
                    "style": "vqa",
                    "timestamp": ts,
                    "tool_calls": None,
                }
            )
        staging.write("module_3", rows)

    def _generate_one(
        self, record: EpisodeRecord, question_type: str, frame_timestamp: float
    ) -> tuple[str, dict[str, Any]] | None:
        prompt = load_prompt("module_3_vqa").format(
            episode_task=record.episode_task,
            question_type=question_type,
        )
        images = self.frame_provider.frames_at(record, [frame_timestamp])
        content = [*to_image_blocks(images), {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        result = self.vlm.generate_json([messages])[0]
        if not isinstance(result, dict):
            return None
        question = result.get("question")
        answer = result.get("answer")
        if not isinstance(question, str) or not question.strip():
            return None
        if not isinstance(answer, dict):
            return None
        # The validator will enforce shape; here we just sanity-check that the
        # answer matches *some* known shape so we can drop garbage early.
        if classify_vqa_answer(answer) is None:
            return None
        return question.strip(), answer
