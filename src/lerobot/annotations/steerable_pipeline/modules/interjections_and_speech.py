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
"""Module 2: interjections + paired speech (EVENT styles + speech atoms).

Two sub-passes:

1. At ``t=0``, emit ONLY a speech tool-call atom (acknowledgement of the
   canonical task). No interjection row — the canonical task is already the
   user utterance from ``meta/tasks.parquet``.

2. For mid-episode interruptions, emit a co-timestamped pair:
       {role:user, style:interjection, content:<text>}
       speech atom (role:assistant, style:None, tool_calls=[say(...)])
   Both rows go in ``language_events`` at the same timestamp.

Module 1's :meth:`run_plan_updates` reuses Module 2's interjection
timestamps to refresh the ``plan`` row at the same instant.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Module2Config
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient
from ..writer import speech_atom


def _snap_to_frame(t: float, frame_timestamps: Sequence[float]) -> float:
    if not frame_timestamps:
        return float(t)
    return float(min(frame_timestamps, key=lambda f: abs(f - t)))


@dataclass
class InterjectionsAndSpeechModule:
    """Generate task-start speech and mid-episode interjection/speech pairs."""

    vlm: VlmClient
    config: Module2Config
    seed: int = 1729

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        if record.frame_timestamps:
            t0 = float(record.frame_timestamps[0])
            initial = self._initial_speech(record)
            if initial:
                rows.append(speech_atom(t0, initial))
        rows.extend(self._mid_episode_interjections(record))
        staging.write("module_2", rows)

    def _initial_speech(self, record: EpisodeRecord) -> str | None:
        prompt = load_prompt("module_2_initial_speech").format(
            episode_task=record.episode_task,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            text = result["text"].strip()
            if text:
                return text
        return None

    def _mid_episode_interjections(self, record: EpisodeRecord) -> list[dict[str, Any]]:
        if self.config.max_interjections_per_episode <= 0:
            return []
        # Deterministic per-episode RNG so reruns are stable across SLURM jobs.
        rng = random.Random(f"{self.seed}:{record.episode_index}:interjection")
        candidate_ts = [t for t in record.frame_timestamps if t >= self.config.interjection_min_t]
        if not candidate_ts:
            return []
        n = min(self.config.max_interjections_per_episode, len(candidate_ts) // 4)
        if n <= 0:
            return []
        chosen = sorted(rng.sample(candidate_ts, n))
        out: list[dict[str, Any]] = []
        for t in chosen:
            t_snap = _snap_to_frame(t, record.frame_timestamps)
            current_subtask = record.episode_task
            prompt = load_prompt("module_2_interjection").format(
                episode_task=record.episode_task,
                current_subtask=current_subtask,
                timestamp=t_snap,
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            result = self.vlm.generate_json([messages])[0]
            if not isinstance(result, dict):
                continue
            interjection_text = result.get("interjection")
            speech_text = result.get("speech")
            if not isinstance(interjection_text, str) or not interjection_text.strip():
                continue
            if not isinstance(speech_text, str) or not speech_text.strip():
                continue
            out.append(
                {
                    "role": "user",
                    "content": interjection_text.strip(),
                    "style": "interjection",
                    "timestamp": t_snap,
                    "tool_calls": None,
                }
            )
            out.append(speech_atom(t_snap, speech_text.strip()))
        return out
