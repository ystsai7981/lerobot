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
from dataclasses import dataclass, field
from typing import Any

from ..config import Module2Config
from ..frames import FrameProvider, null_provider, to_image_blocks
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
    frame_provider: FrameProvider = field(default_factory=null_provider)

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
        # Pull Module 1's subtask spans for this episode so the
        # interjection prompt can ground itself in the actual current
        # subtask at each chosen timestamp. Module 1 ran first.
        subtask_spans = self._read_subtask_spans(staging)
        rows.extend(self._mid_episode_interjections(record, subtask_spans))
        staging.write("module_2", rows)

    @staticmethod
    def _read_subtask_spans(staging: EpisodeStaging) -> list[dict[str, Any]]:
        rows = [r for r in staging.read("module_1") if r.get("style") == "subtask"]
        rows.sort(key=lambda r: float(r["timestamp"]))
        spans: list[dict[str, Any]] = []
        last_t: float | None = None
        for r in rows:
            t = float(r["timestamp"])
            if last_t is not None and spans:
                spans[-1]["end"] = t
            spans.append({"text": r.get("content") or "", "start": t, "end": t})
            last_t = t
        return spans

    @staticmethod
    def _subtask_at(spans: Sequence[dict[str, Any]], t: float) -> str | None:
        current: str | None = None
        for span in spans:
            if float(span["start"]) <= t:
                current = span.get("text")
            else:
                break
        return current

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

    def _mid_episode_interjections(
        self,
        record: EpisodeRecord,
        subtask_spans: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.config.max_interjections_per_episode <= 0:
            return []
        # Deterministic per-episode RNG so reruns are stable across SLURM jobs.
        rng = random.Random(f"{self.seed}:{record.episode_index}:interjection")
        candidate_ts = [t for t in record.frame_timestamps if t >= self.config.interjection_min_t]
        if not candidate_ts:
            return []
        # Pick at most ``max_interjections_per_episode`` distinct timestamps.
        # Previously capped at ``len(candidate_ts) // 4`` — that floor was
        # only relevant for very short episodes; for any real ~20-30s
        # episode it had no effect, but it silently set the count to 0 on
        # short fixtures. Just take ``min(max, len)`` directly.
        n = min(self.config.max_interjections_per_episode, len(candidate_ts))
        if n <= 0:
            return []
        chosen = sorted(rng.sample(candidate_ts, n))

        out: list[dict[str, Any]] = []
        for t in chosen:
            t_snap = _snap_to_frame(t, record.frame_timestamps)
            window_ts = self._window_timestamps(t_snap, record.frame_timestamps)
            current_subtask = (
                self._subtask_at(subtask_spans, t_snap) or record.episode_task
            )
            prompt = load_prompt("module_2_interjection").format(
                episode_task=record.episode_task,
                current_subtask=current_subtask,
                timestamp=t_snap,
                window_seconds=self.config.interjection_window_seconds,
            )
            images = self.frame_provider.frames_at(record, window_ts)
            content = [*to_image_blocks(images), {"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]
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

    def _window_timestamps(
        self, t_anchor: float, frame_timestamps: Sequence[float]
    ) -> list[float]:
        """Return a small set of frame timestamps spanning the lead-up to ``t``.

        The VLM receives roughly ``num_frames`` frames over the
        ``window_seconds`` immediately before ``t_anchor``, snapped to
        actual source frame timestamps. This gives the interjection
        prompt enough temporal context to read what's visibly happening
        instead of looking at one frozen frame.
        """
        if not frame_timestamps:
            return [t_anchor]
        n = max(1, int(self.config.interjection_window_frames))
        if n == 1:
            return [t_anchor]
        window = float(self.config.interjection_window_seconds)
        step = window / max(1, n - 1)
        targets = [t_anchor - step * (n - 1 - i) for i in range(n)]
        snapped: list[float] = []
        seen: set[float] = set()
        for tgt in targets:
            t = _snap_to_frame(max(0.0, tgt), frame_timestamps)
            if t not in seen:
                seen.add(t)
                snapped.append(t)
        return snapped or [t_anchor]
