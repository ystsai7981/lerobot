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
"""Module 1: subtask decomposition + plan + memory (PERSISTENT styles)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pathlib import Path

from ..config import Module1Config
from ..frames import (
    FrameProvider,
    VideoFrameProvider,
    episode_clip_path,
    null_provider,
    to_video_block,
    to_video_url_block,
)
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient


def _snap_to_frame(t: float, frame_timestamps: Sequence[float]) -> float:
    """Snap an arbitrary float to the nearest exact source frame timestamp."""
    if not frame_timestamps:
        return float(t)
    nearest = min(frame_timestamps, key=lambda f: abs(f - t))
    return float(nearest)


@dataclass
class PlanSubtasksMemoryModule:
    """Generate subtask spans, plan, and memory rows.

    All output is persistent (lives in ``language_persistent``):

    - ``subtask`` rows: one per span, stamped at the span's *start* timestamp
      (snapped to an exact frame).
    - ``plan`` rows: emitted at ``t=0``; refreshed at every interjection
      timestamp via :meth:`run_plan_updates` (called by the executor after
      Module 2 completes).
    - ``memory`` rows: emitted at each subtask boundary (= subtask start
      timestamp from the second subtask onward).
    """

    vlm: VlmClient
    config: Module1Config
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        subtask_spans = self._generate_subtasks(record)
        # subtask rows
        for span in subtask_spans:
            rows.append(
                {
                    "role": "assistant",
                    "content": span["text"],
                    "style": "subtask",
                    "timestamp": _snap_to_frame(span["start"], record.frame_timestamps),
                    "tool_calls": None,
                }
            )
        # plan row at t=0
        plan_text = self._generate_plan(record, subtask_spans)
        if plan_text is not None:
            t0 = record.frame_timestamps[0] if record.frame_timestamps else 0.0
            rows.append(
                {
                    "role": "assistant",
                    "content": plan_text,
                    "style": "plan",
                    "timestamp": float(t0),
                    "tool_calls": None,
                }
            )
        # memory rows at every subtask boundary except the very first start
        prior_memory = ""
        for i, span in enumerate(subtask_spans[1:], start=1):
            completed = subtask_spans[i - 1]["text"]
            remaining = [s["text"] for s in subtask_spans[i:]]
            mem_text = self._generate_memory(record, prior_memory, completed, remaining)
            if mem_text:
                ts = _snap_to_frame(span["start"], record.frame_timestamps)
                rows.append(
                    {
                        "role": "assistant",
                        "content": mem_text,
                        "style": "memory",
                        "timestamp": ts,
                        "tool_calls": None,
                    }
                )
                prior_memory = mem_text
        staging.write("module_1", rows)

    def run_plan_updates(
        self,
        record: EpisodeRecord,
        staging: EpisodeStaging,
        interjection_times: Sequence[float],
        interjection_texts: Sequence[str] | None = None,
    ) -> None:
        """Append additional ``plan`` rows at every interjection timestamp.

        Plans refresh ONLY on user interjections — subtask generation
        runs ~1 Hz at inference, but plan re-emission is event-driven.
        Now also forwards the interjection's own text into the prompt so
        the refreshed plan can actually reflect the user's correction
        (the previous version told the model "an interjection happened"
        without telling it what the user said).
        """
        existing = staging.read("module_1")
        spans = self._reconstruct_subtasks_from_rows(existing)
        already_planned: set[float] = {
            float(r["timestamp"]) for r in existing if r.get("style") == "plan"
        }
        new_rows = list(existing)

        texts: list[str | None] = (
            [None] * len(interjection_times)
            if interjection_texts is None
            else [str(t) if t else None for t in interjection_texts]
        )
        for raw_t, inter_text in zip(interjection_times, texts):
            t = _snap_to_frame(raw_t, record.frame_timestamps)
            if t in already_planned:
                continue
            already_planned.add(t)
            plan_text = self._generate_plan(
                record, spans, refresh_t=t, interjection=inter_text
            )
            if plan_text is not None:
                new_rows.append(
                    {
                        "role": "assistant",
                        "content": plan_text,
                        "style": "plan",
                        "timestamp": t,
                        "tool_calls": None,
                    }
                )
        staging.write("module_1", new_rows)

    @staticmethod
    def _reconstruct_subtasks_from_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        last_t: float | None = None
        for row in sorted(
            (r for r in rows if r.get("style") == "subtask"),
            key=lambda r: float(r["timestamp"]),
        ):
            t = float(row["timestamp"])
            if last_t is not None:
                out[-1]["end"] = t
            out.append({"text": row.get("content") or "", "start": t, "end": t})
            last_t = t
        return out

    def _generate_subtasks(self, record: EpisodeRecord) -> list[dict[str, Any]]:
        if record.row_count == 0 or not record.frame_timestamps:
            return []
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        prompt = load_prompt("module_1_subtasks").format(
            episode_task=record.episode_task,
            min_subtask_seconds=self.config.min_subtask_seconds,
            max_steps=self.config.plan_max_steps,
            episode_duration=f"{episode_duration:.3f}",
        )
        if self.config.use_video_url and isinstance(self.frame_provider, VideoFrameProvider):
            cache_dir = Path(self.frame_provider.root) / ".annotate_staging" / ".video_clips"
            clip = episode_clip_path(record, self.frame_provider, cache_dir)
            video_block = (
                to_video_url_block(f"file://{clip}", fps=self.config.use_video_url_fps)
                if clip is not None
                else []
            )
        else:
            target_count = max(
                1,
                int(round(episode_duration * self.config.frames_per_second)),
            )
            target_count = min(target_count, self.config.max_video_frames)
            video_frames = self.frame_provider.video_for_episode(record, target_count)
            video_block = to_video_block(video_frames)
        content = [*video_block, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        result = self.vlm.generate_json([messages])[0]
        spans = result.get("subtasks") if isinstance(result, dict) else None
        if not spans:
            return []
        # clamp to [t0, t_last] and sort
        t0 = record.frame_timestamps[0]
        t_last = record.frame_timestamps[-1]
        cleaned: list[dict[str, Any]] = []
        for span in spans:
            try:
                start = float(span["start"])
                end = float(span["end"])
                text = str(span["text"]).strip()
            except (KeyError, ValueError, TypeError):
                continue
            start = max(t0, min(start, t_last))
            end = max(t0, min(end, t_last))
            if end < start:
                start, end = end, start
            if not text:
                continue
            cleaned.append({"text": text, "start": start, "end": end})
        cleaned.sort(key=lambda s: s["start"])
        return cleaned

    def _generate_plan(
        self,
        record: EpisodeRecord,
        subtask_spans: Sequence[dict[str, Any]],
        *,
        refresh_t: float | None = None,
        interjection: str | None = None,
    ) -> str | None:
        if not subtask_spans:
            return None
        subtasks_text = "\n".join(f"- {s['text']}" for s in subtask_spans)
        prompt = load_prompt("module_1_plan").format(
            episode_task=record.episode_task,
            subtasks_text=subtasks_text,
            plan_max_steps=self.config.plan_max_steps,
        )
        if refresh_t is not None:
            # ``current_subtask`` is the span the refresh time falls into,
            # so the model knows where in the demonstration the planner is
            # standing when it re-emits.
            current_subtask = ""
            for span in subtask_spans:
                if float(span["start"]) <= refresh_t and (
                    "end" not in span or float(span["end"]) > refresh_t
                ):
                    current_subtask = span.get("text", "")
                    break
            if interjection:
                prompt += (
                    f"\n\n(Plan refresh at t={refresh_t:.2f}s after a user "
                    f"interjection: {interjection!r}. Current subtask just "
                    f"before the interjection: {current_subtask!r}. Update "
                    f"the plan so it reflects the interjection — drop or "
                    f"reorder steps as needed; do not just restate.)\n"
                )
            else:
                # Refresh without an interjection text: still tell the model
                # where in the episode the plan stands so the re-emission
                # is grounded. Should be rare — plan refreshes are
                # interjection-driven by design.
                prompt += (
                    f"\n\n(Plan refresh at t={refresh_t:.2f}s. Current "
                    f"subtask: {current_subtask!r}.)\n"
                )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict) and isinstance(result.get("plan"), str):
            return result["plan"].strip()
        return None

    def _generate_memory(
        self,
        record: EpisodeRecord,
        prior_memory: str,
        completed: str,
        remaining: Sequence[str],
    ) -> str:
        prompt = load_prompt("module_1_memory").format(
            episode_task=record.episode_task,
            prior_memory=prior_memory or "(none)",
            completed_subtask=completed,
            remaining_subtasks=", ".join(remaining) if remaining else "(none)",
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict) and isinstance(result.get("memory"), str):
            return result["memory"].strip()
        return ""
