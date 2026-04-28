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
"""Keyframe extraction for the annotation pipeline.

Modules attach decoded camera frames to their VLM prompts so the model can
ground subtask decomposition, interjection scenarios, and VQA in actual
visual content. The pipeline shares one provider across modules and one
episode at a time, with a small per-episode cache so multiple modules
querying the same timestamp pay decode cost once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .reader import EpisodeRecord


class FrameProvider(Protocol):
    """Decodes camera frames at episode-relative timestamps."""

    def frames_at(self, record: EpisodeRecord, timestamps: list[float]) -> list[Any]:
        """Return one PIL.Image per timestamp; empty list if no camera available."""

    def video_for_episode(self, record: EpisodeRecord, max_frames: int) -> list[Any]:
        """Return up to ``max_frames`` PIL images covering the whole episode.

        Sampling is uniform across the episode duration. The returned list is
        intended to be passed as one ``{"type":"video", "video":<list>}``
        block to a Qwen-VL-compatible model that pools temporally itself.
        Empty list if no camera available.
        """


@dataclass
class _NullProvider:
    """No-op provider used when the dataset has no video keys or in tests."""

    def frames_at(self, record: EpisodeRecord, timestamps: list[float]) -> list[Any]:
        return []

    def video_for_episode(self, record: EpisodeRecord, max_frames: int) -> list[Any]:
        return []


def null_provider() -> FrameProvider:
    return _NullProvider()


@dataclass
class VideoFrameProvider:
    """Decodes frames from the dataset's first ``observation.images.*`` stream.

    The first camera key is used unconditionally — Module 1/2/3 prompts care
    about *what is happening*, not which camera angle the model sees, so a
    single canonical viewpoint is enough. Override ``camera_key`` if you
    want a specific stream.

    Caches up to ``cache_size`` decoded frames per process to keep
    co-timestamped Module 2 + Module 1 plan-update calls cheap.
    """

    root: Path
    camera_key: str | None = None
    tolerance_s: float = 1e-2
    cache_size: int = 256
    _meta: Any = field(default=None, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415

        self._meta = LeRobotDatasetMetadata(repo_id="local", root=self.root)
        if self.camera_key is None:
            keys = self._meta.video_keys
            self.camera_key = keys[0] if keys else None

    def frames_at(self, record: EpisodeRecord, timestamps: list[float]) -> list[Any]:
        if not timestamps or self.camera_key is None:
            return []

        out: list[Any] = []
        misses: list[float] = []
        miss_indices: list[int] = []
        for i, ts in enumerate(timestamps):
            key = (record.episode_index, round(float(ts), 6))
            cached = self._cache.get(key)
            if cached is not None:
                out.append(cached)
            else:
                out.append(None)
                misses.append(float(ts))
                miss_indices.append(i)

        if misses:
            decoded = self._decode(record.episode_index, misses)
            # decoder may return fewer frames than requested when some
            # timestamps fall outside the video; pair what we have and
            # leave the rest as None to be filtered below.
            for i, img in zip(miss_indices, decoded):
                out[i] = img
                key = (record.episode_index, round(float(timestamps[i]), 6))
                if len(self._cache) >= self.cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = img
        # filter out any None left over from decode failures
        return [img for img in out if img is not None]

    def _decode(self, episode_index: int, timestamps: list[float]) -> list[Any]:
        import os as _os  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        from lerobot.datasets.video_utils import decode_video_frames  # noqa: PLC0415

        ep = self._meta.episodes[episode_index]
        from_timestamp = ep[f"videos/{self.camera_key}/from_timestamp"]
        shifted = [from_timestamp + ts for ts in timestamps]
        video_path = self.root / self._meta.get_video_file_path(episode_index, self.camera_key)
        # ``torchcodec`` import currently bad-allocs on cu128/torch-2.8 in
        # some environments; default to ``pyav`` (always available via
        # the ``av`` package) and let users override with
        # LEROBOT_VIDEO_BACKEND=torchcodec when their stack supports it.
        backend = _os.environ.get("LEROBOT_VIDEO_BACKEND", "pyav")
        try:
            frames = decode_video_frames(
                video_path,
                shifted,
                self.tolerance_s,
                backend=backend,
                return_uint8=True,
            )
        except Exception:
            return []
        # frames: [N, C, H, W] uint8, RGB
        out: list[Any] = []
        arr = frames.cpu().numpy() if hasattr(frames, "cpu") else frames
        for i in range(arr.shape[0]):
            chw = arr[i]
            hwc = chw.transpose(1, 2, 0)
            out.append(Image.fromarray(hwc, mode="RGB"))
        return out

    def video_for_episode(self, record: EpisodeRecord, max_frames: int) -> list[Any]:
        """Return up to ``max_frames`` images uniformly sampled across the episode.

        The whole episode duration is covered; the model picks subtask
        boundaries from the temporal pooling it does internally.
        """
        if max_frames <= 0 or self.camera_key is None or not record.frame_timestamps:
            return []
        n_frames = min(max_frames, len(record.frame_timestamps))
        if n_frames == len(record.frame_timestamps):
            timestamps = list(record.frame_timestamps)
        else:
            t0 = record.frame_timestamps[0]
            t_last = record.frame_timestamps[-1]
            if t_last <= t0:
                timestamps = [float(t0)] * n_frames
            else:
                step = (t_last - t0) / (n_frames - 1) if n_frames > 1 else 0.0
                timestamps = [float(t0 + i * step) for i in range(n_frames)]
        return self.frames_at(record, timestamps)


def make_frame_provider(root: Path, camera_key: str | None = None) -> FrameProvider:
    """Build a :class:`VideoFrameProvider` if videos are present, else null."""
    try:
        provider = VideoFrameProvider(root=root, camera_key=camera_key)
    except Exception:
        return null_provider()
    if provider.camera_key is None:
        return null_provider()
    return provider


def to_image_blocks(images: list[Any]) -> list[dict[str, Any]]:
    """Convert PIL images to Qwen-VL-compatible content blocks."""
    return [{"type": "image", "image": img} for img in images]


def to_video_block(images: list[Any]) -> list[dict[str, Any]]:
    """Wrap a list of PIL images as one Qwen-VL video block.

    Returns ``[]`` when the list is empty, so the caller can splat the result
    into a content array without a separate emptiness check.
    """
    if not images:
        return []
    return [{"type": "video", "video": list(images)}]
