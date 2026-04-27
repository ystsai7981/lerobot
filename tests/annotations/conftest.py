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
"""Shared fixtures for annotation-pipeline tests.

Builds a minimal LeRobot-shaped dataset on disk so writer/validator tests
can exercise real parquet reads and writes without needing a checked-in
LFS dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _make_episode_table(
    episode_index: int,
    num_frames: int,
    *,
    fps: int = 10,
    task_index: int = 0,
) -> pa.Table:
    timestamps = [round(i / fps, 6) for i in range(num_frames)]
    frame_indices = list(range(num_frames))
    return pa.Table.from_pydict(
        {
            "episode_index": [episode_index] * num_frames,
            "frame_index": frame_indices,
            "timestamp": timestamps,
            "task_index": [task_index] * num_frames,
            "subtask_index": [0] * num_frames,  # legacy column the writer must drop
        }
    )


def _build_dataset(root: Path, episode_specs: list[tuple[int, int, str]], *, fps: int = 10) -> Path:
    """Create a fixture dataset under ``root``.

    ``episode_specs`` is a list of ``(episode_index, num_frames, task_text)``.
    Each episode goes into its own ``data/chunk-000/file-{ep:03d}.parquet``
    so the writer's per-shard rewrite path is exercised.
    """
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for episode_index, num_frames, task_text in episode_specs:
        task_index = len(tasks)
        if task_text not in tasks.values():
            tasks[task_index] = task_text
        else:
            task_index = next(k for k, v in tasks.items() if v == task_text)
        table = _make_episode_table(episode_index, num_frames, fps=fps, task_index=task_index)
        path = data_dir / f"file-{episode_index:03d}.parquet"
        pq.write_table(table, path)

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    tasks_table = pa.Table.from_pydict(
        {
            "task_index": list(tasks.keys()),
            "task": list(tasks.values()),
        }
    )
    pq.write_table(tasks_table, meta_dir / "tasks.parquet")

    info = {
        "codebase_version": "v3.1",
        "fps": fps,
        "total_episodes": len(episode_specs),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    return root


@pytest.fixture
def fixture_dataset_root(tmp_path: Path) -> Path:
    """A tiny dataset with two episodes, 12 frames each at 10 fps."""
    return _build_dataset(
        tmp_path / "ds",
        episode_specs=[
            (0, 12, "Could you tidy the kitchen please?"),
            (1, 12, "Please clean up the kitchen"),
        ],
        fps=10,
    )


@pytest.fixture
def single_episode_root(tmp_path: Path) -> Path:
    return _build_dataset(
        tmp_path / "ds_one",
        episode_specs=[(0, 30, "Pour water from the bottle into the cup.")],
        fps=10,
    )
