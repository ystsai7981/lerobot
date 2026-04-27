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
"""Opt-in E2E smoke run for ``make annotation-e2e``.

Builds the same fixture used by the pytest suite, runs the full
annotation pipeline against it with a stub VLM, and prints a short report.
This is intentionally not a pytest test — it exercises the CLI plumbing
without depending on conftest.py fixtures.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
from lerobot.annotations.steerable_pipeline.executor import Executor
from lerobot.annotations.steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.validator import StagingValidator
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter


def _build_dataset(root: Path) -> Path:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    n = 30
    timestamps = [round(i / 10, 6) for i in range(n)]
    table = pa.Table.from_pydict(
        {
            "episode_index": [0] * n,
            "frame_index": list(range(n)),
            "timestamp": timestamps,
            "task_index": [0] * n,
            "subtask_index": [0] * n,
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict({"task_index": [0], "task": ["Pour water into the cup."]}),
        meta / "tasks.parquet",
    )
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.1", "fps": 10}))
    return root


def _stub_responder(messages):
    text = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
            elif isinstance(content, str):
                text = content
    if "Decompose the demonstration" in text:
        return {
            "subtasks": [
                {"text": "grasp the bottle", "start": 0.0, "end": 1.0},
                {"text": "pour into the cup", "start": 1.0, "end": 2.0},
                {"text": "place the bottle down", "start": 2.0, "end": 3.0},
            ]
        }
    if "concise hierarchical PLAN" in text:
        return {"plan": "1. grasp\n2. pour\n3. place"}
    if "Update the memory" in text:
        return {"memory": "poured once"}
    if "acknowledgement the robot" in text:
        return {"text": "Sure."}
    if "ONE realistic interruption" in text:
        return {"interjection": "use less water", "speech": "Using less water."}
    if "frame-grounded visual question" in text:
        return {"question": "How many cups?", "answer": {"label": "cup", "count": 1}}
    return None


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_dataset(Path(tmp) / "ds")
        vlm = StubVlmClient(responder=_stub_responder)
        cfg = AnnotationPipelineConfig()
        executor = Executor(
            config=cfg,
            module_1=PlanSubtasksMemoryModule(vlm=vlm, config=cfg.module_1),
            module_2=InterjectionsAndSpeechModule(vlm=vlm, config=cfg.module_2, seed=cfg.seed),
            module_3=GeneralVqaModule(vlm=vlm, config=cfg.module_3, seed=cfg.seed),
            writer=LanguageColumnsWriter(),
            validator=StagingValidator(),
        )
        summary = executor.run(root)
        print(f"phases={[(p.name, p.episodes_processed) for p in summary.phases]}")
        print(f"validation: {summary.validation_report.summary()}")
        print(f"shards rewritten: {len(summary.written_paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
