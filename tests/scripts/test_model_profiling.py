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

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


def _import_model_profiling_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "run_model_profiling.py"
    module_name = "tests.scripts.run_model_profiling"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profiling_specs_cover_expected_policies():
    module = _import_model_profiling_script()
    spec_path = Path(__file__).resolve().parents[2] / "profiling" / "model_profiling_specs.json"
    specs = module.load_specs(spec_path)

    assert set(specs) == {
        "act",
        "diffusion",
        "groot",
        "multi_task_dit",
        "pi0",
        "pi0_fast",
        "pi05",
        "smolvla",
        "wall_x",
        "xvla",
    }
    for excluded in ("sac", "sarm", "tdmpc", "vqbet", "reward_classifier"):
        assert excluded not in specs


def test_build_train_command_includes_profiling_outputs(tmp_path):
    module = _import_model_profiling_script()
    spec_path = Path(__file__).resolve().parents[2] / "profiling" / "model_profiling_specs.json"
    spec = module.load_specs(spec_path)["act"]

    cmd = module.build_train_command(spec, tmp_path / "run", "trace")

    assert cmd[:3] == ["uv", "run", "lerobot-train"]
    assert any(arg.startswith("--output_dir=") for arg in cmd)
    assert any(arg.startswith("--profile_output_dir=") for arg in cmd)
    assert "--profile_mode=trace" in cmd
    assert "--eval_freq=0" in cmd
    assert "--cudnn_deterministic=true" in cmd


def test_build_artifact_index_collects_cprofile_tables_and_traces(tmp_path):
    module = _import_model_profiling_script()
    run_dir = tmp_path / "act" / "20260415T000000Z__act"
    profiling_dir = run_dir / "profiling"
    (profiling_dir / "cprofile").mkdir(parents=True, exist_ok=True)
    (profiling_dir / "torch_tables").mkdir(parents=True, exist_ok=True)
    (profiling_dir / "torch_traces").mkdir(parents=True, exist_ok=True)
    (profiling_dir / "step_timing_summary.json").write_text("{}")
    (profiling_dir / "deterministic_forward.json").write_text(
        json.dumps({"operator_fingerprint": "ops123", "output_fingerprint": "out123"})
    )
    (profiling_dir / "cprofile" / "policy_setup.txt").write_text("policy setup")
    (profiling_dir / "torch_tables" / "cpu_time_total.txt").write_text("cpu table")
    (profiling_dir / "torch_traces" / "trace_step_9.json").write_text("{}")
    (run_dir / "stdout.txt").write_text("stdout")
    (run_dir / "stderr.txt").write_text("stderr")

    artifact_paths, artifact_urls, targets, row_path_in_repo = module.build_artifact_index(
        repo_id="lerobot/model-profiling-history",
        run_dir=run_dir,
        policy_name="act",
        run_id="20260415T000000Z__act",
    )

    assert row_path_in_repo == "rows/act/20260415T000000Z__act.json"
    assert artifact_paths["stdout"].endswith("/stdout.txt")
    assert artifact_paths["step_timing_summary"].endswith("/profiling/step_timing_summary.json")
    assert "policy_setup" in artifact_paths["cprofile_summaries"]
    assert "cpu_time_total.txt" in artifact_paths["torch_tables"]
    assert "trace_step_9.json" in artifact_paths["trace_files"]
    assert artifact_paths["profiling_files"]["profiling/deterministic_forward.json"].endswith(
        "/profiling/deterministic_forward.json"
    )
    assert artifact_urls["row"].startswith("https://huggingface.co/datasets/lerobot/model-profiling-history/")
    assert len(targets) == 7


def test_model_profiling_main_smoke_writes_row(monkeypatch, tmp_path):
    module = _import_model_profiling_script()

    spec_file = tmp_path / "specs.json"
    spec_file.write_text(
        json.dumps(
            {
                "act": {
                    "steps": 4,
                    "train_args": [
                        "--dataset.repo_id=lerobot/pusht",
                        "--dataset.episodes=[0]",
                        "--policy.type=act",
                        "--policy.device=cuda",
                        "--batch_size=4",
                    ],
                }
            }
        )
    )
    args = argparse.Namespace(
        spec_file=spec_file,
        policies=["act"],
        output_dir=tmp_path / "results",
        hub_org="lerobot",
        results_repo="model-profiling-history",
        publish=False,
        profile_mode="summary",
        git_commit="",
    )

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "deadbeef\n")

    def _fake_run(cmd, capture_output, text):
        assert capture_output is True
        assert text is True
        profile_dir = Path(
            next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--profile_output_dir="))
        )
        (profile_dir / "cprofile").mkdir(parents=True, exist_ok=True)
        (profile_dir / "torch_tables").mkdir(parents=True, exist_ok=True)
        (profile_dir / "step_timing_summary.json").write_text(
            json.dumps(
                {
                    "forward_s": {"count": 1, "mean": 0.1, "median": 0.1, "min": 0.1, "max": 0.1},
                    "total_update_s": {"count": 1, "mean": 0.3, "median": 0.3, "min": 0.3, "max": 0.3},
                    "peak_memory_allocated_bytes": 1024,
                }
            )
        )
        (profile_dir / "deterministic_forward.json").write_text(
            json.dumps(
                {
                    "operator_fingerprint": "ops-fingerprint",
                    "output_fingerprint": "output-fingerprint",
                }
            )
        )
        (profile_dir / "cprofile" / "policy_setup.txt").write_text("policy setup profile")
        (profile_dir / "torch_tables" / "cpu_time_total.txt").write_text("cpu time table")
        return subprocess.CompletedProcess(cmd, 0, "stdout ok", "")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    assert module.main() == 0

    row_paths = list((tmp_path / "results").rglob("profiling_row.json"))
    assert len(row_paths) == 1
    row = json.loads(row_paths[0].read_text())
    assert row["policy"] == "act"
    assert row["status"] == "success"
    assert row["git_commit"] == "deadbeef"
    assert row["step_timing_summary"]["forward_s"]["mean"] == 0.1
    assert row["deterministic_forward"]["operator_fingerprint"] == "ops-fingerprint"
    assert "policy_setup" in row["artifact_paths"]["cprofile_summaries"]


def test_deterministic_forward_artifacts_preserve_policy_mode(tmp_path):
    from lerobot.utils.profiling_utils import write_deterministic_forward_artifacts

    class _TrainingOnlyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, batch):
            self.forward_calls += 1
            assert self.training
            return batch["value"].sum(), {"value": batch["value"]}

    dataset = [{"value": torch.tensor([1.0, 2.0])}]
    policy = _TrainingOnlyPolicy()
    policy.train()

    write_deterministic_forward_artifacts(
        policy=policy,
        dataset=dataset,
        batch_size=2,
        preprocessor=lambda batch: batch,
        output_dir=tmp_path,
        device_type="cpu",
    )

    payload = json.loads((tmp_path / "deterministic_forward.json").read_text())
    assert policy.training is True
    assert policy.forward_calls == 1
    assert payload["reference_batch_size"] == 2
    assert "operator_fingerprint" in payload
    assert payload["outputs"]["loss"]["numel"] == 1


def test_step_timing_collector_accepts_metric_like_values(tmp_path):
    from lerobot.utils.profiling_utils import StepTimingCollector

    class _MetricLike:
        def __init__(self, val):
            self.val = val

    collector = StepTimingCollector()
    collector.record(
        forward_s=0.1,
        backward_s=0.2,
        optimizer_s=0.3,
        total_update_s=_MetricLike(0.6),
    )
    collector.record_dataloading(_MetricLike(0.05))
    collector.write_json(tmp_path / "step_timing_summary.json")

    payload = json.loads((tmp_path / "step_timing_summary.json").read_text())
    assert payload["total_update_s"]["mean"] == 0.6
    assert payload["dataloading_s"]["mean"] == 0.05
