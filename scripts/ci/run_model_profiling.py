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
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


@dataclass(frozen=True)
class ProfilingSpec:
    name: str
    steps: int
    train_args: list[str]


@dataclass(frozen=True)
class UploadTarget:
    local_path: Path
    path_in_repo: str


def utc_timestamp_slug(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y%m%dT%H%M%SZ")


def make_hub_file_url(repo_id: str, path_in_repo: str, repo_type: str = "dataset") -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/main/{path_in_repo}"


def upload_targets(
    repo_id: str,
    targets: list[UploadTarget],
    *,
    repo_type: str = "dataset",
    token: str | None = None,
    private: bool | None = None,
    commit_message: str | None = None,
) -> dict[str, str]:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    uploaded: dict[str, str] = {}
    for target in targets:
        api.upload_file(
            path_or_fileobj=str(target.local_path),
            path_in_repo=target.path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {target.path_in_repo}",
        )
        uploaded[target.path_in_repo] = make_hub_file_url(repo_id, target.path_in_repo, repo_type=repo_type)
    return uploaded


def normalize_repo_id(repo: str, hub_org: str) -> str:
    return repo if "/" in repo else f"{hub_org}/{repo}"


def load_specs(path: Path) -> dict[str, ProfilingSpec]:
    payload = json.loads(path.read_text())
    return {
        name: ProfilingSpec(name=name, steps=spec["steps"], train_args=spec["train_args"])
        for name, spec in payload.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-file", type=Path, default=Path("profiling/model_profiling_specs.json"))
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hub_org", default="lerobot")
    parser.add_argument("--results_repo", default="model-profiling-history")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--profile_mode", choices=["summary", "trace"], default="trace")
    parser.add_argument("--git_commit", default="")
    return parser.parse_args()


def get_selected_names(requested: list[str] | None, specs: dict[str, ProfilingSpec]) -> list[str]:
    if not requested:
        return list(specs)
    unknown = sorted(set(requested) - set(specs))
    if unknown:
        raise ValueError(f"Unknown profiling policies: {', '.join(unknown)}")
    return requested


def build_train_command(spec: ProfilingSpec, run_dir: Path, profile_mode: str) -> list[str]:
    train_output_dir = run_dir / "train"
    profile_output_dir = run_dir / "profiling"
    return [
        "uv",
        "run",
        "lerobot-train",
        *spec.train_args,
        f"--output_dir={train_output_dir}",
        f"--steps={spec.steps}",
        "--eval_freq=0",
        "--save_checkpoint=false",
        f"--save_freq={spec.steps}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
        "--num_workers=0",
        "--log_freq=1",
        "--cudnn_deterministic=true",
        f"--profile_mode={profile_mode}",
        f"--profile_output_dir={profile_output_dir}",
    ]


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_artifact_index(
    *,
    repo_id: str,
    run_dir: Path,
    policy_name: str,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[UploadTarget], str]:
    row_path_in_repo = f"rows/{policy_name}/{run_id}.json"
    artifact_root = f"artifacts/{policy_name}/{run_id}"
    artifact_paths: dict[str, Any] = {
        "row": row_path_in_repo,
        "profiling_files": {},
        "cprofile_summaries": {},
        "torch_tables": {},
        "trace_files": {},
    }
    artifact_urls: dict[str, Any] = {
        "row": make_hub_file_url(repo_id, row_path_in_repo),
        "profiling_files": {},
        "cprofile_summaries": {},
        "torch_tables": {},
        "trace_files": {},
    }
    targets: list[UploadTarget] = []

    for name in ("stdout.txt", "stderr.txt"):
        path = run_dir / name
        if not path.exists():
            continue
        repo_path = f"{artifact_root}/{name}"
        artifact_paths[name.removesuffix(".txt")] = repo_path
        artifact_urls[name.removesuffix(".txt")] = make_hub_file_url(repo_id, repo_path)
        targets.append(UploadTarget(local_path=path, path_in_repo=repo_path))

    profiling_dir = run_dir / "profiling"
    for path in sorted(profiling_dir.rglob("*")) if profiling_dir.exists() else []:
        if not path.is_file():
            continue
        relative_path = str(path.relative_to(run_dir))
        repo_path = f"{artifact_root}/{relative_path}"
        artifact_paths["profiling_files"][relative_path] = repo_path
        artifact_urls["profiling_files"][relative_path] = make_hub_file_url(repo_id, repo_path)
        targets.append(UploadTarget(local_path=path, path_in_repo=repo_path))

        if path.name == "step_timing_summary.json":
            artifact_paths["step_timing_summary"] = repo_path
            artifact_urls["step_timing_summary"] = make_hub_file_url(repo_id, repo_path)
        elif "cprofile" in path.parts:
            artifact_paths["cprofile_summaries"][path.stem] = repo_path
            artifact_urls["cprofile_summaries"][path.stem] = make_hub_file_url(repo_id, repo_path)
        elif "torch_tables" in path.parts:
            artifact_paths["torch_tables"][path.name] = repo_path
            artifact_urls["torch_tables"][path.name] = make_hub_file_url(repo_id, repo_path)
        elif "torch_traces" in path.parts:
            artifact_paths["trace_files"][path.name] = repo_path
            artifact_urls["trace_files"][path.name] = make_hub_file_url(repo_id, repo_path)

    return artifact_paths, artifact_urls, targets, row_path_in_repo


def upload_profile_run(
    *,
    repo_id: str,
    row_path: Path,
    row_path_in_repo: str,
    artifact_targets: list[UploadTarget],
) -> dict[str, str]:
    return upload_targets(
        repo_id=repo_id,
        targets=[*artifact_targets, UploadTarget(local_path=row_path, path_in_repo=row_path_in_repo)],
        repo_type="dataset",
        private=False,
        commit_message=f"Add model profiling row {row_path_in_repo}",
    )


def main() -> int:
    args = parse_args()
    specs = load_specs(args.spec_file)
    selected = get_selected_names(args.policies, specs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_id = normalize_repo_id(args.results_repo, args.hub_org)
    git_commit = args.git_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    for policy_name in selected:
        spec = specs[policy_name]
        run_id = f"{utc_timestamp_slug()}__{policy_name}"
        run_dir = args.output_dir / policy_name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(spec, run_dir, args.profile_mode)

        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration_s = time.perf_counter() - start

        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)

        profile_summary = load_json_if_exists(run_dir / "profiling" / "step_timing_summary.json") or {}
        deterministic_forward = (
            load_json_if_exists(run_dir / "profiling" / "deterministic_forward.json") or {}
        )
        artifact_paths, artifact_urls, artifact_targets, row_path_in_repo = build_artifact_index(
            repo_id=repo_id,
            run_dir=run_dir,
            policy_name=policy_name,
            run_id=run_id,
        )
        row = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "policy": policy_name,
            "git_commit": git_commit,
            "status": "success" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "profile_mode": args.profile_mode,
            "wall_time_s": duration_s,
            "spec": {
                "steps": spec.steps,
                "train_args": spec.train_args,
            },
            "step_timing_summary": profile_summary,
            "deterministic_forward": deterministic_forward,
            "artifact_paths": artifact_paths,
            "artifact_urls": artifact_urls,
            "stderr_tail": result.stderr.splitlines()[-20:],
        }

        row_path = run_dir / "profiling_row.json"
        row_path.write_text(json.dumps(row, indent=2, sort_keys=True))

        if args.publish:
            uploaded_paths = upload_profile_run(
                repo_id=repo_id,
                row_path=row_path,
                row_path_in_repo=row_path_in_repo,
                artifact_targets=artifact_targets,
            )
            row["uploaded_paths"] = uploaded_paths
            row_path.write_text(json.dumps(row, indent=2, sort_keys=True))

        print(json.dumps(row, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
