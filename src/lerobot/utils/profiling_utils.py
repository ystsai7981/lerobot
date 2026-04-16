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

import cProfile
import hashlib
import io
import json
import pstats
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_cprofile_summary(
    profile: cProfile.Profile, *, sort_by: str = "cumulative", limit: int = 40
) -> str:
    output = io.StringIO()
    stats = pstats.Stats(profile, stream=output).strip_dirs().sort_stats(sort_by)
    stats.print_stats(limit)
    return output.getvalue()


def write_profiler_table(
    profiler: Any,
    output_path: Path,
    *,
    sort_by: str,
    row_limit: int = 40,
) -> None:
    try:
        table = profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    except Exception:
        return
    output_path.write_text(table)


def make_torch_profiler(cfg: Any, output_dir: Path, device_type: str) -> Any:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_dir = ensure_dir(output_dir / "torch_traces")

    def _trace_ready(profiler: Any) -> None:
        if cfg.profile_mode != "trace":
            return
        profiler.export_chrome_trace(str(trace_dir / f"trace_step_{profiler.step_num}.json"))

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=cfg.profile_wait_steps,
            warmup=cfg.profile_warmup_steps,
            active=cfg.profile_active_steps,
            repeat=cfg.profile_repeat,
        ),
        on_trace_ready=_trace_ready,
        record_shapes=cfg.profile_record_shapes,
        profile_memory=cfg.profile_with_memory,
        with_flops=cfg.profile_with_flops,
        with_stack=cfg.profile_with_stack,
    )


def write_torch_profiler_outputs(
    profiler: Any,
    output_dir: Path,
    *,
    device_type: str,
) -> None:
    tables_dir = ensure_dir(output_dir / "torch_tables")
    write_profiler_table(profiler, tables_dir / "cpu_time_total.txt", sort_by="cpu_time_total")
    if device_type == "cuda":
        write_profiler_table(profiler, tables_dir / "cuda_time_total.txt", sort_by="self_cuda_time_total")
        write_profiler_table(profiler, tables_dir / "cuda_memory.txt", sort_by="self_cuda_memory_usage")
    write_profiler_table(profiler, tables_dir / "cpu_memory.txt", sort_by="self_cpu_memory_usage")
    write_profiler_table(profiler, tables_dir / "flops.txt", sort_by="flops")


def run_with_cprofile[T](
    label: str,
    output_dir: Path,
    fn: Callable[..., T],
    *args: Any,
    sort_by: str = "cumulative",
    limit: int = 40,
    **kwargs: Any,
) -> T:
    ensure_dir(output_dir)
    profile = cProfile.Profile()
    profile.enable()
    try:
        return fn(*args, **kwargs)
    finally:
        profile.disable()
        summary = render_cprofile_summary(profile, sort_by=sort_by, limit=limit)
        (output_dir / f"{label}.txt").write_text(summary)


def _stable_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 8)


def _tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.numel() == 0:
        stats = {"sum": None, "mean": None, "std": None, "min": None, "max": None}
    else:
        stats_tensor = (
            cpu_tensor.to(torch.float64) if cpu_tensor.is_floating_point() else cpu_tensor.to(torch.int64)
        )
        stats = {
            "sum": _stable_float(stats_tensor.sum().item()),
            "mean": _stable_float(stats_tensor.float().mean().item()),
            "std": _stable_float(stats_tensor.float().std(unbiased=False).item())
            if cpu_tensor.numel() > 1
            else 0.0,
            "min": _stable_float(stats_tensor.min().item()),
            "max": _stable_float(stats_tensor.max().item()),
        }
    hash_tensor = cpu_tensor.float() if cpu_tensor.dtype == torch.bfloat16 else cpu_tensor
    digest = hashlib.sha256(hash_tensor.contiguous().numpy().tobytes()).hexdigest()
    return {
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "numel": cpu_tensor.numel(),
        "sha256": digest,
        **stats,
    }


def _summarize_forward_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, dict):
        return {key: _summarize_forward_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_forward_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _build_reference_batch(dataset: Any, batch_size: int) -> Any:
    if len(dataset) == 0:
        raise ValueError("Cannot build a reference batch from an empty dataset.")
    indices = [idx % len(dataset) for idx in range(batch_size)]
    samples = [dataset[idx] for idx in indices]
    return default_collate(samples)


def write_deterministic_forward_artifacts(
    *,
    policy: Any,
    dataset: Any,
    batch_size: int,
    preprocessor: Any,
    output_dir: Path,
    device_type: str,
) -> None:
    reference_batch = preprocessor(_build_reference_batch(dataset, batch_size))
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Keep the caller-selected module mode so the fingerprint matches the actual
    # train-path forward used by the policy. Some policies, such as ACT with VAE,
    # only materialize their full forward outputs while in training mode.
    with torch.random.fork_rng(devices=[] if device_type != "cuda" else None):
        torch.manual_seed(0)
        if device_type == "cuda":
            torch.cuda.manual_seed_all(0)
        with torch.no_grad(), torch.profiler.profile(activities=activities) as profiler:
            loss, output_dict = policy.forward(reference_batch)

    operator_entries = []
    for event in profiler.key_averages():
        entry = {
            "key": event.key,
            "count": event.count,
            "cpu_time_total_us": _stable_float(getattr(event, "cpu_time_total", None)),
        }
        if device_type == "cuda":
            entry["self_cuda_time_total_us"] = _stable_float(getattr(event, "self_cuda_time_total", None))
        operator_entries.append(entry)
    operator_entries = sorted(operator_entries, key=lambda item: item["key"])

    output_summary = {
        "loss": _summarize_forward_value(loss),
        "output_dict": _summarize_forward_value(output_dict),
    }
    payload = {
        "seed": 0,
        "reference_batch_size": batch_size,
        "operator_fingerprint": _hash_payload([(entry["key"], entry["count"]) for entry in operator_entries]),
        "output_fingerprint": _hash_payload(output_summary),
        "operators": operator_entries,
        "outputs": output_summary,
    }
    (output_dir / "deterministic_forward.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    table_sort = "self_cuda_time_total" if device_type == "cuda" else "cpu_time_total"
    write_profiler_table(profiler, output_dir / "deterministic_forward_ops.txt", sort_by=table_sort)


def _summary(values: list[float]) -> dict[str, float] | dict[str, None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


@dataclass
class StepTimingCollector:
    forward_s: list[float] = field(default_factory=list)
    backward_s: list[float] = field(default_factory=list)
    optimizer_s: list[float] = field(default_factory=list)
    total_update_s: list[float] = field(default_factory=list)
    dataloading_s: list[float] = field(default_factory=list)
    memory_timeline: list[dict[str, float | int]] = field(default_factory=list)

    def record(
        self,
        *,
        forward_s: float,
        backward_s: float,
        optimizer_s: float,
        total_update_s: float,
    ) -> None:
        self.forward_s.append(forward_s)
        self.backward_s.append(backward_s)
        self.optimizer_s.append(optimizer_s)
        self.total_update_s.append(total_update_s)

    def record_dataloading(self, dataloading_s: float) -> None:
        self.dataloading_s.append(dataloading_s)

    def record_memory(self, *, step: int, allocated_bytes: int, reserved_bytes: int) -> None:
        self.memory_timeline.append(
            {
                "step": step,
                "allocated_bytes": allocated_bytes,
                "reserved_bytes": reserved_bytes,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_s": _summary(self.forward_s),
            "backward_s": _summary(self.backward_s),
            "optimizer_s": _summary(self.optimizer_s),
            "total_update_s": _summary(self.total_update_s),
            "dataloading_s": _summary(self.dataloading_s),
            "memory_timeline": self.memory_timeline,
        }

    def write_json(self, output_path: Path, extra: dict[str, Any] | None = None) -> None:
        payload = self.to_dict()
        if extra:
            payload.update(extra)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
