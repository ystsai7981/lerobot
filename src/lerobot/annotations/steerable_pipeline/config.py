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

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Module1Config:
    """Module 1 hyperparameters: plan + subtasks + memory.

    Subtask decomposition sees the **whole episode** as one Qwen-VL video
    block — no keyframe stride or count: the model handles temporal pooling
    itself and decides where to cut. ``max_video_frames`` only caps the
    number of frames packed into the video block (a model-capacity bound,
    not an annotation-logic knob).
    """

    enabled: bool = True
    max_video_frames: int = 32
    min_subtask_seconds: float = 1.5
    plan_max_steps: int = 8


@dataclass
class Module2Config:
    """Module 2 hyperparameters: interjections + paired speech."""

    enabled: bool = True
    max_interjections_per_episode: int = 1
    interjection_min_t: float = 2.0


@dataclass
class Module3Config:
    """Module 3 hyperparameters: general VQA."""

    enabled: bool = True
    vqa_emission_hz: float = 1.0
    K: int = 3
    question_types: tuple[str, ...] = ("bbox", "keypoint", "count", "attribute", "spatial")


@dataclass
class VlmConfig:
    """Shared Qwen-VL client configuration."""

    backend: str = "vllm"
    """One of ``vllm``, ``transformers``, or ``stub`` (tests only)."""
    model_id: str = "Qwen/Qwen3.6-27B-FP8"
    max_new_tokens: int = 512
    temperature: float = 0.2
    json_mode: bool = True
    batch_size: int = 4
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    """Fraction of GPU memory vllm allocates for weights + KV cache.
    Lower (e.g. 0.7) when the vision encoder needs cuDNN workspace, or to
    avoid CUDNN_STATUS_NOT_INITIALIZED on tight VRAM (30B BF16 on 80 GB)."""
    max_model_len: int | None = None
    """Cap context length. ``None`` keeps the model's default; on H100 80 GB
    a 30B BF16 model often needs ``max_model_len=8192`` or smaller to leave
    room for KV cache."""
    trust_remote_code: bool = False
    """Pass ``trust_remote_code`` to HF auto-classes. Default ``False`` —
    only enable for models that actually ship custom code in their repo
    (rare for first-class VL releases). On Qwen3-VL it triggers an
    std::bad_alloc post-load even though the official transformers class
    is sufficient, so leaving this off is safest."""
    camera_key: str | None = None
    """Override the camera stream used for keyframe attachment. ``None`` picks
    the first ``observation.images.*`` key the dataset declares."""


@dataclass
class ExecutorConfig:
    """Executor selection and SLURM hyperparameters."""

    auto_threshold: int = 32
    force_local: bool = False
    slurm_partition: str | None = None
    slurm_gpus: int = 1
    slurm_time: str = "06:00:00"
    workers: int = 1


@dataclass
class AnnotationPipelineConfig:
    """Top-level config for ``lerobot-annotate``.

    Mirrors the structure of :class:`lerobot.configs.train.TrainPipelineConfig`:
    a draccus-parsed dataclass that contains nested per-module sub-configs and
    leaves the dataset, executor, and VLM choices independently knobbable.

    Output is always in-place: the writer rewrites ``data/chunk-*/file-*.parquet``
    in place. Multiple revisions of the same dataset live in separate copies.
    """

    repo_id: str | None = None
    root: Path | None = None

    staging_dir: Path | None = None
    """If unset, defaults to ``<root>/.annotate_staging/``."""

    seed: int = 1729

    module_1: Module1Config = field(default_factory=Module1Config)
    module_2: Module2Config = field(default_factory=Module2Config)
    module_3: Module3Config = field(default_factory=Module3Config)

    vlm: VlmConfig = field(default_factory=VlmConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)

    skip_validation: bool = False
    only_episodes: tuple[int, ...] | None = None

    def resolved_staging_dir(self, root: Path) -> Path:
        return self.staging_dir if self.staging_dir is not None else root / ".annotate_staging"
