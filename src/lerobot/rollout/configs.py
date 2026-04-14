# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration dataclasses for the rollout deployment engine."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus

from lerobot.configs import PreTrainedConfig, parser
from lerobot.policies.rtc import RTCConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy configs (polymorphic dispatch via draccus ChoiceRegistry)
# ---------------------------------------------------------------------------


@dataclass
class RolloutStrategyConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for rollout strategy configurations.

    Use ``--strategy.type=<name>`` on the CLI to select a strategy.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@RolloutStrategyConfig.register_subclass("base")
@dataclass
class BaseStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with no data recording."""

    pass


@RolloutStrategyConfig.register_subclass("sentry")
@dataclass
class SentryStrategyConfig(RolloutStrategyConfig):
    """Continuous autonomous rollout with always-on recording.

    Episodes are auto-rotated every ``episode_duration_s`` seconds and
    uploaded in the background every ``upload_every_n_episodes`` episodes.
    """

    episode_duration_s: float = 120.0
    upload_every_n_episodes: int = 5


@RolloutStrategyConfig.register_subclass("highlight")
@dataclass
class HighlightStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with on-demand recording via ring buffer.

    A memory-bounded ring buffer continuously captures telemetry.  When
    the user presses the save key, the buffer contents are flushed to
    the dataset and live recording continues until the key is pressed
    again.
    """

    ring_buffer_seconds: float = 30.0
    ring_buffer_max_memory_mb: float = 2048.0
    save_key: str = "s"


@RolloutStrategyConfig.register_subclass("dagger")
@dataclass
class DAggerStrategyConfig(RolloutStrategyConfig):
    """Human-in-the-loop data collection (DAgger / RaC).

    Alternates between autonomous policy execution and human intervention.
    Intervention frames are tagged with ``intervention=True``.
    """

    episode_time_s: float = 120.0
    num_episodes: int = 50
    play_sounds: bool = True
    calibrate: bool = False
    log_hz: bool = True
    hz_log_interval_s: float = 2.0


# ---------------------------------------------------------------------------
# Dataset recording config (shared across recording strategies)
# ---------------------------------------------------------------------------


@dataclass
class RolloutDatasetConfig:
    """Dataset configuration for rollout strategies that record data."""

    repo_id: str = ""
    single_task: str = ""
    root: str | Path | None = None
    fps: int = 30
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "auto"
    streaming_encoding: bool = True
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None
    rename_map: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level rollout config
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    """Top-level configuration for the ``lerobot-rollout`` CLI.

    Combines hardware, policy, strategy, and runtime settings.  The
    ``__post_init__`` method performs fail-fast validation to reject
    invalid flag combinations early.
    """

    # Hardware
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None

    # Policy (loaded from --policy.path via __post_init__)
    policy: PreTrainedConfig | None = None

    # Strategy (polymorphic: --strategy.type=base|sentry|highlight|dagger)
    strategy: RolloutStrategyConfig = field(default_factory=BaseStrategyConfig)

    # Inference backend
    rtc: RTCConfig = field(default_factory=RTCConfig)

    # Dataset (required for sentry, highlight, dagger; None for base)
    dataset: RolloutDatasetConfig | None = None

    # Runtime
    fps: float = 30.0
    duration: float = 0.0  # 0 = infinite (24/7 mode)
    interpolation_multiplier: int = 1
    device: str | None = None
    task: str = ""
    display_data: bool = False
    resume: bool = False

    # Torch compile
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    compile_warmup_inferences: int = 2

    def __post_init__(self):
        # --- Policy loading (same pattern as existing scripts) ---
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("--policy.path is required for rollout")

        if self.robot is None:
            raise ValueError("--robot.type is required for rollout")

        # --- Strategy-specific validation ---
        if isinstance(self.strategy, DAggerStrategyConfig) and self.teleop is None:
            raise ValueError("DAgger strategy requires --teleop.type to be set")

        needs_dataset = isinstance(
            self.strategy, (SentryStrategyConfig, HighlightStrategyConfig, DAggerStrategyConfig)
        )
        if needs_dataset and (self.dataset is None or not self.dataset.repo_id):
            raise ValueError(f"{self.strategy.type} strategy requires --dataset.repo_id to be set")

        if isinstance(self.strategy, BaseStrategyConfig) and self.dataset is not None:
            raise ValueError(
                "Base strategy does not record data. Use sentry, highlight, or dagger for recording."
            )

        # Sentry MUST use streaming encoding to avoid disk I/O blocking the control loop
        if isinstance(self.strategy, SentryStrategyConfig) and self.dataset is not None:
            if not self.dataset.streaming_encoding:
                logger.warning("Sentry mode forces streaming_encoding=True")
                self.dataset.streaming_encoding = True

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]
