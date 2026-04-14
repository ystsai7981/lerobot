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

"""Rollout strategy ABC and factory."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.rollout.configs import RolloutStrategyConfig
    from lerobot.rollout.context import RolloutContext


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires."""

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: save dataset, stop threads, disconnect hardware."""


def create_strategy(config: RolloutStrategyConfig) -> RolloutStrategy:
    """Instantiate the appropriate strategy from a config object."""
    from lerobot.rollout.configs import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    if isinstance(config, BaseStrategyConfig):
        from .base import BaseStrategy

        return BaseStrategy(config)
    if isinstance(config, SentryStrategyConfig):
        from .sentry import SentryStrategy

        return SentryStrategy(config)
    if isinstance(config, HighlightStrategyConfig):
        from .highlight import HighlightStrategy

        return HighlightStrategy(config)
    if isinstance(config, DAggerStrategyConfig):
        from .dagger import DAggerStrategy

        return DAggerStrategy(config)

    raise ValueError(f"Unknown strategy config type: {type(config).__name__}")
