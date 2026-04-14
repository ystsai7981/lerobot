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

"""Strategy factory: config type-name → strategy class dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import RolloutStrategy

if TYPE_CHECKING:
    from lerobot.rollout.configs import RolloutStrategyConfig


def _lazy_strategy_map() -> dict[str, type[RolloutStrategy]]:
    """Build the strategy type-name → class mapping with lazy imports."""
    from .base import BaseStrategy
    from .dagger import DAggerStrategy
    from .highlight import HighlightStrategy
    from .sentry import SentryStrategy

    return {
        "base": BaseStrategy,
        "sentry": SentryStrategy,
        "highlight": HighlightStrategy,
        "dagger": DAggerStrategy,
    }


def create_strategy(config: RolloutStrategyConfig) -> RolloutStrategy:
    """Instantiate the appropriate strategy from a config object.

    Uses ``config.type`` (the name registered via ``draccus.ChoiceRegistry``)
    to look up the strategy class, so adding a new strategy only requires
    registering its config subclass and adding one entry to
    ``_lazy_strategy_map``.
    """
    strategy_map = _lazy_strategy_map()
    strategy_cls = strategy_map.get(config.type)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy type '{config.type}'. Available: {sorted(strategy_map.keys())}")
    return strategy_cls(config)
