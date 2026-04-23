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
from typing import TYPE_CHECKING

import torch

from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    CriticNetworkConfig,
    GaussianActorConfig,
)
from lerobot.rl.algorithms.configs import RLAlgorithmConfig

if TYPE_CHECKING:
    from lerobot.rl.algorithms.sac.sac_algorithm import SACAlgorithm


@RLAlgorithmConfig.register_subclass("sac")
@dataclass
class SACAlgorithmConfig(RLAlgorithmConfig):
    """SAC algorithm hyperparameters."""

    # Policy config
    sac_config: GaussianActorConfig

    # Optimizer learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4

    # Bellman update
    discount: float = 0.99
    use_backup_entropy: bool = True
    critic_target_update_weight: float = 0.005

    # Critic ensemble
    num_critics: int = 2
    num_subsample_critics: int | None = None
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    # Temperature / entropy
    temperature_init: float = 1.0

    # Update loop
    utd_ratio: int = 1
    policy_update_freq: int = 1
    grad_clip_norm: float = 40.0

    @classmethod
    def from_policy_config(cls, policy_cfg: GaussianActorConfig) -> SACAlgorithmConfig:
        """Build an algorithm config by copying hyperparameters from the policy config."""
        return cls(
            actor_lr=policy_cfg.actor_lr,
            critic_lr=policy_cfg.critic_lr,
            temperature_lr=policy_cfg.temperature_lr,
            discount=policy_cfg.discount,
            use_backup_entropy=policy_cfg.use_backup_entropy,
            critic_target_update_weight=policy_cfg.critic_target_update_weight,
            num_critics=policy_cfg.num_critics,
            num_subsample_critics=policy_cfg.num_subsample_critics,
            critic_network_kwargs=policy_cfg.critic_network_kwargs,
            discrete_critic_network_kwargs=policy_cfg.discrete_critic_network_kwargs,
            temperature_init=policy_cfg.temperature_init,
            utd_ratio=policy_cfg.utd_ratio,
            policy_update_freq=policy_cfg.policy_update_freq,
            grad_clip_norm=policy_cfg.grad_clip_norm,
            sac_config=policy_cfg,
        )

    def build_algorithm(self, policy: torch.nn.Module) -> SACAlgorithm:
        from lerobot.rl.algorithms.sac.sac_algorithm import SACAlgorithm

        return SACAlgorithm(policy=policy, config=self)
