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
"""SmolVLA2 modeling — dual-head subclass of SmolVLAPolicy.

This module defines :class:`SmolVLA2Policy`, which extends SmolVLA with:

* an unfrozen SmolVLM ``lm_head`` so language tokens can be supervised,
* a forward path that routes to the flow head, the text head, or both,
  driven by ``batch["predict_actions"]`` and ``batch["text_labels"]``.

The text-head computation itself is NOT wired up in this scaffold commit
(the processor doesn't yet produce ``text_labels`` either). This file is
the structural placeholder that:

1. registers the ``SmolVLA2Policy`` class with the right config name so
   ``policies/factory.py`` can build it,
2. unfreezes ``lm_head`` at construction time when the config asks for it
   (otherwise SmolVLA's ``train_expert_only`` freezes it again on every
   ``train()`` call),
3. forwards to ``SmolVLAPolicy.forward`` so behaviour is identical to
   SmolVLA when no text labels are present — i.e. existing SmolVLA
   training scripts keep working.

The next commit on this branch fills in the actual text-loss path.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_smolvla2 import SmolVLA2Config


class SmolVLA2Policy(SmolVLAPolicy):
    """SmolVLA + re-enabled SmolVLM language head.

    Compatible drop-in for ``SmolVLAPolicy`` from a checkpoint or factory
    perspective. Behaviourally identical to SmolVLA until the text-head
    code path lands in the next commit on this branch.
    """

    config_class = SmolVLA2Config
    name = "smolvla2"

    def __init__(self, config: SmolVLA2Config, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        if not isinstance(config, SmolVLA2Config):
            # Allow loading a SmolVLA checkpoint into a SmolVLA2 model by
            # widening the config type — the new fields fall back to their
            # defaults, which preserves the existing SmolVLA behaviour.
            config = SmolVLA2Config(**{
                f.name: getattr(config, f.name)
                for f in config.__dataclass_fields__.values()
                if hasattr(config, f.name)
            })
        super().__init__(config, dataset_stats=dataset_stats)
        if config.unfreeze_lm_head and config.text_loss_weight > 0:
            self._unfreeze_lm_head()

    # ------------------------------------------------------------------
    # Backbone surgery
    # ------------------------------------------------------------------

    def _unfreeze_lm_head(self) -> None:
        """Re-enable gradients on the SmolVLM ``lm_head`` (and the bits of
        the text path SmolVLA freezes) so the text-loss can flow back.

        SmolVLA's ``SmolVLMWithExpertModel.set_requires_grad`` freezes
        ``lm_head``, ``text_model.model.norm.weight``, and the last
        ``text_model.layers.<N-1>`` block. We undo that selectively when
        text training is enabled.
        """
        vlm_with_expert = getattr(self.model, "vlm_with_expert", None)
        if vlm_with_expert is None:
            return
        vlm = getattr(vlm_with_expert, "vlm", None)
        if vlm is None:
            return
        for name, param in vlm.named_parameters():
            if (
                "lm_head" in name
                or "text_model.model.norm.weight" in name
            ):
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        """Forward pass with optional text-head loss.

        SCAFFOLD: forwards directly to ``SmolVLAPolicy.forward``. The
        actual text-loss / dual-head routing lands in the next commit on
        this branch — it will read ``batch["text_labels"]`` and
        ``batch["predict_actions"]`` (both produced by the SmolVLA2
        processor) to decide which head(s) to run.
        """
        return super().forward(batch, noise=noise, time=time, reduction=reduction)
