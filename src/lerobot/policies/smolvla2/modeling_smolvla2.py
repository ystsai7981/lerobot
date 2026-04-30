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

Adds:

* an unfrozen SmolVLM ``lm_head`` so language tokens can be supervised,
* a forward path that runs the flow head, the text head, or both,
  driven by ``batch["predict_actions"]`` and ``batch["text_labels"]``
  produced by :class:`SmolVLA2ChatTokenizerStep` (the previous commit on
  this branch).

Per-sample routing — within one batch:

* ``predict_actions[i] = True`` ⇒ sample ``i`` contributes to the flow
  loss (action chunk supervision).
* ``predict_actions[i] = False`` ⇒ sample ``i`` is masked out of the
  flow loss; only its text tokens (where ``text_labels[i, t] != -100``)
  contribute to the LM-head cross-entropy.

Falls back to ``SmolVLAPolicy.forward`` cleanly when neither
``text_labels`` nor ``predict_actions`` is in the batch — unannotated
datasets keep working unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from .configuration_smolvla2 import SmolVLA2Config


class SmolVLA2Policy(SmolVLAPolicy):
    """SmolVLA + re-enabled SmolVLM language head."""

    config_class = SmolVLA2Config
    name = "smolvla2"

    def __init__(self, config: SmolVLA2Config, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        if not isinstance(config, SmolVLA2Config):
            config = SmolVLA2Config(
                **{
                    f.name: getattr(config, f.name)
                    for f in config.__dataclass_fields__.values()
                    if hasattr(config, f.name)
                }
            )
        super().__init__(config, dataset_stats=dataset_stats)
        if config.unfreeze_lm_head and config.text_loss_weight > 0:
            self._unfreeze_lm_head()

    # ------------------------------------------------------------------
    # Backbone surgery
    # ------------------------------------------------------------------

    def _unfreeze_lm_head(self) -> None:
        """Re-enable gradients on the SmolVLM ``lm_head`` (and the bits
        of the text path SmolVLA freezes) so the text-loss can flow back.
        """
        vlm_with_expert = getattr(self.model, "vlm_with_expert", None)
        if vlm_with_expert is None:
            return
        vlm = getattr(vlm_with_expert, "vlm", None)
        if vlm is None:
            return
        for name, param in vlm.named_parameters():
            if "lm_head" in name or "text_model.model.norm.weight" in name:
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
        """Forward pass with optional dual-head loss.

        Two routing knobs from the batch (produced by
        :class:`SmolVLA2ChatTokenizerStep`):

        * ``text_labels`` — per-token labels with ``-100`` for non-target
          positions. Triggers the text-loss path through ``lm_head``.
        * ``predict_actions`` — per-sample bool tensor. ``True`` ⇒
          include this sample's action chunk in the flow loss.

        When neither is present, delegate to ``SmolVLAPolicy.forward``.
        """
        text_labels = batch.get("text_labels")
        predict_actions_t = batch.get("predict_actions")

        has_text_data = (
            text_labels is not None
            and isinstance(text_labels, Tensor)
            and self.config.text_loss_weight > 0
        )
        has_per_sample_routing = (
            predict_actions_t is not None and isinstance(predict_actions_t, Tensor)
        )

        if not has_text_data and not has_per_sample_routing:
            return super().forward(batch, noise=noise, time=time, reduction=reduction)

        loss_dict: dict[str, Any] = {}
        device = batch[OBS_STATE].device
        total = torch.zeros((), device=device, dtype=torch.float32)

        # ------------------------------------------------------------
        # Flow loss path — only when at least one sample wants actions.
        # ------------------------------------------------------------
        run_flow = self.config.flow_loss_weight > 0 and (
            not has_per_sample_routing or bool(predict_actions_t.any().item())
        )
        if run_flow and ACTION in batch:
            per_sample_flow, flow_diag = super().forward(
                batch, noise=noise, time=time, reduction="none"
            )
            # ``per_sample_flow`` has shape (B,) from the SmolVLA
            # reduction="none" branch.
            if has_per_sample_routing:
                mask = predict_actions_t.to(per_sample_flow.dtype)
                masked = per_sample_flow * mask
                denom = mask.sum().clamp(min=1.0)
                flow_loss = masked.sum() / denom
            else:
                flow_loss = per_sample_flow.mean()
            total = total + self.config.flow_loss_weight * flow_loss
            loss_dict["flow_loss"] = float(flow_loss.detach().item())
            for k, v in flow_diag.items():
                loss_dict[f"flow_{k}"] = v

        # ------------------------------------------------------------
        # Text loss path — prefix-only forward → lm_head → CE.
        # ------------------------------------------------------------
        if has_text_data:
            text_loss = self._compute_text_loss(batch, text_labels)
            total = total + self.config.text_loss_weight * text_loss
            loss_dict["text_loss"] = float(text_loss.detach().item())

        loss_dict["loss"] = float(total.detach().item())

        if reduction == "none":
            # Per-sample loss isn't meaningfully defined for the dual
            # path; broadcast the scalar to (B,) for caller compat.
            return total.expand(batch[OBS_STATE].shape[0]), loss_dict
        return total, loss_dict

    # ------------------------------------------------------------------
    # Text-loss internals
    # ------------------------------------------------------------------

    def _compute_text_loss(self, batch: dict[str, Tensor], text_labels: Tensor) -> Tensor:
        """Cross-entropy on the SmolVLM ``lm_head`` over target tokens."""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Prefix-only forward.
        out_pair, _ = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=False,
        )
        prefix_out = out_pair[0] if isinstance(out_pair, (tuple, list)) else out_pair
        if prefix_out is None:
            raise RuntimeError(
                "SmolVLA2: vlm_with_expert.forward returned no prefix hidden "
                "states — text-loss path needs them."
            )

        # Lang token positions inside the prefix. ``embed_prefix`` lays
        # out the prefix as ``[image_blocks..., lang, state]`` so the
        # lang range is identifiable from the trailing state size and
        # the known lang length.
        num_lang = lang_tokens.shape[1]
        state_for_dim = state if state.ndim >= 2 else state[:, None]
        num_state = state_for_dim.shape[1] if state_for_dim.ndim >= 2 else 1
        if num_state < 1:
            num_state = 1
        prefix_len = prefix_out.shape[1]
        lang_end = prefix_len - num_state
        lang_start = lang_end - num_lang
        if lang_start < 0 or lang_end > prefix_len:
            raise RuntimeError(
                f"SmolVLA2: could not locate lang token range in prefix "
                f"(prefix_len={prefix_len}, num_lang={num_lang}, "
                f"num_state={num_state})."
            )

        lang_hidden = prefix_out[:, lang_start:lang_end]
        vlm = self.model.vlm_with_expert.vlm
        logits = vlm.lm_head(lang_hidden)  # (B, num_lang, vocab)

        if text_labels.shape[1] != num_lang:
            common = min(text_labels.shape[1], num_lang)
            logits = logits[:, :common]
            text_labels = text_labels[:, :common]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            text_labels.reshape(-1).long(),
            ignore_index=-100,
        )
        return loss
