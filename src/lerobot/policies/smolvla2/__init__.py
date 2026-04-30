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
"""SmolVLA2 — SmolVLA with the SmolVLM language head re-enabled.

SmolVLA strips the LM head from the SmolVLM backbone because it only does
flow-matching action prediction. SmolVLA2 keeps the LM head so the same
model can train on the full Hi Robot / MEM / ECoT message blend defined in
the steerable annotation plan (PR1 + PR2):

* action-only sub-recipes (e.g. ``low_level_execution``) → flow loss
* text-only sub-recipes (e.g. ``memory_update``, ``ask_vqa``,
  ``user_interjection_response``, ``high_level_subtask``) → CE loss on
  ``lm_head`` over the recipe's target message tokens
* mixed sub-recipes → both losses summed (weighted)

The ``predict_actions`` toggle follows the Pi0.5 convention from Section
I.7 of the plan: ``True`` if any ``low_level`` target is present in the
sample, else ``False``.

This package is a thin subclass of ``lerobot.policies.smolvla`` so most of
the model code stays in one place — only the dual-loss path and the
chat-template processor live here.
"""

from .configuration_smolvla2 import SmolVLA2Config

__all__ = ["SmolVLA2Config"]
