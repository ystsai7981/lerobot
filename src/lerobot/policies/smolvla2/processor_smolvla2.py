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
"""SmolVLA2 processor pipelines.

SCAFFOLD: this currently delegates to SmolVLA's processor. The next
commit on this branch replaces that with a chat-template aware pipeline:

  RenderMessagesStep (PR1) → SmolVLA2ChatTokenizerStep → existing SmolVLA
  normalization / device steps.

The chat tokenizer step will:

* take ``messages`` / ``message_streams`` / ``target_message_indices``
  from the rendered sample,
* call ``apply_chat_template(messages, tools=DEFAULT_TOOLS, ...)`` on the
  SmolVLM tokenizer,
* tokenize the resulting prompt,
* build a ``text_labels`` tensor with ``-100`` everywhere except the
  token positions belonging to messages whose index is in
  ``target_message_indices``,
* derive ``predict_actions = bool(targets_by_stream.get("low_level"))``.
"""

from __future__ import annotations

from typing import Any

import torch

from ..smolvla.processor_smolvla import make_smolvla_pre_post_processors
from .configuration_smolvla2 import SmolVLA2Config


def make_smolvla2_pre_post_processors(
    config: SmolVLA2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Build SmolVLA2's pre/post-processor pipelines.

    SCAFFOLD: just delegates to ``make_smolvla_pre_post_processors`` so
    SmolVLA2 inherits SmolVLA's tokenization + normalization for now.
    The recipe-driven chat-template rendering arrives in the next commit.
    """
    return make_smolvla_pre_post_processors(config, dataset_stats=dataset_stats)
