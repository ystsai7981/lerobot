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

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline when its local action
    queue is empty, postprocesses the whole predicted chunk immediately,
    and then returns one already-postprocessed CPU action at a time.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._processed_action_queue: deque[torch.Tensor] = deque()

        self._relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        if self._relative_step is not None and self._relative_step.action_names is None:
            cfg_names = getattr(policy.config, "action_feature_names", None)
            action_names = cfg_names or dataset_features.get(ACTION, {}).get("names")
            if action_names:
                self._relative_step.action_names = list(action_names)
            logger.info("Relative actions enabled: sync chunks will be postprocessed before queueing")

        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._processed_action_queue.clear()

    def _enqueue_processed_chunk(self, action_chunk: torch.Tensor) -> None:
        """Queue postprocessed per-step actions in policy output order."""
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)

        n_action_steps = getattr(self._policy.config, "n_action_steps", action_chunk.shape[1])
        action_chunk = action_chunk[:, : min(n_action_steps, action_chunk.shape[1])]

        for action in action_chunk.squeeze(0):
            action_tensor = action.detach().cpu()
            if len(action_tensor) != len(self._ordered_action_keys):
                raise ValueError(
                    f"Action tensor length ({len(action_tensor)}) != action keys "
                    f"({len(self._ordered_action_keys)})"
                )
            self._processed_action_queue.append(action_tensor)

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if self._processed_action_queue:
            return self._processed_action_queue.popleft().clone()
        if obs_frame is None:
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            action_chunk = self._policy.predict_action_chunk(observation)
            processed_chunk = self._postprocessor(action_chunk)

        self._enqueue_processed_chunk(processed_chunk)
        if not self._processed_action_queue:
            return None
        return self._processed_action_queue.popleft().clone()
