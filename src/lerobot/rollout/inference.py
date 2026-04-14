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

"""Unified inference engine supporting both synchronous and RTC backends.

The :class:`InferenceEngine` abstracts whether prediction happens inline
(sync) or in a background thread (RTC), so rollout strategies don't need
to branch on the inference backend.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from copy import copy
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    to_relative_actions,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.feature_utils import build_dataset_frame

from .robot_wrapper import ThreadSafeRobot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RTC helpers (extracted from examples/rtc and examples/hil)
# ---------------------------------------------------------------------------


def _reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> torch.Tensor:
    """Convert absolute leftover actions into model-space for relative-action RTC policies."""
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Abstracts sync vs. RTC (async) inference for rollout strategies.

    Parameters
    ----------
    policy:
        The loaded policy (already on device, in eval mode, with RTC
        processor initialised if applicable).
    preprocessor / postprocessor:
        Policy processor pipelines.
    robot_wrapper:
        Thread-safe robot wrapper.
    rtc_config:
        RTC configuration.  If ``rtc_config.enabled`` is False, the
        engine operates in synchronous mode.
    hw_features:
        Dataset-level feature dict built from ``hw_to_dataset_features``.
    action_keys:
        Ordered list of action feature names.
    task:
        Task description string.
    fps:
        Control loop frequency.
    device:
        Torch device string.
    use_torch_compile:
        Whether torch.compile warmup is needed.
    compile_warmup_inferences:
        Number of warmup inferences before live rollout.
    rtc_queue_threshold:
        Maximum RTC action queue size before the background thread
        pauses generation.  Prevents unbounded queue growth.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        rtc_config: RTCConfig,
        hw_features: dict,
        action_keys: list[str],
        task: str,
        fps: float,
        device: str | None,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        self._hw_features = hw_features
        self._action_keys = action_keys
        self._task = task
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold

        # RTC state
        self._use_rtc = rtc_config.enabled
        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._global_shutdown_event = shutdown_event
        self._rtc_thread: Thread | None = None

        if not self._use_torch_compile:
            self._compile_warmup_done.set()

        # Processor introspection for relative-action re-anchoring
        self._relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self._normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if self._relative_step is not None:
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                if cfg_names:
                    self._relative_step.action_names = list(cfg_names)
                else:
                    self._relative_step.action_names = [
                        k for k in robot_wrapper.action_features if k.endswith(".pos")
                    ]
            logger.info("Relative actions enabled: RTC prefix will be re-anchored")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_rtc(self) -> bool:
        return self._use_rtc

    @property
    def action_queue(self) -> ActionQueue | None:
        return self._action_queue

    @property
    def compile_warmup_done(self) -> Event:
        return self._compile_warmup_done

    @property
    def rtc_failed(self) -> bool:
        """True if the RTC background thread exited due to an unrecoverable error."""
        return self._rtc_error.is_set()

    def start(self) -> None:
        """Start the inference engine.  Launches the RTC background thread if enabled."""
        if self._use_rtc:
            self._action_queue = ActionQueue(self._rtc_config)
            self._obs_holder = {
                "obs": None,
                "robot_type": self._robot.robot_type,
            }
            self._shutdown_event.clear()
            self._rtc_thread = Thread(
                target=self._rtc_loop,
                daemon=True,
                name="RTCInference",
            )
            self._rtc_thread.start()
            logger.info("RTC inference thread started")

    def stop(self) -> None:
        """Signal the RTC thread to stop and wait for it."""
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._rtc_thread is not None and self._rtc_thread.is_alive():
            self._rtc_thread.join(timeout=3.0)
            self._rtc_thread = None

    def pause(self) -> None:
        """Pause the RTC background thread (used during DAgger takeover)."""
        self._policy_active.clear()

    def resume(self) -> None:
        """Resume the RTC background thread."""
        self._policy_active.set()

    def reset(self) -> None:
        """Reset policy, processors, and action queue between episodes."""
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._use_rtc and self._action_queue is not None:
            self._action_queue.clear()

    # ------------------------------------------------------------------
    # Sync inference
    # ------------------------------------------------------------------

    def get_action_sync(self, obs_frame: dict) -> torch.Tensor:
        """Run synchronous single-step inference.

        Parameters
        ----------
        obs_frame:
            Observation dict with numpy arrays (output of ``build_dataset_frame``).

        Returns
        -------
        Action tensor on CPU.
        """
        observation = copy(obs_frame)
        policy_device = torch.device(self._device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=policy_device.type)
            if policy_device.type == "cuda" and self._policy.config.use_amp
            else torch.inference_mode(),
        ):
            observation = prepare_observation_for_inference(
                observation, policy_device, self._task, self._robot.robot_type
            )
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        return action.squeeze(0).cpu()

    # ------------------------------------------------------------------
    # RTC: action consumption (called from main thread)
    # ------------------------------------------------------------------

    def consume_rtc_action(self) -> torch.Tensor | None:
        """Pop the next action from the RTC action queue.  Returns None if empty."""
        if self._action_queue is None:
            return None
        return self._action_queue.get()

    def update_observation(self, obs_filtered: dict) -> None:
        """Push the latest observation to the shared holder for the RTC thread."""
        with self._obs_lock:
            self._obs_holder["obs"] = obs_filtered

    # ------------------------------------------------------------------
    # RTC: background inference thread
    # ------------------------------------------------------------------

    def _rtc_loop(self) -> None:
        """Background thread that generates action chunks via RTC."""
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            policy_device = torch.device(self._device)

            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 0
            inference_count = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(0.01)
                    continue

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                if queue is None or obs is None:
                    time.sleep(0.01)
                    continue

                if queue.qsize() <= self._rtc_queue_threshold:
                    try:
                        current_time = time.perf_counter()
                        idx_before = queue.get_action_index()
                        prev_actions = queue.get_left_over()

                        latency = latency_tracker.max()
                        delay = math.ceil(latency / time_per_chunk) if latency else 0

                        # Build observation batch using the same pipeline as sync inference
                        obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                        obs_batch = prepare_observation_for_inference(
                            obs_batch, policy_device, self._task, self._robot.robot_type
                        )
                        # predict_action_chunk expects batched task format
                        obs_batch["task"] = [self._task]

                        preprocessed = self._preprocessor(obs_batch)

                        # Re-anchor leftover for relative-action policies
                        if prev_actions is not None and self._relative_step is not None:
                            state_tensor = preprocessed.get(OBS_STATE)
                            if state_tensor is not None:
                                prev_abs = queue.get_processed_left_over()
                                if prev_abs is not None and prev_abs.numel() > 0:
                                    prev_actions = _reanchor_relative_rtc_prefix(
                                        prev_actions_absolute=prev_abs,
                                        current_state=state_tensor,
                                        relative_step=self._relative_step,
                                        normalizer_step=self._normalizer_step,
                                        policy_device=policy_device,
                                    )

                        if prev_actions is not None:
                            prev_actions = _normalize_prev_actions_length(
                                prev_actions, target_steps=self._rtc_config.execution_horizon
                            )

                        actions = self._policy.predict_action_chunk(
                            preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
                        )

                        original = actions.squeeze(0).clone()
                        processed = self._postprocessor(actions).squeeze(0)
                        new_latency = time.perf_counter() - current_time
                        new_delay = math.ceil(new_latency / time_per_chunk)

                        inference_count += 1
                        is_warmup = self._use_torch_compile and inference_count <= warmup_required
                        if is_warmup:
                            latency_tracker.reset()
                        else:
                            latency_tracker.add(new_latency)

                        queue.merge(original, processed, new_delay, idx_before)

                        if (
                            is_warmup
                            and inference_count >= warmup_required
                            and not self._compile_warmup_done.is_set()
                        ):
                            self._compile_warmup_done.set()
                            logger.info("Compile warmup complete (%d inferences)", inference_count)

                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except Exception as e:
                        logger.error("RTC inference error: %s", e)
                        logger.debug(traceback.format_exc())
                        time.sleep(0.5)
                else:
                    time.sleep(0.01)

        except Exception as e:
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(traceback.format_exc())
            self._rtc_error.set()
            # Unblock any warmup waiters so the main loop doesn't spin forever
            self._compile_warmup_done.set()
            # Signal the top-level shutdown so strategies exit their control loops
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
