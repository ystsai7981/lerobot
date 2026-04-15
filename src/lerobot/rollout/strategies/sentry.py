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

"""Sentry rollout strategy: continuous autonomous recording with auto-upload."""

from __future__ import annotations

import contextlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock

from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..configs import SentryStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class SentryStrategy(RolloutStrategy):
    """Continuous autonomous rollout with always-on recording.

    Episodes are auto-rotated every ``episode_duration_s`` seconds.  The
    dataset is pushed to the Hub via a bounded single-worker executor so
    no push is ever silently dropped and exactly one push runs at a time.

    Policy state (hidden state, RTC queue) intentionally persists across
    episode boundaries — Sentry slices one continuous rollout, the robot
    does not reset between slices.

    Requires ``streaming_encoding=True`` (enforced in config validation)
    to prevent disk I/O from blocking the control loop.
    """

    config: SentryStrategyConfig

    def __init__(self, config: SentryStrategyConfig):
        super().__init__(config)
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentry-push")
        logger.info(
            "Sentry strategy ready (episode_duration=%.0fs, upload_every=%d eps)",
            self.config.episode_duration_s,
            self.config.upload_every_n_episodes,
        )

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        ordered_keys = ctx.data.ordered_action_keys
        features = dataset.features

        engine.resume()

        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task

        with VideoEncodingManager(dataset):
            try:
                while not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        break

                    obs = robot.get_observation()
                    obs_processed = ctx.processors.robot_observation_processor(obs)
                    engine.notify_observation(obs_processed)

                    if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                        continue

                    action_dict = send_next_action(
                        engine, obs_processed, obs, ctx, interpolator, ordered_keys, features
                    )

                    if action_dict is not None:
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                        frame = {**obs_frame, **action_frame, "task": task_str}
                        dataset.add_frame(frame)

                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= self.config.episode_duration_s:
                        with self._episode_lock:
                            dataset.save_episode()
                        episodes_since_push += 1
                        self._needs_push.set()
                        logger.info("Episode saved (total: %d)", dataset.num_episodes)

                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0

                        episode_start = time.perf_counter()

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)

            finally:
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()

    def teardown(self, ctx: RolloutContext) -> None:
        # Flush any queued/running push cleanly.
        if self._push_executor is not None:
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            ctx.data.dataset.finalize()
            if self._needs_push.is_set() and ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                ctx.data.dataset.push_to_hub(
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                )

        self._teardown_hardware(ctx)
        logger.info("Sentry strategy teardown complete")

    def _background_push(self, dataset, cfg) -> None:
        """Queue a Hub push on the single-worker executor.

        The executor's max_workers=1 guarantees at most one push runs at
        a time; submitted tasks are queued rather than dropped.
        """
        if self._push_executor is None:
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def _push():
            try:
                with self._episode_lock:
                    dataset.push_to_hub(
                        tags=cfg.dataset.tags if cfg.dataset else None,
                        private=cfg.dataset.private if cfg.dataset else False,
                    )
                self._needs_push.clear()
                logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
