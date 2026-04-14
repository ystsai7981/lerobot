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
from threading import Event, Lock, Thread

from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..configs import SentryStrategyConfig
from ..context import RolloutContext
from . import RolloutStrategy, infer_action

logger = logging.getLogger(__name__)


class SentryStrategy(RolloutStrategy):
    """Continuous autonomous rollout with always-on recording.

    Episodes are auto-rotated every ``episode_duration_s`` seconds.
    The dataset is pushed to Hub in the background every
    ``upload_every_n_episodes`` episodes.

    Requires ``streaming_encoding=True`` (enforced in config validation)
    to prevent disk I/O from blocking the control loop.

    All actions flow through ``robot_observation_processor`` (observations)
    and ``robot_action_processor`` (actions) before reaching the robot,
    supporting EE-space recording with joint-space robots.

    **Thread safety:** A lock (``_episode_lock``) serialises
    ``save_episode`` and ``push_to_hub`` calls so the background push
    thread never reads an episode that is still being finalised.
    """

    config: SentryStrategyConfig

    def __init__(self, config: SentryStrategyConfig):
        super().__init__(config)
        self._push_thread: Thread | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)
        logger.info(
            "Sentry strategy ready (episode_duration=%.0fs, upload_every=%d eps)",
            self.config.episode_duration_s,
            self.config.upload_every_n_episodes,
        )

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.cfg
        robot = ctx.robot_wrapper
        dataset = ctx.dataset
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        ordered_keys = ctx.ordered_action_keys
        features = dataset.features

        if engine.is_rtc:
            engine.resume()

        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task

        with VideoEncodingManager(dataset):
            try:
                while not ctx.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        break

                    obs = robot.get_observation()
                    obs_processed = ctx.robot_observation_processor(obs)

                    if engine.is_rtc:
                        engine.update_observation(obs_processed)

                    if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                        continue

                    action_dict = infer_action(
                        engine, obs_processed, obs, ctx, interpolator, ordered_keys, features
                    )

                    # Record frame
                    if action_dict is not None:
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                        frame = {**obs_frame, **action_frame, "task": task_str}
                        dataset.add_frame(frame)

                    # Auto-rotate episodes
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
                        engine.reset()
                        interpolator.reset()
                        if engine.is_rtc:
                            engine.resume()

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)

            finally:
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()

    def teardown(self, ctx: RolloutContext) -> None:
        # Wait for any in-flight background push
        if self._push_thread is not None and self._push_thread.is_alive():
            self._push_thread.join(timeout=60)

        if ctx.dataset is not None:
            ctx.dataset.finalize()
            # Only push if there are unsaved changes since last background push
            if self._needs_push.is_set() and ctx.cfg.dataset and ctx.cfg.dataset.push_to_hub:
                ctx.dataset.push_to_hub(
                    tags=ctx.cfg.dataset.tags,
                    private=ctx.cfg.dataset.private,
                )

        self._teardown_hardware(ctx)
        logger.info("Sentry strategy teardown complete")

    def _background_push(self, dataset, cfg) -> None:
        """Push dataset to hub in a background thread (non-blocking).

        Acquires ``_episode_lock`` during the push to prevent
        ``save_episode`` from finalising a new episode mid-upload.
        """
        if self._push_thread is not None and self._push_thread.is_alive():
            logger.info("Previous push still in progress, skipping")
            return

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

        self._push_thread = Thread(target=_push, daemon=True)
        self._push_thread.start()
