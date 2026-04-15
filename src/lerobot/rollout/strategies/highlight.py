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

"""Highlight Reel strategy: on-demand recording via ring buffer."""

from __future__ import annotations

import contextlib
import logging
import time
from threading import Event as ThreadingEvent

from lerobot.common.control_utils import is_headless
from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..configs import HighlightStrategyConfig
from ..context import RolloutContext
from ..ring_buffer import RolloutRingBuffer
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class HighlightStrategy(RolloutStrategy):
    """Autonomous rollout with on-demand recording via ring buffer.

    The robot runs autonomously while a memory-bounded ring buffer
    captures continuous telemetry.  When the user presses the save key:

    1. The ring buffer is flushed to the dataset (last *Z* seconds).
    2. Live recording continues until the save key is pressed again.
    3. The episode is saved and the ring buffer resumes capturing.

    Requires ``streaming_encoding=True`` (enforced in config validation)
    so that ``dataset.add_frame`` is a non-blocking queue put — draining
    900 frames stays sub-ms per frame.
    """

    config: HighlightStrategyConfig

    def __init__(self, config: HighlightStrategyConfig):
        super().__init__(config)
        self._ring: RolloutRingBuffer | None = None
        self._listener = None
        self._save_requested = ThreadingEvent()
        self._recording_live = ThreadingEvent()
        self._shutdown_event: ThreadingEvent | None = None

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)

        self._ring = RolloutRingBuffer(
            max_seconds=self.config.ring_buffer_seconds,
            max_memory_mb=self.config.ring_buffer_max_memory_mb,
            fps=ctx.runtime.cfg.fps,
        )

        self._shutdown_event = ctx.runtime.shutdown_event
        self._setup_keyboard()
        logger.info(
            "Highlight strategy ready (buffer=%.0fs, key='%s')",
            self.config.ring_buffer_seconds,
            self.config.save_key,
        )

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        ring = self._ring
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        ordered_keys = ctx.data.ordered_action_keys
        features = dataset.features

        engine.resume()

        start_time = time.perf_counter()
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

                        # NOTE: ``is_set()`` then ``clear()`` is not atomic
                        # against the keyboard thread setting the flag again
                        # in between — but that is benign: we lose at most one
                        # toggle, processed on the next iteration.  The
                        # ``_recording_live`` branch below is reached in the
                        # SAME iteration after ``clear()`` runs, so a frame
                        # finalised by ``save_episode()`` is never re-added to
                        # the next episode.
                        if self._save_requested.is_set():
                            self._save_requested.clear()
                            if not self._recording_live.is_set():
                                logger.info(
                                    "Flushing ring buffer (%d frames) + starting live recording",
                                    len(ring),
                                )
                                for buffered_frame in ring.drain():
                                    dataset.add_frame(buffered_frame)
                                self._recording_live.set()
                            else:
                                dataset.add_frame(frame)
                                dataset.save_episode()
                                logger.info("Episode saved")
                                self._recording_live.clear()
                                engine.reset()
                                interpolator.reset()
                                engine.resume()

                        if self._recording_live.is_set():
                            dataset.add_frame(frame)
                        else:
                            ring.append(frame)

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)

            finally:
                if self._recording_live.is_set():
                    with contextlib.suppress(Exception):
                        dataset.save_episode()

    def teardown(self, ctx: RolloutContext) -> None:
        if self._listener is not None:
            self._listener.stop()

        if ctx.data.dataset is not None:
            ctx.data.dataset.finalize()
            if ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                ctx.data.dataset.push_to_hub(
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                )

        self._teardown_hardware(ctx)
        logger.info("Highlight strategy teardown complete")

    def _setup_keyboard(self) -> None:
        """Set up keyboard listener for the save key."""
        if is_headless():
            logger.warning("Headless environment — highlight save key unavailable")
            return

        try:
            from pynput import keyboard

            save_key = self.config.save_key

            def on_press(key):
                with contextlib.suppress(Exception):
                    if hasattr(key, "char") and key.char == save_key:
                        self._save_requested.set()
                    elif key == keyboard.Key.esc:
                        self._save_requested.clear()
                        if self._shutdown_event is not None:
                            self._shutdown_event.set()

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
        except ImportError:
            logger.warning("pynput not available — keyboard listener disabled")
