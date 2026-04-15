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

"""Inference strategy package — backend-agnostic action production.

Concrete strategies (sync, RTC, …) expose the same small interface so
rollout strategies never branch on the inference backend.
"""

from .base import InferenceStrategy
from .factory import (
    InferenceStrategyConfig,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_strategy,
)
from .rtc import RTCInferenceStrategy
from .sync import SyncInferenceStrategy

__all__ = [
    "InferenceStrategy",
    "InferenceStrategyConfig",
    "RTCInferenceConfig",
    "RTCInferenceStrategy",
    "SyncInferenceConfig",
    "SyncInferenceStrategy",
    "create_inference_strategy",
]
