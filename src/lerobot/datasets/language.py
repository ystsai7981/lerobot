#!/usr/bin/env python

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

from __future__ import annotations

from typing import Literal

import datasets
import pyarrow as pa

LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"
LANGUAGE_COLUMNS = (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS)
LANGUAGE_ROW_FIELDS = ("role", "content", "style", "timestamp", "tool_calls")

CORE_STYLES = {"subtask", "plan", "memory", "interjection", "vqa"}
EXTENDED_STYLES = set()
RESERVED_STYLES = {"motion", "trace"}
STYLE_REGISTRY = CORE_STYLES | EXTENDED_STYLES | RESERVED_STYLES

PERSISTENT_STYLES = {"subtask", "plan", "memory"}
EVENT_ONLY_STYLES = {"interjection", "vqa"}

LanguageColumn = Literal["language_persistent", "language_events"]


def language_row_arrow_type() -> pa.StructType:
    json_type = pa.json_() if hasattr(pa, "json_") else pa.string()
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("timestamp", pa.float64(), nullable=False),
            pa.field("tool_calls", pa.list_(json_type), nullable=True),
        ]
    )


def language_persistent_arrow_type() -> pa.ListType:
    return pa.list_(language_row_arrow_type())


def language_events_arrow_type() -> pa.ListType:
    return pa.list_(language_row_arrow_type())


def language_row_feature() -> dict[str, object]:
    json_feature = datasets.Json() if hasattr(datasets, "Json") else datasets.Value("string")
    return {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
        "style": datasets.Value("string"),
        "timestamp": datasets.Value("float64"),
        "tool_calls": datasets.List(json_feature),
    }


def language_column_feature() -> datasets.List:
    return datasets.List(language_row_feature())


def language_feature_info() -> dict[str, dict]:
    return {
        LANGUAGE_PERSISTENT: {"dtype": "language", "shape": (1,), "names": None},
        LANGUAGE_EVENTS: {"dtype": "language", "shape": (1,), "names": None},
    }


def is_language_column(key: str) -> bool:
    return key in LANGUAGE_COLUMNS


def column_for_style(style: str | None) -> LanguageColumn:
    if style is None:
        return LANGUAGE_EVENTS
    if style in PERSISTENT_STYLES:
        return LANGUAGE_PERSISTENT
    if style in EVENT_ONLY_STYLES:
        return LANGUAGE_EVENTS
    if style in RESERVED_STYLES:
        raise ValueError(f"Style {style!r} is registered but has no storage column yet.")
    raise ValueError(f"Unknown language style: {style!r}")
