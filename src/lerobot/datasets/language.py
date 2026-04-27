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
PERSISTENT_ROW_FIELDS = ("role", "content", "style", "timestamp", "tool_calls")
EVENT_ROW_FIELDS = ("role", "content", "style", "tool_calls")

CORE_STYLES = {"subtask", "plan", "memory", "interjection", "vqa"}
EXTENDED_STYLES = set()
STYLE_REGISTRY = CORE_STYLES | EXTENDED_STYLES

PERSISTENT_STYLES = {"subtask", "plan", "memory"}
EVENT_ONLY_STYLES = {"interjection", "vqa"}

LanguageColumn = Literal["language_persistent", "language_events"]


def _json_arrow_type() -> pa.DataType:
    """Return the Arrow JSON type, falling back to ``string`` on older pyarrow."""
    return pa.json_() if hasattr(pa, "json_") else pa.string()


def _json_feature() -> object:
    """Return the HF ``datasets`` JSON feature, falling back to a string value."""
    return datasets.Json() if hasattr(datasets, "Json") else datasets.Value("string")


def language_persistent_row_arrow_type() -> pa.StructType:
    """Return the Arrow struct type for a single persistent language row.

    Persistent rows carry their own ``timestamp`` because they represent a state
    that became active at a specific moment and remains active until superseded.
    """
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("timestamp", pa.float64(), nullable=False),
            pa.field("tool_calls", pa.list_(_json_arrow_type()), nullable=True),
        ]
    )


def language_event_row_arrow_type() -> pa.StructType:
    """Return the Arrow struct type for a single event language row.

    Event rows have no ``timestamp`` field: each event is stored on the dataset
    row whose frame timestamp is the event's firing time.
    """
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_json_arrow_type()), nullable=True),
        ]
    )


def language_persistent_arrow_type() -> pa.ListType:
    """Return the Arrow list type for the ``language_persistent`` column."""
    return pa.list_(language_persistent_row_arrow_type())


def language_events_arrow_type() -> pa.ListType:
    """Return the Arrow list type for the ``language_events`` column."""
    return pa.list_(language_event_row_arrow_type())


def language_persistent_row_feature() -> dict[str, object]:
    """Return the HF ``datasets`` feature mapping for a persistent language row."""
    return {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
        "style": datasets.Value("string"),
        "timestamp": datasets.Value("float64"),
        "tool_calls": datasets.List(_json_feature()),
    }


def language_event_row_feature() -> dict[str, object]:
    """Return the HF ``datasets`` feature mapping for an event language row."""
    return {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
        "style": datasets.Value("string"),
        "tool_calls": datasets.List(_json_feature()),
    }


def language_persistent_column_feature() -> datasets.List:
    """Return the HF ``datasets`` feature for the ``language_persistent`` column."""
    return datasets.List(language_persistent_row_feature())


def language_events_column_feature() -> datasets.List:
    """Return the HF ``datasets`` feature for the ``language_events`` column."""
    return datasets.List(language_event_row_feature())


def language_feature_info() -> dict[str, dict]:
    """Return the ``info["features"]`` entries for both language columns."""
    return {
        LANGUAGE_PERSISTENT: {"dtype": "language", "shape": (1,), "names": None},
        LANGUAGE_EVENTS: {"dtype": "language", "shape": (1,), "names": None},
    }


def is_language_column(key: str) -> bool:
    """Return ``True`` if ``key`` is one of the dataset's language column names."""
    return key in LANGUAGE_COLUMNS


def column_for_style(style: str | None) -> LanguageColumn:
    """Map a language style to the column where rows of that style are stored.

    Styles in :data:`PERSISTENT_STYLES` route to :data:`LANGUAGE_PERSISTENT`.
    Styles in :data:`EVENT_ONLY_STYLES` and the implicit ``None`` style route
    to :data:`LANGUAGE_EVENTS`.
    """
    if style is None:
        return LANGUAGE_EVENTS
    if style in PERSISTENT_STYLES:
        return LANGUAGE_PERSISTENT
    if style in EVENT_ONLY_STYLES:
        return LANGUAGE_EVENTS
    raise ValueError(f"Unknown language style: {style!r}")
