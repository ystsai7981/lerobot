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

import copy
import hashlib
import re
from collections.abc import Sequence
from typing import Any

from lerobot.configs.recipe import DEFAULT_BINDINGS, TrainingRecipe

from .language import (
    EVENT_ONLY_STYLES,
    LANGUAGE_PERSISTENT,
    PERSISTENT_STYLES,
    column_for_style,
)

LanguageRow = dict[str, Any]
RenderedMessages = dict[str, list[Any]]

_RESOLVER_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def active_at(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow] | None = None,
    style: str | None = None,
    role: str | None = None,
    tool_name: str | None = None,
) -> LanguageRow | None:
    _validate_persistent_resolver("active_at", style)
    matches = _matching_rows(persistent, style=style, role=role, tool_name=tool_name)
    matches = [row for row in matches if _timestamp(row) <= t]
    return _select_latest(matches, style=style, role=role, tool_name=tool_name)


def emitted_at(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    style: str | None = None,
    role: str | None = None,
    tool_name: str | None = None,
) -> LanguageRow | None:
    column = column_for_style(style)
    rows = persistent if column == LANGUAGE_PERSISTENT else events
    matches = [
        row
        for row in _matching_rows(rows, style=style, role=role, tool_name=tool_name)
        if _timestamp(row) == t
    ]
    return _select_exact(matches, style=style, role=role, tool_name=tool_name)


def nth_prev(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow] | None = None,
    style: str | None = None,
    offset: int = 1,
    role: str | None = None,
    tool_name: str | None = None,
) -> LanguageRow | None:
    return _nth_relative(
        t,
        persistent=persistent,
        style=style,
        offset=-offset,
        role=role,
        tool_name=tool_name,
        resolver_name="nth_prev",
    )


def nth_next(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow] | None = None,
    style: str | None = None,
    offset: int = 1,
    role: str | None = None,
    tool_name: str | None = None,
) -> LanguageRow | None:
    return _nth_relative(
        t,
        persistent=persistent,
        style=style,
        offset=offset,
        role=role,
        tool_name=tool_name,
        resolver_name="nth_next",
    )


def render_sample(
    *,
    recipe: TrainingRecipe,
    persistent: Sequence[LanguageRow] | None,
    events: Sequence[LanguageRow] | None,
    t: float,
    sample_idx: int,
    task: str | None = None,
    dataset_ctx: Any | None = None,
) -> RenderedMessages | None:
    persistent_rows = _normalize_rows(persistent or [])
    event_rows = _normalize_rows(events or [])
    selected_recipe = _select_recipe(recipe, sample_idx)
    bindings = _resolve_bindings(
        selected_recipe,
        persistent=persistent_rows,
        events=event_rows,
        t=t,
        task=task,
        dataset_ctx=dataset_ctx,
    )
    return _render_message_recipe(selected_recipe, bindings)


def _select_recipe(recipe: TrainingRecipe, sample_idx: int) -> TrainingRecipe:
    if recipe.blend is None:
        return recipe

    total_weight = sum(component.weight or 0.0 for component in recipe.blend.values())
    if total_weight <= 0:
        raise ValueError("Blend weights must sum to a positive value.")

    digest = hashlib.blake2b(str(sample_idx).encode(), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / 2**64 * total_weight
    cumulative = 0.0
    last_component: TrainingRecipe | None = None
    for component in recipe.blend.values():
        last_component = component
        cumulative += component.weight or 0.0
        if draw < cumulative:
            return component
    assert last_component is not None
    return last_component


def _resolve_bindings(
    recipe: TrainingRecipe,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    t: float,
    task: str | None,
    dataset_ctx: Any | None,
) -> dict[str, LanguageRow | str | None]:
    bindings: dict[str, LanguageRow | str | None] = {"task": _resolve_task(task, dataset_ctx)}
    specs = {**DEFAULT_BINDINGS, **(recipe.bindings or {})}
    for name, spec in specs.items():
        bindings[name] = _resolve_spec(spec, persistent=persistent, events=events, t=t)
    return bindings


def _resolve_task(task: str | None, dataset_ctx: Any | None) -> str | None:
    if task is not None:
        return task
    if dataset_ctx is None:
        return None
    if isinstance(dataset_ctx, dict):
        return dataset_ctx.get("task")
    return getattr(dataset_ctx, "task", None)


def _resolve_spec(
    spec: str,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    t: float,
) -> LanguageRow | None:
    match = _RESOLVER_RE.match(spec.strip())
    if match is None:
        raise ValueError(f"Invalid resolver expression: {spec!r}")
    name = match.group("name")
    kwargs = _parse_resolver_args(match.group("args"))
    kwargs.pop("t_arg", None)

    resolvers = {
        "active_at": active_at,
        "emitted_at": emitted_at,
        "nth_prev": nth_prev,
        "nth_next": nth_next,
    }
    if name not in resolvers:
        raise ValueError(f"Unknown language resolver: {name!r}")
    return resolvers[name](t, persistent=persistent, events=events, **kwargs)


def _parse_resolver_args(args: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if not args.strip():
        return kwargs

    parts = [part.strip() for part in args.split(",") if part.strip()]
    for part in parts:
        if part == "t":
            kwargs["t_arg"] = True
            continue
        if "=" not in part:
            raise ValueError(f"Invalid resolver argument: {part!r}")
        key, value = (item.strip() for item in part.split("=", 1))
        if key == "offset":
            kwargs[key] = int(value)
        else:
            kwargs[key] = value.strip("\"'")
    return kwargs


def _render_message_recipe(
    recipe: TrainingRecipe,
    bindings: dict[str, LanguageRow | str | None],
) -> RenderedMessages | None:
    assert recipe.messages is not None
    messages: list[dict[str, Any]] = []
    streams: list[str | None] = []
    target_indices: list[int] = []

    for turn in recipe.messages:
        if turn.if_present is not None and bindings.get(turn.if_present) is None:
            continue

        message = {"role": turn.role}
        if turn.content is not None:
            message["content"] = _render_content(turn.content, bindings)

        if turn.tool_calls_from is not None:
            row = bindings.get(turn.tool_calls_from)
            tool_calls = row.get("tool_calls") if isinstance(row, dict) else None
            if tool_calls:
                message["tool_calls"] = copy.deepcopy(tool_calls)

        message_idx = len(messages)
        messages.append(message)
        streams.append(turn.stream)
        if turn.target:
            target_indices.append(message_idx)

    if not target_indices:
        return None

    rendered = {
        "messages": messages,
        "message_streams": streams,
        "target_message_indices": target_indices,
    }
    _validate_rendered(rendered)
    return rendered


def _render_content(
    content: str | list[dict[str, Any]],
    bindings: dict[str, LanguageRow | str | None],
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return _substitute(content, bindings)

    rendered_blocks = []
    for block in content:
        rendered_block = copy.deepcopy(block)
        for key, value in rendered_block.items():
            if isinstance(value, str):
                rendered_block[key] = _substitute(value, bindings)
        rendered_blocks.append(rendered_block)
    return rendered_blocks


def _substitute(template: str, bindings: dict[str, LanguageRow | str | None]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in bindings:
            raise ValueError(f"Unknown template binding: {name!r}")
        value = bindings[name]
        if value is None:
            return ""
        if isinstance(value, dict):
            content = value.get("content")
            return "" if content is None else str(content)
        return str(value)

    return _PLACEHOLDER_RE.sub(replace, template)


def _validate_rendered(rendered: RenderedMessages) -> None:
    messages = rendered["messages"]
    streams = rendered["message_streams"]
    target_indices = rendered["target_message_indices"]

    if len(streams) != len(messages):
        raise ValueError("message_streams must be aligned with messages.")
    if not target_indices:
        raise ValueError("Rendered samples must contain at least one target message.")
    for idx in target_indices:
        if idx < 0 or idx >= len(messages):
            raise ValueError(f"Target message index {idx} is out of bounds.")
    for idx, stream in enumerate(streams):
        if stream is None:
            raise ValueError(f"Rendered message {idx} has no stream.")


def _nth_relative(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    style: str | None,
    offset: int,
    role: str | None,
    tool_name: str | None,
    resolver_name: str,
) -> LanguageRow | None:
    _validate_persistent_resolver(resolver_name, style)
    if abs(offset) < 1:
        raise ValueError(f"{resolver_name} offset must be non-zero.")

    rows = _sort_rows(_matching_rows(persistent, style=style, role=role, tool_name=tool_name))
    if not rows:
        return None

    anchor_idx = None
    for idx, row in enumerate(rows):
        if _timestamp(row) <= t:
            anchor_idx = idx
        else:
            break

    target_idx = (offset - 1 if offset > 0 else None) if anchor_idx is None else anchor_idx + offset

    if target_idx is None or target_idx < 0 or target_idx >= len(rows):
        return None
    return rows[target_idx]


def _validate_persistent_resolver(resolver_name: str, style: str | None) -> None:
    if style is None:
        raise ValueError(f"{resolver_name} requires a persistent style.")
    if style in EVENT_ONLY_STYLES:
        raise ValueError(f"{resolver_name} cannot be used with event-only style {style!r}.")
    if style not in PERSISTENT_STYLES:
        column_for_style(style)


def _matching_rows(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
) -> list[LanguageRow]:
    return [
        row
        for row in rows
        if (style is None or row.get("style") == style)
        and (role is None or row.get("role") == role)
        and (tool_name is None or _row_has_tool_name(row, tool_name))
    ]


def _select_latest(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
) -> LanguageRow | None:
    if not rows:
        return None
    rows = _sort_rows(rows)
    latest_ts = _timestamp(rows[-1])
    return _select_exact(
        [row for row in rows if _timestamp(row) == latest_ts],
        style=style,
        role=role,
        tool_name=tool_name,
    )


def _select_exact(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
) -> LanguageRow | None:
    if not rows:
        return None
    if len(rows) > 1 and role is None and tool_name is None:
        raise ValueError(
            f"Ambiguous resolver for style={style!r}; add role=... or tool_name=... to disambiguate."
        )
    return _sort_rows(rows)[0]


def _sort_rows(rows: Sequence[LanguageRow]) -> list[LanguageRow]:
    return sorted(rows, key=lambda row: (_timestamp(row), row.get("style") or "", row.get("role") or ""))


def _timestamp(row: LanguageRow) -> float:
    value = row["timestamp"]
    return float(value.item() if hasattr(value, "item") else value)


def _row_has_tool_name(row: LanguageRow, tool_name: str) -> bool:
    for tool_call in row.get("tool_calls") or []:
        if isinstance(tool_call, str):
            continue
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if isinstance(function, dict) and function.get("name") == tool_name:
            return True
    return False


def _normalize_rows(rows: Sequence[Any]) -> list[LanguageRow]:
    normalized = []
    for row in rows:
        if row is None:
            continue
        if hasattr(row, "as_py"):
            row = row.as_py()
        if not isinstance(row, dict):
            raise TypeError(f"Language rows must be dictionaries, got {type(row).__name__}.")
        normalized.append(dict(row))
    return normalized
