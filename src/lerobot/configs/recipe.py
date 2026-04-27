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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MessageRole = Literal["user", "assistant", "system", "tool"]
MessageStream = Literal["high_level", "low_level"]

DEFAULT_BINDINGS = {
    "subtask": "active_at(t, style=subtask)",
    "memory": "active_at(t, style=memory)",
    "plan": "active_at(t, style=plan)",
    "speech": "emitted_at(t, role=assistant, tool_name=say)",
    "interjection": "emitted_at(t, style=interjection)",
    "vqa": "emitted_at(t, style=vqa, role=assistant)",
    "vqa_query": "emitted_at(t, style=vqa, role=user)",
}

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VALID_ROLES = {"user", "assistant", "system", "tool"}
_VALID_STREAMS = {"high_level", "low_level"}


@dataclass
class MessageTurn:
    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    stream: MessageStream | None = None
    target: bool = False
    if_present: str | None = None
    tool_calls_from: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(f"Unsupported message role: {self.role!r}")
        if self.stream is not None and self.stream not in _VALID_STREAMS:
            raise ValueError(f"Unsupported message stream: {self.stream!r}")
        if self.content is None and self.tool_calls_from is None:
            raise ValueError("MessageTurn.content is required unless tool_calls_from is set.")
        if self.content is not None and not isinstance(self.content, (str, list)):
            raise TypeError("MessageTurn.content must be a string, a list of HF-style blocks, or None.")
        if isinstance(self.content, list):
            for block in self.content:
                if not isinstance(block, dict) or "type" not in block:
                    raise ValueError(
                        "Multimodal content blocks must be HF-style dictionaries with a type key."
                    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageTurn:
        return cls(**data)


@dataclass
class TrainingRecipe:
    messages: list[MessageTurn] | None = None
    bindings: dict[str, str] | None = None
    blend: dict[str, TrainingRecipe] | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        if self.messages is not None and self.blend is not None:
            raise ValueError("TrainingRecipe must set only one of messages or blend.")
        if self.messages is None and self.blend is None:
            raise ValueError("TrainingRecipe must set one of messages or blend.")

        if self.messages is not None:
            self._validate_message_recipe()
        if self.blend is not None:
            self._validate_blend_recipe()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRecipe:
        data = dict(data)
        if data.get("messages") is not None:
            data["messages"] = [
                turn if isinstance(turn, MessageTurn) else MessageTurn.from_dict(turn)
                for turn in data["messages"]
            ]
        if data.get("blend") is not None:
            data["blend"] = {
                name: recipe if isinstance(recipe, TrainingRecipe) else cls.from_dict(recipe)
                for name, recipe in data["blend"].items()
            }
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingRecipe:
        import yaml  # type: ignore[import-untyped]

        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Recipe YAML must contain a mapping at the top level: {path}")
        return cls.from_dict(data)

    def _validate_message_recipe(self) -> None:
        assert self.messages is not None
        known_bindings = set(DEFAULT_BINDINGS) | set(self.bindings or {}) | {"task"}

        for turn in self.messages:
            missing = self._referenced_bindings(turn) - known_bindings
            if missing:
                raise ValueError(f"MessageTurn references unknown binding(s): {sorted(missing)}")

        if not any(turn.target for turn in self.messages):
            raise ValueError("Message recipes must contain at least one target turn.")

    def _validate_blend_recipe(self) -> None:
        assert self.blend is not None
        if not self.blend:
            raise ValueError("Blend recipes must contain at least one component.")

        for name, recipe in self.blend.items():
            if recipe.blend is not None:
                raise ValueError(f"Blend component {name!r} cannot itself define a blend.")
            if recipe.messages is None:
                raise ValueError(f"Blend component {name!r} must define messages.")
            if recipe.weight is None:
                raise ValueError(f"Blend component {name!r} must define weight.")
            if recipe.weight <= 0:
                raise ValueError(f"Blend component {name!r} must have a positive weight.")

    def _referenced_bindings(self, turn: MessageTurn) -> set[str]:
        names: set[str] = set()
        if turn.if_present is not None:
            names.add(turn.if_present)
        if turn.tool_calls_from is not None:
            names.add(turn.tool_calls_from)
        names.update(_placeholders_in_content(turn.content))
        return names


def _placeholders_in_content(content: str | list[dict[str, Any]] | None) -> set[str]:
    if content is None:
        return set()
    if isinstance(content, str):
        return set(_PLACEHOLDER_RE.findall(content))

    names: set[str] = set()
    for block in content:
        for value in block.values():
            if isinstance(value, str):
                names.update(_PLACEHOLDER_RE.findall(value))
    return names


def load_recipe(path: str | Path) -> TrainingRecipe:
    return TrainingRecipe.from_yaml(path)
