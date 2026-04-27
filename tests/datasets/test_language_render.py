#!/usr/bin/env python

from pathlib import Path

import pytest

from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.datasets.language_render import active_at, emitted_at, nth_next, nth_prev, render_sample


def row(role, content, style, timestamp, tool_calls=None):
    return {
        "role": role,
        "content": content,
        "style": style,
        "timestamp": timestamp,
        "tool_calls": tool_calls,
    }


PERSISTENT = [
    row("assistant", "plan 0", "plan", 0.0),
    row("assistant", "memory 0", "memory", 0.0),
    row("assistant", "subtask 0", "subtask", 0.0),
    row("assistant", "memory 1", "memory", 1.0),
    row("assistant", "subtask 1", "subtask", 1.0),
]
EVENTS = [
    row("user", "what is visible?", "vqa", 1.0),
    row("assistant", '{"count": 2}', "vqa", 1.0),
    row("user", "skip wiping", "interjection", 2.0),
    row(
        "assistant",
        None,
        None,
        2.0,
        [{"type": "function", "function": {"name": "say", "arguments": {"text": "Skipping wiping."}}}],
    ),
]


def test_resolver_temporal_semantics():
    assert active_at(0.5, persistent=PERSISTENT, style="subtask")["content"] == "subtask 0"
    assert active_at(1.0, persistent=PERSISTENT, style="subtask")["content"] == "subtask 1"
    assert emitted_at(0.5, persistent=PERSISTENT, events=EVENTS, style="vqa", role="assistant") is None
    assert (
        emitted_at(1.0, persistent=PERSISTENT, events=EVENTS, style="vqa", role="assistant")["content"]
        == '{"count": 2}'
    )


def test_persistent_relative_resolvers_reject_event_styles():
    with pytest.raises(ValueError, match="event-only"):
        active_at(1.0, persistent=PERSISTENT, style="vqa")
    with pytest.raises(ValueError, match="event-only"):
        nth_prev(1.0, persistent=PERSISTENT, style="interjection")


def test_nth_prev_and_next():
    assert nth_prev(1.0, persistent=PERSISTENT, style="subtask", offset=1)["content"] == "subtask 0"
    assert nth_next(0.0, persistent=PERSISTENT, style="subtask", offset=1)["content"] == "subtask 1"


def test_substitution_if_present_multimodal_and_tool_calls():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(
                role="user",
                content=[
                    {"type": "image", "feature": "observation.images.top"},
                    {"type": "text", "text": "${task}: ${interjection}"},
                ],
                stream="high_level",
                if_present="interjection",
            ),
            MessageTurn(
                role="assistant",
                content="${plan}",
                stream="high_level",
                target=True,
                tool_calls_from="speech",
            ),
        ],
        bindings={"plan": "active_at(t, style=plan)"},
    )

    rendered = render_sample(
        recipe=recipe,
        persistent=PERSISTENT,
        events=EVENTS,
        t=2.0,
        sample_idx=0,
        task="clean kitchen",
    )

    assert rendered["messages"][0]["content"][1]["text"] == "clean kitchen: skip wiping"
    assert rendered["messages"][1]["content"] == "plan 0"
    assert rendered["messages"][1]["tool_calls"][0]["function"]["name"] == "say"
    assert rendered["message_streams"] == ["high_level", "high_level"]
    assert rendered["target_message_indices"] == [1]


def test_exact_event_miss_returns_none_when_target_skips():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${vqa_query}", stream="high_level", if_present="vqa_query"),
            MessageTurn(
                role="assistant",
                content="${vqa}",
                stream="high_level",
                target=True,
                if_present="vqa",
            ),
        ]
    )

    assert render_sample(recipe=recipe, persistent=PERSISTENT, events=EVENTS, t=0.0, sample_idx=0) is None


def test_deterministic_blend_sampling():
    recipe = TrainingRecipe(
        blend={
            "a": TrainingRecipe(
                weight=1.0,
                messages=[
                    MessageTurn(role="user", content="${task}", stream="high_level"),
                    MessageTurn(role="assistant", content="a", stream="high_level", target=True),
                ],
            ),
            "b": TrainingRecipe(
                weight=1.0,
                messages=[
                    MessageTurn(role="user", content="${task}", stream="high_level"),
                    MessageTurn(role="assistant", content="b", stream="high_level", target=True),
                ],
            ),
        }
    )

    first = render_sample(
        recipe=recipe, persistent=PERSISTENT, events=EVENTS, t=0.0, sample_idx=123, task="x"
    )
    second = render_sample(
        recipe=recipe, persistent=PERSISTENT, events=EVENTS, t=0.0, sample_idx=123, task="x"
    )
    assert first == second


def test_canonical_recipe_can_render_low_level_branch():
    recipe = TrainingRecipe.from_yaml(Path("src/lerobot/configs/recipes/pi05_hirobot.yaml"))
    low_level = TrainingRecipe(blend={"low": recipe.blend["low_level_execution"]})

    rendered = render_sample(
        recipe=low_level,
        persistent=PERSISTENT,
        events=[],
        t=0.5,
        sample_idx=0,
        task="clean kitchen",
    )

    assert rendered["messages"][-1] == {"role": "assistant", "content": "subtask 0"}
    assert rendered["message_streams"][-1] == "low_level"
    assert rendered["target_message_indices"] == [1]
