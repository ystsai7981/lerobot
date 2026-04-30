#!/usr/bin/env python

from pathlib import Path

import pytest

from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.datasets.language_render import active_at, emitted_at, nth_next, nth_prev, render_sample


def persistent_row(role, content, style, timestamp, tool_calls=None, camera=None):
    return {
        "role": role,
        "content": content,
        "style": style,
        "timestamp": timestamp,
        "camera": camera,
        "tool_calls": tool_calls,
    }


def event_row(role, content, style, tool_calls=None, camera=None):
    return {
        "role": role,
        "content": content,
        "style": style,
        "camera": camera,
        "tool_calls": tool_calls,
    }


PERSISTENT = [
    persistent_row("assistant", "plan 0", "plan", 0.0),
    persistent_row("assistant", "memory 0", "memory", 0.0),
    persistent_row("assistant", "subtask 0", "subtask", 0.0),
    persistent_row("assistant", "memory 1", "memory", 1.0),
    persistent_row("assistant", "subtask 1", "subtask", 1.0),
]
EVENTS_AT_1 = [
    event_row("user", "what is visible?", "vqa", camera="observation.images.top"),
    event_row("assistant", '{"count": 2}', "vqa", camera="observation.images.top"),
]
EVENTS_AT_2 = [
    event_row("user", "skip wiping", "interjection"),
    event_row(
        "assistant",
        None,
        None,
        [{"type": "function", "function": {"name": "say", "arguments": {"text": "Skipping wiping."}}}],
    ),
]
# Same emission tick, two cameras: triggers per-camera disambiguation in
# resolvers, mirroring how Module 3 of the annotation pipeline writes one
# (vqa, user) + (vqa, assistant) pair per camera.
EVENTS_AT_3_TWO_CAMERAS = [
    event_row("user", "how many cups (top)?", "vqa", camera="observation.images.top"),
    event_row("assistant", '{"count": 3}', "vqa", camera="observation.images.top"),
    event_row("user", "how many cups (wrist)?", "vqa", camera="observation.images.wrist"),
    event_row("assistant", '{"count": 1}', "vqa", camera="observation.images.wrist"),
]


def test_resolver_temporal_semantics():
    assert active_at(0.5, persistent=PERSISTENT, style="subtask")["content"] == "subtask 0"
    assert active_at(1.0, persistent=PERSISTENT, style="subtask")["content"] == "subtask 1"
    assert emitted_at(0.5, persistent=PERSISTENT, events=[], style="vqa", role="assistant") is None
    assert (
        emitted_at(1.0, persistent=PERSISTENT, events=EVENTS_AT_1, style="vqa", role="assistant")["content"]
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
        events=EVENTS_AT_2,
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

    assert (
        render_sample(recipe=recipe, persistent=PERSISTENT, events=EVENTS_AT_2, t=0.0, sample_idx=0) is None
    )


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
        recipe=recipe, persistent=PERSISTENT, events=EVENTS_AT_2, t=0.0, sample_idx=123, task="x"
    )
    second = render_sample(
        recipe=recipe, persistent=PERSISTENT, events=EVENTS_AT_2, t=0.0, sample_idx=123, task="x"
    )
    assert first == second


def test_emitted_at_filters_vqa_by_camera():
    top = emitted_at(
        3.0,
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        style="vqa",
        role="assistant",
        camera="observation.images.top",
    )
    wrist = emitted_at(
        3.0,
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        style="vqa",
        role="assistant",
        camera="observation.images.wrist",
    )
    assert top["content"] == '{"count": 3}'
    assert wrist["content"] == '{"count": 1}'


def test_emitted_at_raises_on_ambiguous_per_camera_vqa():
    with pytest.raises(ValueError, match="Ambiguous resolver"):
        emitted_at(
            3.0,
            persistent=PERSISTENT,
            events=EVENTS_AT_3_TWO_CAMERAS,
            style="vqa",
            role="assistant",
        )


def test_per_camera_blend_renders_both_views():
    recipe = TrainingRecipe(
        blend={
            "top": TrainingRecipe(
                weight=1.0,
                bindings={
                    "vqa_query": (
                        "emitted_at(t, style=vqa, role=user, camera=observation.images.top)"
                    ),
                    "vqa": (
                        "emitted_at(t, style=vqa, role=assistant, camera=observation.images.top)"
                    ),
                },
                messages=[
                    MessageTurn(
                        role="user",
                        content=[
                            {"type": "image", "feature": "observation.images.top"},
                            {"type": "text", "text": "${vqa_query}"},
                        ],
                        stream="high_level",
                        if_present="vqa_query",
                    ),
                    MessageTurn(
                        role="assistant",
                        content="${vqa}",
                        stream="high_level",
                        target=True,
                        if_present="vqa",
                    ),
                ],
            ),
            "wrist": TrainingRecipe(
                weight=1.0,
                bindings={
                    "vqa_query": (
                        "emitted_at(t, style=vqa, role=user, camera=observation.images.wrist)"
                    ),
                    "vqa": (
                        "emitted_at(t, style=vqa, role=assistant, camera=observation.images.wrist)"
                    ),
                },
                messages=[
                    MessageTurn(
                        role="user",
                        content=[
                            {"type": "image", "feature": "observation.images.wrist"},
                            {"type": "text", "text": "${vqa_query}"},
                        ],
                        stream="high_level",
                        if_present="vqa_query",
                    ),
                    MessageTurn(
                        role="assistant",
                        content="${vqa}",
                        stream="high_level",
                        target=True,
                        if_present="vqa",
                    ),
                ],
            ),
        }
    )

    rendered_top = render_sample(
        recipe=recipe.blend["top"],
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        t=3.0,
        sample_idx=0,
    )
    rendered_wrist = render_sample(
        recipe=recipe.blend["wrist"],
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        t=3.0,
        sample_idx=0,
    )

    assert rendered_top["messages"][0]["content"][0]["feature"] == "observation.images.top"
    assert rendered_top["messages"][0]["content"][1]["text"] == "how many cups (top)?"
    assert rendered_top["messages"][1]["content"] == '{"count": 3}'

    assert rendered_wrist["messages"][0]["content"][0]["feature"] == "observation.images.wrist"
    assert rendered_wrist["messages"][0]["content"][1]["text"] == "how many cups (wrist)?"
    assert rendered_wrist["messages"][1]["content"] == '{"count": 1}'


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
