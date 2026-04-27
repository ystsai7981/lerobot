#!/usr/bin/env python

import torch

from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.processor.converters import create_transition
from lerobot.processor.render_messages_processor import RenderMessagesStep
from lerobot.types import TransitionKey


def test_render_messages_step_noops_without_language_columns():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
        ]
    )
    transition = create_transition(complementary_data={"task": "do it"})

    assert RenderMessagesStep(recipe)(transition) == transition


def test_render_messages_step_renders_and_drops_raw_language():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
        ]
    )
    transition = create_transition(
        complementary_data={
            "task": "do it",
            "timestamp": torch.tensor(0.0),
            "index": torch.tensor(7),
            "language_persistent": [
                {
                    "role": "assistant",
                    "content": "reach carefully",
                    "style": "subtask",
                    "timestamp": 0.0,
                    "tool_calls": None,
                }
            ],
            "language_events": [],
        }
    )

    out = RenderMessagesStep(recipe)(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert "language_persistent" not in data
    assert "language_events" not in data
    assert data["messages"][-1]["content"] == "reach carefully"
    assert data["message_streams"] == ["high_level", "low_level"]
    assert data["target_message_indices"] == [1]
