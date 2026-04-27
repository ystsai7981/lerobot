#!/usr/bin/env python

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.language import (
    EVENT_ONLY_STYLES,
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    PERSISTENT_STYLES,
    STYLE_REGISTRY,
    column_for_style,
    language_events_arrow_type,
    language_feature_info,
    language_persistent_arrow_type,
)
from lerobot.datasets.utils import DEFAULT_DATA_PATH


def test_language_arrow_schema_has_expected_fields():
    row_type = language_persistent_arrow_type().value_type

    assert isinstance(row_type, pa.StructType)
    assert row_type.names == ["role", "content", "style", "timestamp", "tool_calls"]
    assert language_events_arrow_type().value_type == row_type


def test_style_registry_routes_columns():
    assert {"subtask", "plan", "memory"} == PERSISTENT_STYLES
    assert {"interjection", "vqa"} == EVENT_ONLY_STYLES
    assert PERSISTENT_STYLES | EVENT_ONLY_STYLES <= STYLE_REGISTRY

    assert column_for_style("subtask") == LANGUAGE_PERSISTENT
    assert column_for_style("plan") == LANGUAGE_PERSISTENT
    assert column_for_style("memory") == LANGUAGE_PERSISTENT
    assert column_for_style("interjection") == LANGUAGE_EVENTS
    assert column_for_style("vqa") == LANGUAGE_EVENTS
    assert column_for_style(None) == LANGUAGE_EVENTS


def test_unknown_style_rejected():
    with pytest.raises(ValueError, match="Unknown language style"):
        column_for_style("surprise")


def test_lerobot_dataset_passes_language_columns_through(tmp_path, empty_lerobot_dataset_factory):
    root = tmp_path / "language_dataset"
    dataset = empty_lerobot_dataset_factory(
        root=root,
        features={"state": {"dtype": "float32", "shape": (2,), "names": None}},
        use_videos=False,
    )
    dataset.add_frame({"state": np.array([0.0, 1.0], dtype=np.float32), "task": "tidy"})
    dataset.add_frame({"state": np.array([1.0, 2.0], dtype=np.float32), "task": "tidy"})
    dataset.save_episode()
    dataset.finalize()

    persistent = [
        {
            "role": "assistant",
            "content": "reach for the cup",
            "style": "subtask",
            "timestamp": 0.0,
            "tool_calls": None,
        }
    ]
    event = {
        "role": "user",
        "content": "what is visible?",
        "style": "vqa",
        "timestamp": 0.0,
        "tool_calls": None,
    }
    data_path = root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    df = pd.read_parquet(data_path)
    df[LANGUAGE_PERSISTENT] = [persistent, persistent]
    df[LANGUAGE_EVENTS] = [[event], []]
    df.to_parquet(data_path)

    info = dataset.meta.info
    info["features"].update(language_feature_info())
    write_info(info, root)

    reloaded = LeRobotDataset(repo_id=dataset.repo_id, root=root)

    first = reloaded[0]
    second = reloaded[1]
    assert first[LANGUAGE_PERSISTENT] == persistent
    assert first[LANGUAGE_EVENTS] == [event]
    assert second[LANGUAGE_PERSISTENT] == persistent
    assert second[LANGUAGE_EVENTS] == []
