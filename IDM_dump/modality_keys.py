"""Resolve LeRobot parquet column names from meta/modality.json (joint vs cartesian)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def state_action_keys_from_modality(modality: dict[str, Any]) -> tuple[str, str]:
    """
    Return (state_column, action_column) used in LeRobot parquet files.

    Matches gr00t.data.schema defaults when original_key is omitted:
    state -> observation.state, action -> action.
    Cartesian ebots modality uses observation.cart_state and cart_action.
    """
    state_meta = next(iter(modality["state"].values()))
    action_meta = next(iter(modality["action"].values()))
    state_key = state_meta.get("original_key") or "observation.state"
    action_key = action_meta.get("original_key") or "action"
    return state_key, action_key


def state_action_keys_from_modality_path(modality_path: str | Path) -> tuple[str, str]:
    with open(modality_path, encoding="utf-8") as f:
        modality = json.load(f)
    return state_action_keys_from_modality(modality)
