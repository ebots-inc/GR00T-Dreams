#!/usr/bin/env python3

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_IDM_DUMP_DIR = str(Path(__file__).resolve().parent)
if _IDM_DUMP_DIR not in sys.path:
    sys.path.insert(0, _IDM_DUMP_DIR)
from modality_keys import state_action_keys_from_modality


def _as_1d_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array-like, got shape {arr.shape}")
    return arr


def _atomic_write_parquet(df: pd.DataFrame, dst_path: str) -> None:
    dst_dir = os.path.dirname(dst_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_fill_states_", suffix=".parquet", dir=dst_dir)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, dst_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def fill_episode_states_inplace(
    parquet_path: str,
    dryrun: bool,
    state_col: str,
    action_col: str,
) -> tuple[int, int]:
    """
    Fill state column from action column with a 1-step lag (LeRobot column names from modality.json):
      - state[0] = action[0]
      - state[t] = action[t-1] for t>=1

    Joint space: observation.state / action. Cartesian: observation.cart_state / cart_action.

    If the state vector is longer than the action vector, only the first A dims are overwritten.
    Returns: (n_rows, n_dims_overwritten)
    """
    df = pd.read_parquet(parquet_path)
    if action_col not in df.columns:
        raise ValueError(f"Missing action column {action_col!r}: {parquet_path}")

    if len(df) == 0:
        return 0, 0

    actions = [_as_1d_float_array(x) for x in df[action_col].to_numpy()]
    a0 = actions[0]
    a_dim = int(a0.shape[0])

    if state_col in df.columns:
        states = [_as_1d_float_array(x) for x in df[state_col].to_numpy()]
        s_dim = int(states[0].shape[0])
    else:
        s_dim = a_dim
        states = [np.zeros((s_dim,), dtype=np.float32) for _ in range(len(df))]

    k = min(s_dim, a_dim)
    out_states: list[list[float]] = []

    for t in range(len(df)):
        base = states[t].copy()
        src = actions[t] if t == 0 else actions[t - 1]
        base[:k] = src[:k]
        out_states.append(base.tolist())

    if not dryrun:
        df[state_col] = out_states
        _atomic_write_parquet(df, parquet_path)

    return len(df), k


def recompute_stats_json(
    dataset: str,
    parquet_paths: list[str],
    stat_keys: list[str] | None = None,
) -> None:
    """
    Recompute meta/stats.json from parquet files.

    If 'stat_keys' is None, uses the state and action column names from 'meta/modality.json'.
    Otherwise computes statistics for each listed column name that exists in the parquets and
    merges into 'meta/stats.json' (existing keys not recomputed are preserved).
    """
    if stat_keys is None:
        modality_path = os.path.join(dataset, "meta", "modality.json")
        with open(modality_path, encoding="utf-8") as f:
            stat_keys = list(state_action_keys_from_modality(json.load(f)))
    all_data: dict[str, list] = {k: [] for k in stat_keys}

    for path in sorted(parquet_paths):
        df = pd.read_parquet(path)
        for k in stat_keys:
            if k in df.columns:
                for row in df[k]:
                    arr = np.asarray(row, dtype=np.float32).ravel()
                    all_data[k].append(arr)

    le_statistics: dict = {}
    for k in stat_keys:
        if not all_data[k]:
            continue
        np_data = np.vstack(all_data[k])
        le_statistics[k] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }

    meta_dir = os.path.join(dataset, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    stats_path = os.path.join(meta_dir, "stats.json")
    existing: dict = {}
    if os.path.isfile(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(le_statistics)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)
    print(f"Recomputed stats from {len(parquet_paths)} parquet files -> {stats_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fill state from action with a 1-step lag (in-place). "
        "Column names default to meta/modality.json; use --state-col/--action-col to override "
        "(e.g. ebots joint vs cartesian pairs without swapping modality.json)."
    )
    p.add_argument("--dataset", type=str, required=True, help="LeRobot dataset directory (contains data/chunk-*/).")
    p.add_argument("--dryrun", action="store_true", help="Only report; do not modify files.")
    p.add_argument(
        "--state-col",
        type=str,
        default=None,
        help="Parquet state column (e.g. observation.state or observation.cart_state). "
        "If set, --action-col must also be set.",
    )
    p.add_argument(
        "--action-col",
        type=str,
        default=None,
        help="Parquet action column (e.g. action or cart_action). If set, --state-col must also be set.",
    )
    p.add_argument(
        "--recompute_stats",
        action="store_true",
        help="After filling states, recompute meta/stats.json from parquet for selected columns.",
    )
    p.add_argument(
        "--recompute-stats-columns",
        type=str,
        nargs="+",
        default=None,
        metavar="COL",
        help="With --recompute_stats, compute stats for these parquet columns and merge into meta/stats.json. "
        "If omitted, uses the state and action columns from meta/modality.json only.",
    )
    p.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip filling states; only run --recompute_stats (requires --recompute-stats-columns). "
        "Use to refresh timestamp/index stats or other columns without re-running the lag fill.",
    )
    args = p.parse_args()

    dataset = os.path.expanduser(args.dataset)
    pattern = os.path.join(dataset, "data", "chunk-*", "episode_*.parquet")
    parquet_paths = sorted(glob.glob(pattern))
    if not parquet_paths:
        raise SystemExit(f"No episode parquets found under: {pattern}")

    if args.stats_only:
        if not args.recompute_stats:
            raise SystemExit("--stats-only requires --recompute_stats")
        if not args.recompute_stats_columns:
            raise SystemExit("--stats-only requires --recompute-stats-columns (list columns to compute)")
        recompute_stats_json(dataset, parquet_paths, stat_keys=list(args.recompute_stats_columns))
        print(f"stats-only: wrote meta/stats.json entries for {len(args.recompute_stats_columns)} column(s)")
        return 0

    sc, ac = args.state_col, args.action_col
    if (sc is None) ^ (ac is None):
        raise SystemExit("Provide both --state-col and --action-col, or neither (use meta/modality.json).")
    if sc is not None and ac is not None:
        state_col, action_col = sc, ac
    else:
        modality_path = os.path.join(dataset, "meta", "modality.json")
        with open(modality_path, encoding="utf-8") as f:
            state_col, action_col = state_action_keys_from_modality(json.load(f))

    files = 0
    total_rows = 0
    last_k = None
    for path in parquet_paths:
        n_rows, k = fill_episode_states_inplace(
            path, dryrun=args.dryrun, state_col=state_col, action_col=action_col
        )
        files += 1
        total_rows += n_rows
        last_k = k

    mode = "dryrun" if args.dryrun else "inplace"
    print(f"mode={mode} files={files} total_rows={total_rows} overwritten_dims={last_k}")

    if args.recompute_stats and not args.dryrun and parquet_paths:
        if args.recompute_stats_columns:
            recompute_stats_json(dataset, parquet_paths, stat_keys=list(args.recompute_stats_columns))
        else:
            recompute_stats_json(dataset, parquet_paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

