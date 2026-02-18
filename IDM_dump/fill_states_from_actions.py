#!/usr/bin/env python3

import argparse
import glob
import os
import tempfile

import numpy as np
import pandas as pd


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


def fill_episode_states_inplace(parquet_path: str, dryrun: bool) -> tuple[int, int]:
    """
    Fill `observation.state` from `action` with a 1-step lag:
      - state[0] = action[0]
      - state[t] = action[t-1] for t>=1

    If the state vector is longer than the action vector, only the first A dims are overwritten.
    Returns: (n_rows, n_dims_overwritten)
    """
    df = pd.read_parquet(parquet_path)
    if "action" not in df.columns:
        raise ValueError(f"Missing 'action' column: {parquet_path}")

    if len(df) == 0:
        return 0, 0

    actions = [_as_1d_float_array(x) for x in df["action"].to_numpy()]
    a0 = actions[0]
    a_dim = int(a0.shape[0])

    if "observation.state" in df.columns:
        states = [_as_1d_float_array(x) for x in df["observation.state"].to_numpy()]
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
        df["observation.state"] = out_states
        _atomic_write_parquet(df, parquet_path)

    return len(df), k


def main() -> int:
    p = argparse.ArgumentParser(description="Fill observation.state from action with a 1-step lag (in-place).")
    p.add_argument("--dataset", type=str, required=True, help="LeRobot dataset directory (contains data/chunk-*/).")
    p.add_argument("--dryrun", action="store_true", help="Only report; do not modify files.")
    args = p.parse_args()

    dataset = os.path.expanduser(args.dataset)
    pattern = os.path.join(dataset, "data", "chunk-*", "episode_*.parquet")
    parquet_paths = sorted(glob.glob(pattern))
    if not parquet_paths:
        raise SystemExit(f"No episode parquets found under: {pattern}")

    files = 0
    total_rows = 0
    last_k = None
    for path in parquet_paths:
        n_rows, k = fill_episode_states_inplace(path, dryrun=args.dryrun)
        files += 1
        total_rows += n_rows
        last_k = k

    mode = "dryrun" if args.dryrun else "inplace"
    print(f"mode={mode} files={files} total_rows={total_rows} overwritten_dims={last_k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

