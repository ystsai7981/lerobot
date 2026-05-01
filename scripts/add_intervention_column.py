"""Add an `intervention` boolean column (all False) to an existing LeRobotDataset.

Existing demos recorded before HIL-DAgger don't have the `intervention` column
that DAgger sessions append. To merge them with DAgger output via
`lerobot-edit-dataset --operation.type=merge`, schemas must match — so we
backfill `intervention=False` on every frame of the source dataset.

Output is a new dataset (this script doesn't modify in-place because
`add_features` always copies — see
src/lerobot/datasets/dataset_tools.py:388).

Usage:

    uv run python scripts/add_intervention_column.py \\
        --repo-id local/so101_pick_lift_cube \\
        --root data/so101_pick_lift_cube
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import add_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="Source repo_id (e.g. local/so101_pick_lift_cube)")
    p.add_argument("--root", required=True, help="Source dataset root (e.g. data/so101_pick_lift_cube)")
    p.add_argument(
        "--new-repo-id",
        default=None,
        help="Output repo_id. Default: <repo-id>_with_intervention",
    )
    p.add_argument(
        "--new-root",
        default=None,
        help="Output root. Default: <root>_with_intervention sibling dir",
    )
    args = p.parse_args()

    src = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    n_frames = src.num_frames
    print(f"Loaded {args.repo_id} ({n_frames} frames, {src.num_episodes} episodes)")

    if "intervention" in src.meta.features:
        raise SystemExit(f"'intervention' column already exists in {args.repo_id}; nothing to do")

    new_repo_id = args.new_repo_id or f"{args.repo_id}_with_intervention"
    new_root = Path(args.new_root) if args.new_root else Path(args.root).parent / f"{Path(args.root).name}_with_intervention"

    intervention_values = np.zeros((n_frames, 1), dtype=bool)
    feature_info = {"dtype": "bool", "shape": [1], "names": None}

    print(f"Writing {new_repo_id} -> {new_root}")
    out = add_features(
        dataset=src,
        features={"intervention": (intervention_values, feature_info)},
        output_dir=new_root,
        repo_id=new_repo_id,
    )
    print(f"Done. New dataset has {out.num_frames} frames, intervention column added (all False).")


if __name__ == "__main__":
    main()
