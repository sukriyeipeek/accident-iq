"""Inspect downloaded STATS19 files without fitting transformations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FILE_TYPES = ("collision", "vehicle", "casualty")


def inspect_file(path: Path) -> None:
    frame = pd.read_csv(path, encoding="latin1", low_memory=False)
    print(f"\n{path.name}")
    print(f"rows={len(frame):,} columns={len(frame.columns)}")
    print("columns:")
    print(", ".join(frame.columns))
    print("top missingness:")
    missing = frame.isna().mean().sort_values(ascending=False).head(10)
    for column, ratio in missing.items():
        print(f"  {column}: {ratio:.2%}")

    if "collision_severity" in frame.columns:
        print("collision_severity distribution:")
        print(frame["collision_severity"].value_counts(dropna=False).sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "stats19",
    )
    parser.add_argument("--year", type=int, default=2021)
    args = parser.parse_args()

    for file_type in FILE_TYPES:
        path = args.data_dir / f"dft-road-casualty-statistics-{file_type}-{args.year}.csv"
        if not path.exists() or path.stat().st_size == 0:
            print(f"MISSING: {path}")
            continue
        inspect_file(path)


if __name__ == "__main__":
    main()
