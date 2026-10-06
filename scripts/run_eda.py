"""Run from the repository root: python -m scripts.run_eda."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.eda_report import run_eda


def main() -> None:
    parser = argparse.ArgumentParser(description="AccidentIQ Phase 2 EDA")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(run_eda(args.root), indent=2))


if __name__ == "__main__":
    main()
