"""Audit STATS19 keys and cross-table integrity without modifying raw records."""

import json
from pathlib import Path

from src.data.eda import audit_tables


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report, _, _ = audit_tables(root / "data" / "raw" / "stats19")
    output = root / "reports" / "eda"
    output.mkdir(parents=True, exist_ok=True)
    (output / "link_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures = {year: [key for key, value in checks.items()
                       if (key == "collision_schema_equal" and not value)
                       or (key not in {"collision_schema_equal", "collision_rows", "vehicle_rows", "casualty_rows"}
                           and value != 0)]
                for year, checks in report.items()}
    print(json.dumps(report, indent=2))
    if any(failures.values()):
        raise ValueError(f"Table integrity failures: {failures}")


if __name__ == "__main__":
    main()
