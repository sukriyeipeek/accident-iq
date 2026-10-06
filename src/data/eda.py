from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_stats19 import YEARS, read_table


NUMERIC_COLUMNS = ["speed_limit", "number_of_vehicles", "number_of_casualties", "latitude", "longitude"]
ANALYSIS_COLUMNS = ["hour", "weekday", "month", "season", "weather_conditions_label",
                    "road_surface_conditions_label", "light_conditions_label", "road_type_label",
                    "junction_detail_historic_label", "junction_control_label", "speed_limit",
                    "urban_or_rural_area_label"]


def code_flags(series: pd.Series, mapping: dict[int, str]) -> pd.DataFrame:
    missing_codes = {code for code, label in mapping.items() if "data missing" in label.lower()}
    unknown_codes = {code for code, label in mapping.items()
                     if label.lower().startswith("unknown") or label.lower() == "unallocated"}
    return pd.DataFrame({
        "null": series.isna(),
        "coded_missing": series.isin(missing_codes),
        "unknown": series.isin(unknown_codes),
    })


def profile_columns(frame: pd.DataFrame, codebook: dict) -> pd.DataFrame:
    rows = []
    for name, series in frame.items():
        flags = code_flags(series, codebook.get(name, {}))
        no_junction = (frame["junction_detail_historic"].eq(0) & flags.any(axis=1)
                       if name == "junction_control" and "junction_detail_historic" in frame
                       else pd.Series(False, index=frame.index))
        rows.append({"column": name, "dtype": str(series.dtype), "rows": len(series),
                     "unique": series.nunique(dropna=True),
                     "null_count": int(flags["null"].sum()),
                     "coded_missing_count": int(flags["coded_missing"].sum()),
                     "unknown_count": int(flags["unknown"].sum()),
                     "missing_with_no_junction_count": int(no_junction.sum()),
                     "unavailable_fraction": float(flags.any(axis=1).mean()),
                     "unavailable_excluding_no_junction_fraction": float((flags.any(axis=1) & ~no_junction).mean())})
    return pd.DataFrame(rows).sort_values("unavailable_fraction", ascending=False)


def severity_rates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = pd.crosstab(frame[column], frame["collision_severity"], dropna=True).reindex(columns=[1, 2, 3], fill_value=0)
    counts.columns = ["fatal", "serious", "slight"]
    counts["n"] = counts[["fatal", "serious", "slight"]].sum(axis=1)
    counts["severe"] = counts["fatal"] + counts["serious"]
    for label in ("fatal", "serious", "slight", "severe"):
        counts[f"{label}_fraction"] = counts[label] / counts["n"]
    low, high = wilson_interval(counts["severe"].to_numpy(), counts["n"].to_numpy())
    counts["severe_ci_low"] = low
    counts["severe_ci_high"] = high
    counts["small_group"] = counts["n"] < 100
    return counts.reset_index()


def wilson_interval(successes: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.divide(successes, n, out=np.zeros_like(n, dtype=float), where=n > 0)
    z = 1.96
    safe_n = np.where(n > 0, n, np.nan)
    center = (p + z * z / (2 * safe_n)) / (1 + z * z / safe_n)
    half = z * np.sqrt(p * (1 - p) / safe_n + z * z / (4 * safe_n ** 2)) / (1 + z * z / safe_n)
    return np.maximum(0, center - half), np.minimum(1, center + half)


def numeric_profile(frame: pd.DataFrame, codebook: dict) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_COLUMNS:
        series = frame[column]
        clean = series.mask(code_flags(series, codebook.get(column, {})).any(axis=1)).dropna()
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"column": column, "valid_n": len(clean), "min": clean.min(),
                     "q1": q1, "median": clean.median(), "q3": q3, "max": clean.max(),
                     "mean": clean.mean(), "std": clean.std(), "iqr_lower": lower,
                     "iqr_upper": upper, "iqr_flagged_n": int(((clean < lower) | (clean > upper)).sum())})
    return pd.DataFrame(rows)


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    counts = pd.crosstab(left, right).to_numpy(dtype=float)
    n = counts.sum()
    r, k = counts.shape
    if n <= 1 or min(r, k) < 2:
        return 0.0
    expected = counts.sum(axis=1, keepdims=True) @ counts.sum(axis=0, keepdims=True) / n
    chi2 = np.divide((counts - expected) ** 2, expected, out=np.zeros_like(expected), where=expected > 0).sum()
    phi2 = max(0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    denominator = min(r - (r - 1) ** 2 / (n - 1) - 1, k - (k - 1) ** 2 / (n - 1) - 1)
    return math.sqrt(phi2 / denominator) if denominator > 0 else 0.0


def audit_tables(data_dir: Path) -> tuple[dict, list[Path], list[str]]:
    report = {}
    paths = []
    raw_columns = []
    for year in YEARS:
        frames = {kind: read_table(data_dir, kind, year) for kind in ("collision", "vehicle", "casualty")}
        collision, vehicle, casualty = (frames[kind] for kind in ("collision", "vehicle", "casualty"))
        if not raw_columns:
            raw_columns = list(collision.columns)
        checks = {"collision_schema_equal": list(collision.columns) == raw_columns}
        collision_ids = set(collision["collision_index"])
        for kind, frame in frames.items():
            paths.append(data_dir / f"dft-road-casualty-statistics-{kind}-{year}.csv")
            key = ["collision_index"]
            if kind == "vehicle":
                key += ["vehicle_reference"]
            elif kind == "casualty":
                key += ["casualty_reference"]
            checks[f"{kind}_rows"] = len(frame)
            checks[f"{kind}_null_keys"] = int(frame[key].isna().any(axis=1).sum())
            checks[f"{kind}_duplicate_keys"] = int(frame.duplicated(key).sum())
            checks[f"{kind}_duplicate_full_rows"] = int(frame.duplicated().sum())
            if kind != "collision":
                checks[f"orphan_{kind}_rows"] = int((~frame["collision_index"].isin(collision_ids)).sum())
                counts = frame.groupby("collision_index").size()
                expected = collision.set_index("collision_index")[f"number_of_{'vehicles' if kind == 'vehicle' else 'casualties'}"]
                checks[f"{kind}_count_mismatch_collisions"] = int(expected.ne(counts.reindex(expected.index, fill_value=0)).sum())
        vehicle_keys = pd.MultiIndex.from_frame(vehicle[["collision_index", "vehicle_reference"]])
        casualty_vehicle_keys = pd.MultiIndex.from_frame(casualty[["collision_index", "vehicle_reference"]])
        checks["casualties_without_matching_vehicle"] = int((~casualty_vehicle_keys.isin(vehicle_keys)).sum())
        max_severity = casualty.groupby("collision_index")["casualty_severity"].min()
        expected_severity = collision.set_index("collision_index")["collision_severity"]
        checks["collision_vs_most_severe_casualty_mismatch"] = int(expected_severity.ne(max_severity.reindex(expected_severity.index)).sum())
        report[str(year)] = checks
    return report, paths, raw_columns


def file_manifest(paths: list[Path], root: Path) -> list[dict]:
    result = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        result.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                       "sha256": digest.hexdigest()})
    return result
