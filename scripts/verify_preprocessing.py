"""Independently check saved Phase 3 outputs against their raw collision records."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.eda import file_manifest
from src.data.load_stats19 import read_table


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = root / "reports/preprocessing"
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    require(not summary["classifier_trained"], "Unexpected classifier flag")
    for manifest_name in ("input_manifest", "artifact_manifest"):
        items = summary[manifest_name]
        require(file_manifest([root / item["path"] for item in items], root) == items,
                f"Changed files: {manifest_name}")
    pipeline = joblib.load(root / "models/preprocessing_context_v1.joblib")
    require(list(pipeline.get_feature_names_out()) == summary["output_feature_names"], "Feature order mismatch")
    state = joblib.hash(pipeline)
    all_keys = set()
    for role, info in summary["splits"].items():
        output = root / "data/processed/context_v1"
        rows = pd.read_csv(output / f"rows_{role}.csv", dtype={"collision_index": "string"})
        raw = read_table(root / "data/raw/stats19", "collision", info["year"])
        require(set(rows["collision_index"]) == set(raw["collision_index"]), f"Lost or extra rows: {role}")
        keys = set(rows["collision_index"])
        require(not keys & all_keys, "Partition key overlap")
        all_keys.update(keys)
        ordered = raw.set_index("collision_index").loc[rows["collision_index"]].reset_index()
        expected_time = (pd.to_datetime(ordered["date"], format="%d/%m/%Y")
                         + pd.to_timedelta(ordered["time"] + ":00"))
        require(expected_time.is_monotonic_increasing, f"Unsorted source rows: {role}")
        require(np.array_equal(expected_time.to_numpy(), pd.to_datetime(rows["timestamp_local"]).to_numpy()),
                f"Sidecar timestamps do not match source: {role}")
        require(np.array_equal(rows["row_position"].to_numpy(), np.arange(len(rows))), "Row position mismatch")
        y = np.load(output / f"y_{role}.npy", allow_pickle=False)
        require(y.dtype == np.int64 and np.array_equal(y, ordered["collision_severity"].to_numpy() - 1),
                f"Target mapping/alignment mismatch: {role}")
        X = sparse.load_npz(output / f"X_{role}.npz")
        require(X.shape == tuple(info["shape"]) and X.dtype == np.float32 and np.isfinite(X.data).all(),
                f"Invalid matrix: {role}")
        require((X - pipeline.transform(ordered)).nnz == 0, f"Full artifact round-trip mismatch: {role}")
        require(joblib.hash(pipeline) == state, "Transform changed fit state")
        if role == "train":
            speed = ordered["speed_limit"].where(ordered["speed_limit"].isin([20, 30, 40, 50, 60, 70]))
            median = speed.median()
            imputed = speed.fillna(median)
            statistics = summary["statistics"]
            require(np.isclose(median, statistics["speed_median_mph"]), "Training median mismatch")
            require(np.isclose(imputed.mean(), statistics["speed_mean_after_imputation_mph"]), "Training mean mismatch")
            require(np.isclose(imputed.var(ddof=0), statistics["speed_variance_after_imputation"]), "Training variance mismatch")
            require(statistics["fit_samples"] == len(ordered), "Training fit count mismatch")
    result = {"all_checks_passed": True, "raw_input_files_verified": len(summary["input_manifest"]),
              "saved_artifacts_verified": len(summary["artifact_manifest"]), "rows_verified": len(all_keys),
              "independent_script_checks": True, "full_partition_round_trip": True,
              "raw_target_and_timestamp_alignment": True, "independent_train_statistics": True}
    (report / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
