from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.eda import (audit_tables, code_flags, cramers_v, file_manifest,
                          numeric_profile, profile_columns, severity_rates, wilson_interval)
from src.data.load_stats19 import YEARS, add_temporal_columns, read_table


def test_missing_codes_do_not_treat_zero_as_missing():
    series = pd.Series([0, -1, 9, np.nan])
    flags = code_flags(series, {0: "Not at junction", -1: "Data missing or out of range", 9: "Unknown"})
    assert flags.any(axis=1).tolist() == [False, True, True, True]
    assert flags.sum().to_dict() == {"null": 1, "coded_missing": 1, "unknown": 1}


def test_partial_unknown_lighting_is_retained():
    flags = code_flags(pd.Series([7]), {7: "Darkness - lighting unknown"})
    assert not flags.any(axis=None)


def test_junction_structural_missingness_is_separate_without_mutation():
    frame = pd.DataFrame({"junction_detail_historic": [0, 3, 3], "junction_control": [-1, -1, 4]})
    original = frame.copy()
    result = profile_columns(frame, {"junction_control": {-1: "Data missing", 4: "Give way"}}).set_index("column")
    assert result.loc["junction_control", "missing_with_no_junction_count"] == 1
    assert result.loc["junction_control", "unavailable_fraction"] == pytest.approx(2 / 3)
    assert result.loc["junction_control", "unavailable_excluding_no_junction_fraction"] == pytest.approx(1 / 3)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("date,season,weekday", [
    ("01/01/2019", "Winter", 1), ("29/02/2020", "Winter", 5),
    ("01/03/2019", "Spring", 4), ("01/06/2019", "Summer", 5),
    ("01/09/2019", "Autumn", 6), ("31/12/2019", "Winter", 1),
])
def test_temporal_features(date, season, weekday):
    original = pd.DataFrame({"date": [date], "time": ["23:59"]})
    result = add_temporal_columns(original)
    assert result.loc[0, "season"] == season
    assert result.loc[0, "weekday"] == weekday
    assert result.loc[0, "hour"] == 23
    assert original.loc[0, "date"] == date


def test_invalid_dates_and_times_are_flagged_not_guessed():
    result = add_temporal_columns(pd.DataFrame({"date": ["31/02/2019"], "time": ["24:60"]}))
    assert result[["date", "hour", "month", "weekday", "season"]].isna().all().all()


def test_severity_rates_use_within_group_denominators_and_all_classes():
    result = severity_rates(pd.DataFrame({"weather": ["dry", "dry", "wet"], "collision_severity": [2, 3, 3]}), "weather").set_index("weather")
    assert result.loc["dry", "fatal"] == 0
    assert result.loc["dry", "n"] == 2
    assert result.loc["dry", "severe_fraction"] == 0.5
    assert result.loc["wet", "severe_fraction"] == 0
    assert np.allclose(result[["fatal_fraction", "serious_fraction", "slight_fraction"]].sum(axis=1), 1)
    assert result["small_group"].all()


def test_wilson_intervals_cover_extremes_and_zero_sample():
    low, high = wilson_interval(np.array([0, 5, 10, 0]), np.array([10, 10, 10, 0]))
    assert low[0] == pytest.approx(0)
    assert high[2] == pytest.approx(1)
    assert low[1] < 0.5 < high[1]
    assert np.isnan(low[3]) and np.isnan(high[3])


@pytest.mark.parametrize("kind,expected", [("perfect", 1), ("independent", 0), ("constant", 0)])
def test_cramers_v(kind, expected):
    left = pd.Series([0, 0, 1, 1] * 20)
    right = left if kind == "perfect" else pd.Series([0, 1, 0, 1] * 20) if kind == "independent" else pd.Series([0] * 80)
    assert cramers_v(left, right) == pytest.approx(expected)


def test_numeric_profile_excludes_coded_missing_without_removing_rows():
    frame = pd.DataFrame({"speed_limit": [-1, 30, 70], "number_of_vehicles": [1, 2, 17],
                          "number_of_casualties": [1, 1, 52], "latitude": [50, 51, 60], "longitude": [-1, -2, -7]})
    original = frame.copy()
    result = numeric_profile(frame, {"speed_limit": {-1: "Data missing"}}).set_index("column")
    assert result.loc["speed_limit", "valid_n"] == 2
    assert result.loc["speed_limit", "min"] == 30
    assert result.loc["number_of_vehicles", "max"] == 17
    pd.testing.assert_frame_equal(frame, original)


def write_tables(directory: Path):
    for year in YEARS:
        key = f"{year}000000001"
        collision = pd.DataFrame({"collision_index": [key], "collision_year": [year],
                                  "collision_severity": [2], "date": [f"01/01/{year}"], "time": ["12:00"],
                                  "day_of_week": [3], "number_of_vehicles": [1], "number_of_casualties": [1]})
        vehicle = pd.DataFrame({"collision_index": [key], "collision_year": [year], "vehicle_reference": [1]})
        casualty = pd.DataFrame({"collision_index": [key], "collision_year": [year], "vehicle_reference": [1],
                                 "casualty_reference": [1], "casualty_severity": [2]})
        for kind, frame in [("collision", collision), ("vehicle", vehicle), ("casualty", casualty)]:
            frame.to_csv(directory / f"dft-road-casualty-statistics-{kind}-{year}.csv", index=False)


def test_table_audit_valid_keys_counts_and_severity(tmp_path):
    write_tables(tmp_path)
    report, paths, columns = audit_tables(tmp_path)
    assert len(paths) == 9 and "collision_index" in columns
    for checks in report.values():
        assert checks["collision_schema_equal"]
        assert all(value == 0 for key, value in checks.items() if not key.endswith("_rows") and key != "collision_schema_equal")
        assert checks["orphan_vehicle_rows"] == checks["orphan_casualty_rows"] == 0


def test_audit_detects_orphan_casualty_and_outcome_mismatch(tmp_path):
    write_tables(tmp_path)
    path = tmp_path / "dft-road-casualty-statistics-casualty-2019.csv"
    frame = pd.read_csv(path)
    frame["collision_index"] = "orphan"
    frame["casualty_severity"] = 1
    frame.to_csv(path, index=False)
    result = audit_tables(tmp_path)[0]["2019"]
    assert result["orphan_casualty_rows"] == 1
    assert result["casualties_without_matching_vehicle"] == 1
    assert result["casualty_count_mismatch_collisions"] == 1
    assert result["collision_vs_most_severe_casualty_mismatch"] == 1


def test_audit_detects_duplicate_composite_keys(tmp_path):
    write_tables(tmp_path)
    path = tmp_path / "dft-road-casualty-statistics-vehicle-2020.csv"
    vehicle = pd.read_csv(path)
    pd.concat([vehicle, vehicle]).to_csv(path, index=False)
    result = audit_tables(tmp_path)[0]["2020"]
    assert result["vehicle_duplicate_keys"] == result["vehicle_duplicate_full_rows"] == 1
    assert result["vehicle_count_mismatch_collisions"] == 1


@pytest.mark.parametrize("problem,message", [("year", "outside requested year"), ("column", "missing required columns")])
def test_loader_rejects_wrong_year_or_schema(tmp_path, problem, message):
    write_tables(tmp_path)
    path = tmp_path / "dft-road-casualty-statistics-collision-2021.csv"
    frame = pd.read_csv(path)
    if problem == "year":
        frame["collision_year"] = 2020
    else:
        frame = frame.drop(columns="time")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        read_table(tmp_path, "collision", 2021)


def test_manifest_uses_relative_paths_and_content_fingerprints(tmp_path):
    path = tmp_path / "fixture.txt"
    path.write_bytes(b"abc")
    result = file_manifest([path], tmp_path)[0]
    assert result == {"path": "fixture.txt", "bytes": 3,
                      "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
