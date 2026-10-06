import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError

from src.data.splits import quarterly_folds, temporal_split
from src.features.preprocessing import (CATEGORICAL_FEATURES, ENGINEERED_FEATURES, RAW_FEATURES,
                                        MISSING, NOT_APPLICABLE, UNKNOWN, UNRECOGNIZED, ContextFeatures,
                                        encode_target, make_model_pipeline, make_preprocessor, recorded_timestamp)


@pytest.fixture
def codebook():
    definitions = {c: {-1: "Data missing or out of range", 1: f"{c}_one", 2: f"{c}_two", 9: "Unknown"}
                   for c in CATEGORICAL_FEATURES}
    definitions["junction_detail_historic"].update({0: "Outside junction", 3: "T junction"})
    definitions["junction_control"].update({0: "Outside junction", 4: "Give way"})
    definitions["light_conditions"].update({7: "Darkness - lighting unknown"})
    return definitions


@pytest.fixture
def records():
    frame = pd.DataFrame({column: [1, 1, 1, 1] for column in CATEGORICAL_FEATURES})
    frame["date"] = ["01/01/2019", "01/04/2019", "01/07/2019", "01/10/2019"]
    frame["time"] = ["00:00", "06:00", "12:00", "23:59"]
    frame["speed_limit"] = [20, 30, -1, 70]
    frame["junction_detail_historic"] = [0, 3, 3, 0]
    frame["junction_control"] = [-1, -1, 4, 4]
    return frame


def test_context_selects_only_allowlisted_inputs_and_preserves_rows(codebook, records):
    original = records.copy()
    builder = ContextFeatures(codebook).fit(records)
    plain = builder.transform(records)
    dirty = records.assign(collision_severity=[1, 2, 3, 1], number_of_casualties=999,
                           collision_year=9999, latitude=100, collision_index="fake", number_of_vehicles=1000)
    pd.testing.assert_frame_equal(builder.transform(dirty), plain)
    pd.testing.assert_frame_equal(records, original)
    assert list(plain.columns) == list(ENGINEERED_FEATURES)
    assert plain.index.equals(records.index)


def test_structural_missing_control_and_actual_junction_missing_are_different(codebook, records):
    result = ContextFeatures(codebook).fit_transform(records)
    assert result["junction_control"].tolist() == [NOT_APPLICABLE, MISSING, "Give way", NOT_APPLICABLE]
    assert result.loc[0, "junction_detail_historic"] == "Outside junction"


def test_category_missing_unknown_and_unrecognized_are_distinct(codebook, records):
    records["weather_conditions"] = [np.nan, -1, 9, 987]
    result = ContextFeatures(codebook).fit_transform(records)
    assert result["weather_conditions"].tolist() == [MISSING, MISSING, UNKNOWN, UNRECOGNIZED]
    assert result.loc[0, "weather_conditions"] != result.loc[2, "weather_conditions"]


def test_unknown_lighting_subfield_is_not_discarded(codebook, records):
    records["light_conditions"] = 7
    result = ContextFeatures(codebook).fit_transform(records)
    assert result["light_conditions"].eq("Darkness - lighting unknown").all()


def test_cyclic_boundaries_and_minute_resolution(codebook, records):
    result = ContextFeatures(codebook).fit_transform(records)
    assert result.loc[0, "hour_sin"] == pytest.approx(0)
    assert result.loc[0, "hour_cos"] == pytest.approx(1)
    assert result.loc[1, "hour_sin"] == pytest.approx(1)
    assert result.loc[2, "hour_cos"] == pytest.approx(-1)
    assert np.linalg.norm(result.loc[0, ["hour_sin", "hour_cos"]].to_numpy(dtype=float)
                          - result.loc[3, ["hour_sin", "hour_cos"]].to_numpy(dtype=float)) < 0.01
    assert result.loc[0, "month_sin"] == pytest.approx(0)
    assert result.loc[1, "month_sin"] == pytest.approx(1)
    assert result.loc[0, "weekday_sin"] == pytest.approx(np.sin(2 * np.pi / 7))


@pytest.mark.parametrize("column,value", [("date", "31/02/2019"), ("date", None), ("date", "01-02-2019"),
                                         ("time", "24:00"), ("time", "12:60"), ("time", None)])
def test_invalid_timestamp_raises(codebook, records, column, value):
    records.loc[0, column] = value
    with pytest.raises(ValueError, match="date/time"):
        ContextFeatures(codebook).fit(records)


def test_iso_and_source_dates_match_and_tz_is_rejected(records):
    iso = records.copy()
    iso["date"] = pd.to_datetime(records["date"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    pd.testing.assert_series_equal(recorded_timestamp(iso), recorded_timestamp(records))
    iso["date"] = pd.to_datetime(iso["date"]).dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        recorded_timestamp(iso)


@pytest.mark.parametrize("value", ["rain", 1.5, np.inf, True])
def test_malformed_codes_raise(codebook, records, value):
    records["weather_conditions"] = records["weather_conditions"].astype(object)
    records.loc[0, "weather_conditions"] = value
    with pytest.raises(ValueError, match="integer codes"):
        ContextFeatures(codebook).fit(records)


def test_input_contract_empty_missing_and_duplicate_columns(codebook, records):
    for bad, message in [(records.iloc[:0], "Nonempty"), (records.drop(columns="time"), "Missing required"),
                          (pd.concat([records, records[["time"]]], axis=1), "Duplicate input")]:
        with pytest.raises(ValueError, match=message):
            ContextFeatures(codebook).fit(bad)


def test_transform_requires_fit(codebook, records):
    with pytest.raises(NotFittedError):
        make_preprocessor(codebook).transform(records)


def test_median_and_scaler_use_train_only_and_transform_never_refits(codebook, records):
    pipeline = make_preprocessor(codebook).fit(records)
    speed = pipeline.named_steps["columns"].named_transformers_["speed"]
    assert speed.named_steps["impute"].statistics_[0] == 30
    assert speed.named_steps["scale"].mean_[0] == pytest.approx(37.5)
    assert speed.named_steps["scale"].n_samples_seen_ == 4
    state = joblib.hash(pipeline)
    heldout = records.assign(speed_limit=70, weather_conditions=2)
    matrix = pipeline.transform(heldout)
    assert joblib.hash(pipeline) == state
    assert matrix.shape[1] == len(pipeline.get_feature_names_out())
    assert matrix.dtype == np.float32 and sparse.isspmatrix_csr(matrix)
    assert np.isfinite(matrix.data).all()
    encoder = pipeline.named_steps["columns"].named_transformers_["categories"]
    assert "weather_conditions_two" not in encoder.categories_[0]
    assert matrix[:, pipeline.named_steps["columns"].output_indices_["categories"].start].nnz == 0


def test_speed_missing_indicator_and_all_missing_fit_failure(codebook, records):
    builder = ContextFeatures(codebook)
    result = builder.fit_transform(records)
    assert result["speed_limit_missing"].tolist() == [0, 0, 1, 0]
    records["speed_limit"] = [0, 99, -1, 180]
    result = builder.transform(records)
    assert result["speed_limit"].isna().all()
    assert result["speed_limit_missing"].eq(1).all()
    with pytest.raises(ValueError, match="no valid speed_limit"):
        make_preprocessor(codebook).fit(records)


def test_pipeline_clone_and_serialization_preserve_input_output_contract(codebook, records, tmp_path):
    pipeline = clone(make_preprocessor(codebook)).fit(records)
    path = tmp_path / "preprocessor.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    assert (restored.transform(records) - pipeline.transform(records)).nnz == 0
    assert list(restored.get_feature_names_out()) == list(pipeline.get_feature_names_out())
    model_pipeline = clone(make_model_pipeline(DummyClassifier(), codebook))
    assert set(model_pipeline.named_steps) == {"preprocessing", "model"}
    assert not hasattr(model_pipeline.named_steps["model"], "classes_")


def test_target_encoding_is_fixed_not_learned():
    np.testing.assert_array_equal(encode_target(pd.Series([3, 1, 2, 3], name="collision_severity")), [2, 0, 1, 2])
    assert encode_target(pd.Series([1])).dtype == np.int64


@pytest.mark.parametrize("values", [[0], [4], [np.nan], [1.5], ["severe"]])
def test_invalid_targets_are_not_silently_dropped(values):
    with pytest.raises(ValueError):
        encode_target(pd.Series(values, name="collision_severity"))


def year_records(records):
    parts = []
    for year in (2019, 2020, 2021):
        part = records.copy()
        part["date"] = part["date"].str.replace("2019", str(year))
        part["collision_year"] = year
        part["collision_severity"] = 3
        part["collision_index"] = [f"{year}-{i}" for i in range(len(part))]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def test_temporal_split_is_disjoint_sorted_and_reproducible(records):
    frame = year_records(records)
    result = temporal_split(frame.sample(frac=1, random_state=42))
    again = temporal_split(frame.sample(frac=1, random_state=17))
    for name, part in result.items():
        pd.testing.assert_frame_equal(part, again[name])
        assert recorded_timestamp(part).is_monotonic_increasing
    keys = [set(part["collision_index"]) for part in result.values()]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert sum(map(len, result.values())) == len(frame)


@pytest.mark.parametrize("problem", ["duplicate", "missing_year", "wrong_year", "year_mismatch", "null_key"])
def test_invalid_temporal_split_raises(records, problem):
    frame = year_records(records)
    if problem == "duplicate":
        frame.loc[1, "collision_index"] = frame.loc[0, "collision_index"]
    elif problem == "missing_year":
        frame = frame[frame["collision_year"].ne(2021)]
    elif problem == "wrong_year":
        frame["collision_year"] = 2022
        frame["date"] = "01/01/2022"
    elif problem == "year_mismatch":
        frame.loc[0, "collision_year"] = 2020
    else:
        frame.loc[0, "collision_index"] = None
    with pytest.raises(ValueError):
        temporal_split(frame)


def test_quarter_folds_refit_preprocessing_without_future_categories(codebook, records):
    records["weather_conditions"] = [1, 1, 2, 2]
    folds = quarterly_folds(records)
    assert [(len(a), len(b)) for a, b in folds] == [(1, 1), (2, 1), (3, 1)]
    for fit, valid in folds:
        assert not set(fit) & set(valid)
        assert recorded_timestamp(records.iloc[fit]).max() < recorded_timestamp(records.iloc[valid]).min()
        pipeline = make_preprocessor(codebook).fit(records.iloc[fit])
        state = joblib.hash(pipeline)
        pipeline.transform(records.iloc[valid])
        assert joblib.hash(pipeline) == state
    pipeline = make_preprocessor(codebook).fit(records.iloc[folds[1][0]])
    encoder = pipeline.named_steps["columns"].named_transformers_["categories"]
    assert "weather_conditions_two" not in encoder.categories_[0]


def test_quarter_folds_reject_unsorted_future_and_empty_periods(records):
    for bad in [records.iloc[::-1], records.assign(date="01/01/2020"), records.iloc[:2]]:
        with pytest.raises(ValueError):
            quarterly_folds(bad)
