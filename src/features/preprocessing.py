from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


CATEGORICAL_FEATURES = (
    "weather_conditions", "road_surface_conditions", "light_conditions", "road_type",
    "junction_detail_historic", "junction_control", "urban_or_rural_area", "first_road_class",
)
RAW_FEATURES = ("date", "time", "speed_limit", *CATEGORICAL_FEATURES)
CYCLIC_FEATURES = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos")
ENGINEERED_FEATURES = (*CATEGORICAL_FEATURES, "speed_limit", *CYCLIC_FEATURES, "speed_limit_missing")
TARGET_CODES = {1: 0, 2: 1, 3: 2}
TARGET_LABELS = {0: "Fatal", 1: "Serious", 2: "Slight"}
MISSING, UNKNOWN, UNRECOGNIZED, NOT_APPLICABLE = "__missing__", "__unknown__", "__unrecognized__", "__not_applicable__"
VALID_SPEEDS = (20, 30, 40, 50, 60, 70)


def numeric_codes(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    malformed = (series.notna() & values.isna()) | (values.notna() & (~np.isfinite(values) | values.mod(1).ne(0)))
    if malformed.any() or series.map(lambda v: isinstance(v, (bool, np.bool_))).any():
        raise ValueError(f"{series.name}: expected integer codes or null; invalid input")
    return values


def recorded_timestamp(frame: pd.DataFrame) -> pd.Series:
    date = frame["date"].astype("string")
    if pd.api.types.is_datetime64_any_dtype(frame["date"]):
        if frame["date"].dt.tz is not None:
            raise ValueError("date must be timezone-naive recorded local date")
        dates = frame["date"].dt.normalize()
    else:
        slash = date.str.fullmatch(r"\d{2}/\d{2}/\d{4}", na=False)
        iso = date.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        dates = pd.to_datetime(date.where(slash), format="%d/%m/%Y", errors="coerce")
        dates = dates.fillna(pd.to_datetime(date.where(iso), format="%Y-%m-%d", errors="coerce"))
    time = frame["time"].astype("string")
    valid = time.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", na=False)
    if dates.isna().any() or not valid.all():
        raise ValueError("date/time: valid DD/MM/YYYY or YYYY-MM-DD date and HH:MM local time required")
    minutes = time.str[:2].astype(int) * 60 + time.str[3:].astype(int)
    return dates + pd.to_timedelta(minutes, unit="m")


class ContextFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, codebook: dict):
        self.codebook = codebook

    def fit(self, X: pd.DataFrame, y=None):
        features = self._build(X)
        if features["speed_limit"].notna().sum() == 0:
            raise ValueError("Training data has no valid speed_limit; cannot learn a median")
        self.n_features_in_ = len(RAW_FEATURES)
        self.feature_names_in_ = np.asarray(RAW_FEATURES, dtype=object)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_")
        return self._build(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "fitted_")
        return np.asarray(ENGINEERED_FEATURES, dtype=object)

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("Nonempty pandas DataFrame required")
        if X.columns.duplicated().any():
            raise ValueError("Duplicate input column names")
        absent = set(RAW_FEATURES) - set(X.columns)
        if absent:
            raise ValueError(f"Missing required input columns: {sorted(absent)}")
        timestamp = recorded_timestamp(X)
        result = pd.DataFrame(index=X.index)
        raw_codes = {}
        for column in CATEGORICAL_FEATURES:
            if column not in self.codebook:
                raise ValueError(f"Codebook missing definition: {column}")
            codes = numeric_codes(X[column])
            raw_codes[column] = codes
            mapping = self.codebook[column]
            labels = codes.map(mapping).fillna(UNRECOGNIZED)
            missing = [k for k, v in mapping.items() if "data missing" in v.lower()]
            unknown = [k for k, v in mapping.items() if v.lower().startswith("unknown") or v.lower() == "unallocated"]
            labels.loc[codes.isna() | codes.isin(missing)] = MISSING
            labels.loc[codes.isin(unknown)] = UNKNOWN
            result[column] = labels
        outside = raw_codes["junction_detail_historic"].eq(0)
        result.loc[outside | raw_codes["junction_control"].eq(0), "junction_control"] = NOT_APPLICABLE
        speed = numeric_codes(X["speed_limit"])
        result["speed_limit"] = speed.where(speed.isin(VALID_SPEEDS)).astype(np.float32)
        result["speed_limit_missing"] = result["speed_limit"].isna().astype(np.float32)
        cycles = {
            "hour": ((timestamp.dt.hour + timestamp.dt.minute / 60), 24),
            "weekday": (timestamp.dt.dayofweek, 7),
            "month": (timestamp.dt.month - 1, 12),
        }
        for name, (values, period) in cycles.items():
            angle = 2 * np.pi * values / period
            result[f"{name}_sin"] = np.sin(angle).astype(np.float32)
            result[f"{name}_cos"] = np.cos(angle).astype(np.float32)
        return result.loc[:, ENGINEERED_FEATURES]


def to_csr_float32(matrix):
    return sparse.csr_matrix(matrix, dtype=np.float32)


def make_preprocessor(codebook: dict) -> Pipeline:
    speed = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    columns = ColumnTransformer([
        ("speed", speed, ["speed_limit"]),
        ("cycles", "passthrough", list(CYCLIC_FEATURES)),
        ("missing", "passthrough", ["speed_limit_missing"]),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), list(CATEGORICAL_FEATURES)),
    ], remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=True)
    return Pipeline([
        ("context", ContextFeatures(codebook)),
        ("columns", columns),
        ("csr", FunctionTransformer(to_csr_float32, feature_names_out="one-to-one")),
    ])


def make_model_pipeline(estimator, codebook: dict) -> Pipeline:
    return Pipeline([("preprocessing", make_preprocessor(codebook)), ("model", estimator)])


def encode_target(series: pd.Series) -> np.ndarray:
    codes = numeric_codes(series)
    if not codes.isin(TARGET_CODES).all():
        raise ValueError("collision_severity: only 1/Fatal, 2/Serious, 3/Slight allowed")
    return codes.map(TARGET_CODES).to_numpy(dtype=np.int64)
