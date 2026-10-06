from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocessing import recorded_timestamp


SPLIT_YEARS = {"train": 2019, "validation": 2020, "test": 2021}


def temporal_split(frame: pd.DataFrame, split_years: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    roles = SPLIT_YEARS if split_years is None else split_years
    if not roles or len(set(roles.values())) != len(roles):
        raise ValueError("Split years must be nonempty and unique")
    required = {"collision_index", "collision_year", "collision_severity", "date", "time"}
    if required - set(frame.columns):
        raise ValueError(f"Missing split columns: {sorted(required - set(frame.columns))}")
    if frame["collision_index"].isna().any() or frame["collision_index"].duplicated().any():
        raise ValueError("Collision keys must be non-null and unique before splitting")
    timestamp = recorded_timestamp(frame)
    if timestamp.dt.year.ne(frame["collision_year"]).any():
        raise ValueError("Recorded date/year mismatch")
    if not frame["collision_year"].isin(roles.values()).all():
        raise ValueError(f"Unexpected years; expected {sorted(roles.values())}")
    ordered = frame.assign(_timestamp=timestamp).sort_values(["_timestamp", "collision_index"], kind="stable")
    result = {}
    for role, year in roles.items():
        part = ordered.loc[ordered["collision_year"].eq(year)].drop(columns="_timestamp").reset_index(drop=True)
        if part.empty:
            raise ValueError(f"Missing {role} year: {year}")
        result[role] = part
    return result


def quarterly_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    timestamp = recorded_timestamp(train)
    if not timestamp.dt.year.eq(2019).all():
        raise ValueError("CV accepts the 2019 training partition only")
    if not timestamp.is_monotonic_increasing:
        raise ValueError("CV input must be chronologically sorted")
    quarter = timestamp.dt.quarter
    folds = []
    for next_quarter in (2, 3, 4):
        fit = np.flatnonzero(quarter.lt(next_quarter))
        valid = np.flatnonzero(quarter.eq(next_quarter))
        if len(fit) == 0 or len(valid) == 0:
            raise ValueError(f"Empty quarter fold: Q{next_quarter}")
        if timestamp.iloc[fit].max() >= timestamp.iloc[valid].min():
            raise ValueError("CV training time must precede validation time")
        folds.append((fit, valid))
    return folds
