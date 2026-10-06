from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocessing import CATEGORICAL_FEATURES, RAW_FEATURES


def build_reference_row(training: pd.DataFrame) -> dict:
    reference = {}
    for feature in RAW_FEATURES:
        values = training[feature].dropna()
        if values.empty:
            raise ValueError(f"No training reference value for {feature}")
        if feature == "date":
            reference[feature] = "01/07/2019"
        elif feature == "time":
            reference[feature] = "12:00"
        else:
            reference[feature] = values.mode().iloc[0].item() if hasattr(values.mode().iloc[0], "item") else values.mode().iloc[0]
    if reference["junction_detail_historic"] == 0:
        reference["junction_control"] = 0
    elif reference["junction_control"] == 0:
        reference["junction_control"] = next(iter(training["junction_control"].dropna().mode()))
    return reference


def _display_value(feature: str, value, codebook: dict) -> str:
    if pd.isna(value):
        return "missing"
    if feature in CATEGORICAL_FEATURES:
        try:
            code = int(value)
            return codebook[feature].get(code, f"Unrecognized code {code}")
        except (TypeError, ValueError):
            return str(value)
    if feature == "speed_limit":
        return f"{int(value)} mph posted speed limit"
    return str(value)


def local_sensitivity(model, row: pd.DataFrame, reference: dict, codebook: dict, *, top_k: int = 4) -> dict:
    if len(row) != 1:
        raise ValueError("Local explanation requires exactly one raw input row")
    probabilities = np.asarray(model.predict_proba(row), dtype=float)
    if probabilities.shape != (1, 3) or not np.isfinite(probabilities).all():
        raise ValueError("Model returned invalid class probabilities")
    predicted = int(probabilities[0].argmax())
    factors = []
    for feature in RAW_FEATURES:
        current = row.iloc[0][feature]
        baseline = reference[feature]
        changed = row.copy()
        changed.loc[:, feature] = baseline
        comparison = np.asarray(model.predict_proba(changed), dtype=float)
        delta = float((probabilities[0, predicted] - comparison[0, predicted]) * 100)
        factors.append({"feature": feature, "value": _display_value(feature, current, codebook),
                        "reference": _display_value(feature, baseline, codebook),
                        "predicted_class_probability_change_pp": delta})
    factors.sort(key=lambda item: (-abs(item["predicted_class_probability_change_pp"]), item["feature"]))
    return {"method": "single-feature replacement sensitivity",
            "explained_class": ("Fatal", "Serious", "Slight")[predicted],
            "reference": reference, "factors": factors[:top_k],
            "interpretation": "Each factor changes one input to a fixed training reference while holding other inputs constant. Changes describe model sensitivity, not causality."}
