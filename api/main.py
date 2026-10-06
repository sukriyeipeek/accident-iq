from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.load_stats19 import read_codebook
from src.evaluation.explainability import local_sensitivity
from src.features.preprocessing import CATEGORICAL_FEATURES, RAW_FEATURES, TARGET_LABELS


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("ACCIDENTIQ_MODEL_PATH", ROOT / "models/release_v1/selected.joblib"))
REFERENCE_PATH = Path(os.getenv("ACCIDENTIQ_REFERENCE_PATH", ROOT / "config/explanation_reference.json"))
GUIDE_PATH = ROOT / "config/stats19_data_guide_2025.xlsx"


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str = Field(description="Collision date, DD/MM/YYYY or YYYY-MM-DD")
    time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$", description="Local time, HH:MM")
    speed_limit: int | None = Field(default=None, description="Posted speed limit in mph")
    weather_conditions: int
    road_surface_conditions: int
    light_conditions: int
    road_type: int
    junction_detail_historic: int
    junction_control: int
    urban_or_rural_area: int
    first_road_class: int

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
            datetime.strptime(value, "%d/%m/%Y")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            datetime.strptime(value, "%Y-%m-%d")
        else:
            raise ValueError("Use DD/MM/YYYY or YYYY-MM-DD")
        return value

    @field_validator("speed_limit")
    @classmethod
    def validate_speed_limit(cls, value):
        if value is not None and value not in {20, 30, 40, 50, 60, 70}:
            raise ValueError("Use a posted speed limit of 20, 30, 40, 50, 60 or 70 mph")
        return value


class FactorOutput(BaseModel):
    feature: str
    value: str
    reference: str
    predicted_class_probability_change_pp: float


class PredictionOutput(BaseModel):
    prediction_target: str
    predicted_severity: str
    class_scores: dict[str, float]
    max_class_score: float
    score_calibration: str
    important_factors: list[FactorOutput]
    explanation_method: str
    explanation_caveat: str


def create_app(*, model_path=MODEL_PATH, reference_path=REFERENCE_PATH, codebook=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model = joblib.load(model_path)
        application.state.reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        application.state.codebook = codebook if codebook is not None else read_codebook(GUIDE_PATH)
        application.state.categories = CATEGORICAL_FEATURES
        yield

    application = FastAPI(title="AccidentIQ API", version="1.0.0", lifespan=lifespan,
                          description="Severity classification conditional on a police-reported injury collision.")
    allowed_origins = [origin.strip() for origin in os.getenv(
        "ACCIDENTIQ_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
                               allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    application.add_api_route("/health", health, methods=["GET"])
    application.add_api_route("/options", options, methods=["GET"])
    application.add_api_route("/predict-severity", predict_severity, methods=["POST"],
                              response_model=PredictionOutput)
    application.add_api_route("/predict", predict_alias, methods=["POST"],
                              response_model=PredictionOutput, include_in_schema=False)
    return application


def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model"),
            "target": "severity conditional on a reported injury collision"}


def options(request: Request):
    codebook = request.app.state.codebook
    categories = {feature: [{"value": int(code), "label": label}
                            for code, label in sorted(codebook[feature].items())]
                  for feature in request.app.state.categories}
    return {"categories": categories, "speed_limit_mph": [20, 30, 40, 50, 60, 70],
            "prediction_scope": "collision severity conditional on a reported injury collision"}


def _predict(payload: PredictionInput, request: Request) -> PredictionOutput:
    data = payload.model_dump()
    for feature in CATEGORICAL_FEATURES:
        if data[feature] not in request.app.state.codebook[feature]:
            raise HTTPException(status_code=422, detail=f"Unknown official code for {feature}")
    row = pd.DataFrame([{feature: data[feature] for feature in RAW_FEATURES}], columns=RAW_FEATURES)
    model = request.app.state.model
    probabilities = model.predict_proba(row)[0]
    target = int(probabilities.argmax())
    explanation = local_sensitivity(model, row, request.app.state.reference, request.app.state.codebook)
    factors = [FactorOutput(**item) for item in explanation["factors"]]
    return PredictionOutput(
        prediction_target="severity | police-reported personal-injury collision",
        predicted_severity=TARGET_LABELS[target],
        class_scores={TARGET_LABELS[int(label)]: float(probabilities[int(label)]) for label in model.classes_},
        max_class_score=float(probabilities[target]),
        score_calibration="Uncalibrated model score; do not interpret as accident risk or calibrated confidence.",
        important_factors=factors,
        explanation_method=explanation["method"],
        explanation_caveat=explanation["interpretation"],
    )


def predict_severity(payload: PredictionInput, request: Request):
    return _predict(payload, request)


def predict_alias(payload: PredictionInput, request: Request):
    return _predict(payload, request)


app = create_app()
