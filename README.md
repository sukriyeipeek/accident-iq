# AccidentIQ

Traffic Accident Risk & Severity Intelligence — a staged ML/DL portfolio project.

**Current implementation: Phases 1–7 and 10–14 complete.** Phases 8–9 were assessed and closed without image/fusion models because no trustworthy paired images exist. The project includes a reproducible, script-only tabular ML/DL pipeline, frozen model, explanation report, FastAPI service, static frontend and tests. No Jupyter runtime or notebook files are used or required.

## Problem and scope

The initial classification target is `collision_severity`: Fatal, Serious or Slight.

Interpretation: **P(severity | police-reported injury collision, X)**. STATS19 contains reported personal-injury collisions, not representative non-collision road/time observations. Accident occurrence risk is therefore **not estimable from these records alone**.

The model uses road, weather, light and time context. Casualty counts, adjusted/enhanced severity and post-outcome reporting variables are excluded. Accident occurrence risk cannot be estimated from this collision-only dataset.

## Dataset

Source: [Department for Transport Road Safety Open Data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data). Acquisition and filenames: [data/README.md](data/README.md).

| Year | Collision | Vehicle | Casualty | Planned role |
| --- | ---: | ---: | ---: | --- |
| 2019 | 117,536 | 216,381 | 153,158 | Train |
| 2020 | 91,199 | 167,375 | 115,584 | Validation |
| 2021 | 101,087 | 186,443 | 128,209 | Test |

All 309,822 collision records and 44 raw collision columns are retained for auditing. Raw files are not committed to GitHub. [Data manifest](reports/eda/data_manifest.json) records SHA-256 fingerprints of the exact inputs and versions used.

The official cumulative files were used to extract 2019–2020; 2021 was obtained as yearly files. The cumulative source can be revised, so a future download may not match this snapshot.

## Installation and reproducible EDA

Python 3.11.9 was used for Phase 2. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
.\.venv\Scripts\python.exe -m scripts.download_stats19 --use-cumulative-archive
.\.venv\Scripts\python.exe -m scripts.run_eda
.\.venv\Scripts\python.exe -m scripts.run_eda
.\.venv\Scripts\python.exe -m pytest -q
```

The cumulative download requires several GB of disk space. Downloaded content is written to `.part` first; incomplete transfers are not published as final files. Existing download sizes are checked against server headers, and existing yearly extractions against archive row counts. This does not replace content/provenance checking via the manifest.

The codebook is checked in under `config/stats19_data_guide_2025.xlsx`; its download URL is recorded in the manifest. `requirements-eda.txt` pins the verified Phase 2 environment; `requirements.txt` contains broader development dependencies for later phases.

## Phase 2: EDA and findings

- [EDA findings](reports/eda/phase2_findings.md)
- Re-run with `python -m scripts.run_eda`; figures and aggregate tables are written under `reports/eda/`.
- [Detailed findings and learning notes](reports/eda/phase2_findings.md)
- [Per-field prediction-time decisions](reports/eda/feature_decisions.csv)
- [Quality checks](reports/eda/quality_checks.json)
- 19 current figures in `reports/eda/figures/`, with aggregate CSV tables in `reports/eda/tables/`

The EDA covers null and coded missing/unknown values, structural missingness, duplicates, composite keys, orphan records, casualty/vehicle counts, target consistency, imbalance, numeric distributions and IQR flags, categorical distributions, geographic density, temporal patterns, Spearman correlation and bias-corrected Cramer's V.

Detailed feature–severity analyses use **2019 only**. All years undergo quality checks. Yearly label distributions were already inspected in earlier work, including 2021; the test set is **not pristine**. Later evaluation must disclose this, and a fresh holdout is needed for a genuinely blind performance claim.

2019 training classes: Fatal 1,658; Serious 22,980; Slight 92,898. The majority share is 79.04%, not a trained model score. In reported injury collisions, the Serious+Fatal share is 20.00% in daylight and 31.83% in darkness without lighting. These are **conditional descriptive proportions**, not occurrence probabilities or causal effects.

Junction control missingness is often structural outside junctions. Category code 0 is not globally missing. Posted speed is in mph, not measured vehicle speed. Extreme values are flagged, not automatically removed. No imputer, encoder or scaler was fitted during Phase 2; fitting is implemented separately in Phase 3.

Verification: 27 tests passed for code semantics, time features, denominators, intervals, associations, table integrity, file fingerprints and partial-download handling. Unit tests do not train models.

## Phase 3: preprocessing and feature engineering

- [Preprocessing verification](reports/preprocessing/verification.json)
- [Transformation contract and learning notes](reports/preprocessing/phase3_findings.md)
- [Split, fold and artifact checks](reports/preprocessing/summary.json)
- [Reusable preprocessing code](src/features/preprocessing.py)

Install and run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-preprocessing.txt
.\.venv\Scripts\python.exe -m scripts.prepare_features
.\.venv\Scripts\python.exe -m scripts.verify_preprocessing
.\.venv\Scripts\python.exe -m pytest -q
```

Eleven raw context fields become 63 CSR float32 features in the full 2019-fitted snapshot. Raw and ISO dates are accepted; valid HH:MM local time is required. Hour (including minutes), weekday and month use cyclic sin/cos encoding. Season is not duplicated as a separate input.

Categorical fields use the official dictionary and one-hot encoding. Missing, unknown and not-applicable junction control are distinct. Speed limits outside 20/30/40/50/60/70 mph are flagged missing, imputed with the training median and standardized with training statistics; a missing indicator is retained. Raw records are preserved. Held-out categories do not widen the training vocabulary; unseen values produce an all-zero block and are reported as schema diagnostics.

Only 2019 fits the pipeline. 2020 and 2021 are transformed without refitting. Processed matrices, fixed target arrays and row-order sidecars are stored under ignored `data/processed/context_v1/`. The ignored `models/preprocessing_context_v1.joblib` contains a **preprocessor, not a classifier**. Load only trusted joblib artifacts using matching dependency versions.

Three expanding-window CV folds use 2019 quarters: Q1 -> Q2, Q1–Q2 -> Q3 and Q1–Q3 -> Q4. Each fold has its own fitted preprocessing. Phase 4 fits `make_model_pipeline` on **raw input rows**, not a full-train-transformed cached matrix. Fold vocabularies and widths may differ; that is expected.

Verification at the end of Phase 3: 62 tests passed. Full-data checks confirm input hashes, finite matrices, matching dimensions, unchanged fit state during transform and serialization round-trip. There were no Phase 3 model scores.

## Phase 4: ML baseline

Phase 4 adds a reproducible comparison of a prior-probability Dummy classifier, standard multiclass Logistic Regression and class-weighted Logistic Regression. All three use the same raw-input preprocessing pipeline. The three 2019 expanding-window quarter folds refit preprocessing independently; each model is then fit on all 2019 records and evaluated on 2020. Hyperparameters are fixed in [the baseline protocol](config/baseline_protocol.json); there is no grid search or sampling.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_baselines
```

The 2021 collision files are not loaded as training or evaluation records. Their source fingerprints are checked against the Phase 2 manifest. The test year is not pristine: its label distribution was inspected during EDA, so final claims must disclose that limitation.

Results, per-class metrics, confusion matrices, precision-recall plots, runtime measurements and learning notes are in [the Phase 4 report](reports/baselines/phase4_findings.md). Reproduce with `python -m scripts.train_baselines`. Classifier pipelines are saved under ignored `models/baseline_v1/`; validation predictions and row keys are under ignored `data/processed/baseline_v1/`. The selected pipeline is chosen solely by mean 2019 quarterly CV macro F1. A selected baseline is a research reference, not a calibrated occurrence-risk model or a deployment approval.

Occurrence risk remains unidentifiable from this injury-collision-only dataset. The Phase 4 target is severity conditional on a police-reported personal-injury collision. Class weighting changes the precision/recall and probability trade-off; do not read its raw probabilities as calibrated confidence without a separate calibration study.

Verification at Phase 4: all model pipelines were saved and reloaded with matching raw-input probabilities; seven report figures were generated. The 2021 test set was not scored.

## Phase 5: controlled advanced ML

Added a class-weighted Random Forest and two LightGBM variants (unweighted and class-weighted), all with fixed settings and the same fold-local preprocessing. Random Forest is capped at 120 trees, depth 18, leaf size 15 and 70% bootstrap samples to control training cost. LightGBM is deterministic and single-threaded. No sampling or grid search was used. Install `requirements-advanced.txt` to reproduce these models.

| Model | 2019 CV macro F1 | 2020 macro F1 | Accuracy | Fatal recall | Serious recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced Random Forest | **0.3701** | **0.3667** | 0.6000 | 0.4206 | 0.2713 |
| LightGBM | 0.2945 | 0.2929 | 0.7835 | 0.0000 | 0.0001 |
| Balanced LightGBM | 0.3509 | 0.3451 | 0.5287 | **0.4881** | **0.3240** |

Advanced-model selection uses only the 2019 quarterly CV mean macro F1. The class-weighted models detect more rare severe outcomes but generate more false positives; their probabilities are not treated as calibrated confidence. Full class metrics, runtime, parameters and fold scores are in [the Phase 5 report](reports/advanced/phase5_findings.md). Reproduce with `python -m scripts.train_advanced`; the saved advanced pipeline is under ignored `models/advanced_v1/`.

The Random Forest fit took about 129 seconds on the full 2019 training set in this environment; timings depend on hardware. LightGBM uses the same 63 encoded context features. The whole-project model was selected after Phase 6–7 comparison; see below.

## Phase 6: PyTorch tabular neural network

A compact 63-input MLP (128→64→3, ReLU, dropout) uses weighted cross-entropy and Adam. Each chronological outer fold has an earlier inner split for early stopping; its outer validation quarter does not select the epoch. The final 2019 model uses the median selected epoch count.

| 2020 validation | Result |
| --- | ---: |
| Macro F1 | 0.3337 |
| Accuracy | 0.5547 |
| Fatal precision / recall | 0.0390 / 0.5945 |
| Serious precision / recall | 0.2358 / 0.1734 |

The MLP did not outperform the selected tree model on macro F1. Training/validation history and learning notes are in [the Phase 6 report](reports/deep_learning/phase6_findings.md). Install `requirements-deep-learning.txt`, then run `python -m scripts.train_tabular_nn` to reproduce it.

## Phase 7: comparison and final evaluation

Seven candidates were compared using the same mean 2019 quarterly CV macro F1 rule. The selected model is the class-weighted Random Forest. Its artifact is `models/release_v1/selected.joblib` (generated locally and not committed); its tracked decision record is in the [Phase 7 report](reports/model_comparison/phase7_findings.md) and machine-readable [comparison summary](reports/model_comparison/summary.json).

| Candidate | 2019 CV macro F1 | 2020 macro F1 | Fatal recall | Serious recall |
| --- | ---: | ---: | ---: | ---: |
| Balanced Random Forest | **0.3701** | **0.3667** | 0.4206 | 0.2713 |
| Balanced LightGBM | 0.3509 | 0.3451 | 0.4881 | 0.3240 |
| Balanced Logistic Regression | 0.3244 | 0.3193 | 0.6175 | 0.2758 |
| PyTorch MLP | 0.3202 | 0.3337 | 0.5945 | 0.1734 |

Inference comparison, artifact size, all CV/validation metrics and the final score are in [the Phase 7 report](reports/model_comparison/phase7_findings.md). 2021 selected-model metrics were accuracy 0.6024 and macro F1 0.3679. The result is exploratory: 2021 labels were inspected during EDA, and a report serialization retry repeated the same frozen model evaluation twice. No model selection or tuning used those scores. A genuinely blind estimate requires a new holdout.

## Phases 8–9: image and multimodal assessment

No image model or fusion model was trained. DfT STATS19 has no linked scene imagery or shared image-event key. The reviewed [AI City staged-video data](https://www.aicitychallenge.org/2024-data-and-evaluation/) and [Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) do not share STATS19 collision events or injury-severity labels. Details and the conditions for reconsidering this decision are in [the Phase 8–9 assessment](reports/multimodal/phase8_9_assessment.md). Independent datasets were not artificially joined.

## Phase 10: explainability

The selected model is explained with raw-input permutation importance on a stratified 2020 sample and local one-feature-at-a-time reference sensitivity. Average global macro-F1 drops were small and variable, so no factor is claimed as a robust global driver. Per-request factors describe model sensitivity, not cause; see [the explanation report](reports/explainability/phase10_findings.md). Model scores are not calibrated confidence.

## Phases 11–12: API and frontend

The FastAPI service loads the selected artifact once at application startup:

- `GET /health` — service/model status.
- `GET /options` — category labels from the official codebook.
- `POST /predict-severity` — severity scores and local sensitivity factors.
- `POST /predict` — compatibility alias for severity prediction.

The static responsive interface is in `frontend/`. It collects road/time context and displays class scores, predicted severity and the local sensitivity factors. There is deliberately no “accident risk %” output: these data cannot support an occurrence-risk target. Weighted-model scores are clearly marked uncalibrated.

Run locally after completing training and explanation steps:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-serving.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 --env-file .env
```

In a second terminal:

```powershell
.\.venv\Scripts\python.exe -m http.server 5173 --directory frontend
```

Open `http://localhost:5173`. The frontend's API address is configured in [frontend/config.js](frontend/config.js). For the full artifact generation sequence, first run the data, EDA and preprocessing commands, then `scripts.train_baselines`, `scripts.train_advanced`, `scripts.train_tabular_nn`, `scripts.compare_models` and `scripts.run_explainability`.

With those artifacts present, build and start the API container from the repository root:

```powershell
docker compose up --build
```

The API is available at `http://localhost:8000`; serve the static frontend separately with the command above. The container packages only the selected model and serving code, not raw data or training dependencies.

## Phase 13: testing

`python -m pytest -q` currently reports **78 passed**. Tests cover data/preprocessing, metrics and probability contracts, model component shapes, explainability, API health/options/input validation/prediction and artifact-backed behavior without retraining models inside unit tests.

## Phase 14: deployment and final notes

The API can run locally or in the included container configuration after model artifacts are generated. Raw data, processed matrices, virtual environments and model weights remain ignored; reproduce artifacts from the documented sources. Keep Python/scikit-learn/joblib versions aligned when loading trusted `joblib` models. Container wiring is in [Dockerfile.api](Dockerfile.api) and [docker-compose.yml](docker-compose.yml).

## Architecture

```text
Official CSVs + codebook
  -> audited data + training-only EDA
  -> fold-fitted preprocessing / feature engineering
  -> Dummy / Logistic Regression / Random Forest / LightGBM / PyTorch MLP
  -> 2019 chronological CV -> frozen candidate -> 2020 validation -> exploratory 2021 score
  -> permutation + local sensitivity explanation
  -> saved sklearn pipeline -> FastAPI -> static frontend
```

Temporal partitions are 2019 train, 2020 validation, and 2021 exploratory holdout. Three chronological within-train folds are implemented; each training fold fits its own preprocessing. 2021 target distribution was previously inspected, and the selected model evaluation was repeated after a serialization retry; see Phase 7 report. Geographic robustness will be needed before adding raw coordinates.

Class weighting was compared without sampling. Results are in `reports/baselines/`, `reports/advanced/`, `reports/deep_learning/` and `reports/model_comparison/`; compare class-specific recall/precision and probability-quality metrics alongside macro F1.

## Phase status

Phases 1–7 and 10–13 are implemented. Phases 8–9 are complete as a documented dataset suitability decision: paired imagery is unavailable, so image/fusion training is not scientifically supported. Phase 14 provides the README, dependency sets, Docker packaging and final repository cleanup.

## Repository structure

```text
config/         official data dictionary
data/           acquisition documentation; ignored raw/processed data
scripts/        acquisition, EDA, preprocessing, training, comparison and explanations (all runnable as Python modules)
src/data/       loading, profiling, integrity checks, report generation
src/features/   allowlisted context features and sklearn preprocessing
src/models/     baseline, advanced and tabular neural classifiers
src/evaluation/ classification metrics, reports and explanations
api/            FastAPI service and request schemas
frontend/       responsive static browser interface
reports/eda/    EDA findings, aggregate tables, figures, manifest
reports/preprocessing/  transformation contract and verification summaries
reports/baselines/      baseline results, figures and learning report
reports/advanced/       Random Forest / LightGBM results
reports/deep_learning/  MLP history and class metrics
reports/model_comparison/ frozen selection and final comparison
reports/explainability/ permutation and local sensitivity outputs
reports/multimodal/     image/fusion data suitability decision
tests/          data, preprocessing, model components, explanation and API tests
```

Other directories will be populated only when their phase needs real code.

## Limitations and future improvements

- Police-reported injury collisions are a selected population; reporting practices affect labels.
- Injury-based recording changes and the 2020 traffic change can produce distribution shift.
- Numeric visibility and traffic exposure are absent; light conditions are not visibility in metres.
- Wilson intervals are descriptive and assume independent observations; spatial/reporting dependence is not corrected.
- Environmental scenario inputs may still be recorded after the collision; deployment input availability must be verified.
- Occurrence risk needs road/time units, exposure and defensible non-collision observations.
- Genuine tabular-image matching, fresh temporal holdout and spatial evaluation are future work.

## License and attribution

Retain Department for Transport attribution and review the source's reuse terms before redistributing data. No raw records or secrets are committed.
