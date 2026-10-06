# Data acquisition

AccidentIQ v1 uses official Great Britain STATS19 collision, vehicle and casualty records for 2019–2021.

Source: [Department for Transport Road Safety Open Data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data).

## Acquired snapshot

| Year | Collision rows | Vehicle rows | Casualty rows | Acquisition |
| --- | ---: | ---: | ---: | --- |
| 2019 | 117,536 | 216,381 | 153,158 | Extracted from official cumulative CSVs |
| 2020 | 91,199 | 167,375 | 115,584 | Extracted from official cumulative CSVs |
| 2021 | 101,087 | 186,443 | 128,209 | Official yearly CSVs |

Run from the repository root after installing `requirements-eda.txt`:

```powershell
python -m scripts.download_stats19 --use-cumulative-archive
python -m scripts.run_eda
```

The official cumulative archives cover 1979–latest and require several GB. Individual historical endpoints were unavailable during acquisition; no unofficial mirror was substituted. Future publication updates can revise the cumulative records.

Files belong under `data/raw/stats19/`:

```text
dft-road-casualty-statistics-collision-2019.csv
dft-road-casualty-statistics-collision-2020.csv
dft-road-casualty-statistics-collision-2021.csv
dft-road-casualty-statistics-vehicle-2019.csv
dft-road-casualty-statistics-vehicle-2020.csv
dft-road-casualty-statistics-vehicle-2021.csv
dft-road-casualty-statistics-casualty-2019.csv
dft-road-casualty-statistics-casualty-2020.csv
dft-road-casualty-statistics-casualty-2021.csv
archive/  cumulative source files
```

Downloads and new extractions are promoted from `.part` only after completion. Existing yearly extractions are compared to source schema/year/row count. This is a completeness check, not a publisher-authenticated checksum. Existing mismatched files are preserved and cause an error.

## Dictionary and integrity

The official codebook is `config/stats19_data_guide_2025.xlsx`. Its URL, raw-file hashes and environment versions are in [data_manifest.json](../reports/eda/data_manifest.json).

Current cumulative schema uses converted newer fields alongside historical fields. For 2019–2021 junction information, retain `junction_detail_historic` and decode it with the official dictionary; do not assume nominal codes are numeric measurements.

Phase 2 preserves all raw columns and records, checks table keys, counts, schemas, dates and severity consistency, and records null/coded-missing/unknown separately. Full integrity results: [quality_checks.json](../reports/eda/quality_checks.json).

No raw or processed records are committed. Aggregate reports, reproducible Python scripts and their figures are the portfolio artifacts.

## Phase 3 processed artifacts

Run `python -m scripts.prepare_features` after installing `requirements-preprocessing.txt`.

`data/processed/context_v1/` contains `X_train.npz`, `X_validation.npz`, `X_test.npz` (CSR float32),
matching `y_*.npy` (0 Fatal / 1 Serious / 2 Slight), and `rows_*.csv` recording collision keys and chronological row positions.
Do not change the row order independently of X/y. Sidecar keys/timestamps are not model features.

The preprocessor fits only 2019. 2020/2021 use transform only; input hashes must match the Phase 2 manifest.
See [Phase 3 contract](../reports/preprocessing/phase3_findings.md) and [verification summary](../reports/preprocessing/summary.json).
These generated caches are useful for full-year train/validation/test experiments, **not** for within-train CV.
Chronological CV must start from raw 2019 rows and refit the whole pipeline inside each fold.

## Evaluation scope

Detailed feature–target EDA uses 2019 only. 2020 and 2021 are planned validation/test sets; yearly target distributions were previously inspected, so 2021 is not a completely blind holdout. STATS19 does not contain a defensible accident-occurrence negative class.
