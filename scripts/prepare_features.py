"""Prepare Phase 3 artifacts. No classifier is fitted or scored."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn
from sklearn.base import clone

from src.data.eda import file_manifest
from src.data.load_stats19 import load_collision_data, read_codebook
from src.data.splits import SPLIT_YEARS, quarterly_folds, temporal_split
from src.features.preprocessing import (CATEGORICAL_FEATURES, ENGINEERED_FEATURES, RAW_FEATURES,
                                        TARGET_LABELS, UNRECOGNIZED, encode_target, make_preprocessor,
                                        numeric_codes, recorded_timestamp)


def fitted_statistics(pipeline) -> dict:
    columns = pipeline.named_steps["columns"]
    speed = columns.named_transformers_["speed"]
    encoder = columns.named_transformers_["categories"]
    return {"speed_median_mph": float(speed.named_steps["impute"].statistics_[0]),
            "speed_mean_after_imputation_mph": float(speed.named_steps["scale"].mean_[0]),
            "speed_variance_after_imputation": float(speed.named_steps["scale"].var_[0]),
            "fit_samples": int(speed.named_steps["scale"].n_samples_seen_),
            "category_vocabulary": {name: list(values) for name, values in zip(CATEGORICAL_FEATURES, encoder.categories_)}}


def save_preprocessor(pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    joblib.dump(pipeline, partial, compress=3)
    partial.replace(path)


def prepare_features(root: Path) -> dict:
    root = Path(root).resolve()
    source_manifest = json.loads((root / "reports/eda/data_manifest.json").read_text(encoding="utf-8"))
    paths = [root / item["path"] for item in source_manifest["files"]]
    if file_manifest(paths, root) != source_manifest["files"]:
        raise ValueError("Input snapshot differs from Phase 2; rerun and review EDA before proceeding")
    guide = root / "config/stats19_data_guide_2025.xlsx"
    codebook = read_codebook(guide)
    partitions = temporal_split(load_collision_data(root / "data/raw/stats19", guide_path=guide))
    output = root / "data/processed/context_v1"
    report = root / "reports/preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    pipeline = make_preprocessor(codebook)
    train = partitions["train"]
    pipeline.fit(train.loc[:, RAW_FEATURES])
    fit_hash = joblib.hash(pipeline)
    names = list(pipeline.get_feature_names_out())
    statistics = fitted_statistics(pipeline)
    if statistics["fit_samples"] != len(train):
        raise ValueError("Preprocessor fit sample count disagrees with train partition")
    split_report = {}
    for role, part in partitions.items():
        X = part.loc[:, RAW_FEATURES]
        matrix = pipeline.transform(X)
        target = encode_target(part["collision_severity"])
        if matrix.shape != (len(part), len(names)) or not np.isfinite(matrix.data).all():
            raise ValueError(f"Invalid transformed matrix: {role}")
        sparse.save_npz(output / f"X_{role}.npz", matrix)
        np.save(output / f"y_{role}.npy", target, allow_pickle=False)
        timestamp = recorded_timestamp(X)
        pd.DataFrame({"collision_index": part["collision_index"], "timestamp_local": timestamp,
                      "row_position": np.arange(len(part))}).to_csv(output / f"rows_{role}.csv", index=False)
        built = pipeline.named_steps["context"].transform(X)
        unseen = {column: int((~built[column].isin(statistics["category_vocabulary"][column])).sum())
                  for column in CATEGORICAL_FEATURES}
        unexpected = {column: int(built[column].eq(UNRECOGNIZED).sum()) for column in CATEGORICAL_FEATURES}
        control = numeric_codes(X["junction_control"])
        outside = numeric_codes(X["junction_detail_historic"]).eq(0)
        split_report[role] = {"year": SPLIT_YEARS[role], "rows": len(part), "shape": list(matrix.shape),
                              "dtype": str(matrix.dtype), "nnz": matrix.nnz,
                              "density": matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
                              "timestamp_min": timestamp.min().isoformat(), "timestamp_max": timestamp.max().isoformat(),
                              "speed_missing_rows": int(built["speed_limit_missing"].sum()),
                              "junction_control_outside_nonmissing_rows": int((outside & control.notna() & ~control.isin([-1, 0, 9])).sum()),
                              "unseen_training_category_rows": unseen, "unrecognized_code_rows": unexpected}
        if joblib.hash(pipeline) != fit_hash:
            raise ValueError("Transform mutated learned training state")
    fold_report = []
    for number, (fit, valid) in enumerate(quarterly_folds(train), start=1):
        candidate = clone(make_preprocessor(codebook))
        candidate.fit(train.iloc[fit].loc[:, RAW_FEATURES])
        fold_hash = joblib.hash(candidate)
        transformed = candidate.transform(train.iloc[valid].loc[:, RAW_FEATURES])
        if not np.isfinite(transformed.data).all() or joblib.hash(candidate) != fold_hash:
            raise ValueError("Fold transformation invalid or changed learned state")
        fold_report.append({"fold": number, "train_rows": len(fit), "validation_rows": len(valid),
                            "train_until": recorded_timestamp(train.iloc[fit]).max().isoformat(),
                            "validation_from": recorded_timestamp(train.iloc[valid]).min().isoformat(),
                            "output_features": transformed.shape[1], "statistics": fitted_statistics(candidate)})
    artifact = root / "models/preprocessing_context_v1.joblib"
    save_preprocessor(pipeline, artifact)
    restored = joblib.load(artifact)
    probe = train.loc[:, RAW_FEATURES].iloc[:32]
    delta = restored.transform(probe) - pipeline.transform(probe)
    if delta.nnz != 0 or list(restored.get_feature_names_out()) != names:
        raise ValueError("Serialized preprocessor round-trip mismatch")
    summary = {"phase": 3, "feature_set": "context_v1", "classifier_trained": False,
               "fit_year": 2019, "raw_features": list(RAW_FEATURES), "engineered_features": list(ENGINEERED_FEATURES),
               "output_feature_count": len(names), "output_feature_names": names,
               "target_mapping": TARGET_LABELS, "statistics": statistics, "splits": split_report,
               "cv_folds": fold_report, "checks": {"source_snapshot_matches_eda": True,
               "training_only_fit": True, "transform_does_not_refit": True,
               "finite_consistent_matrices": True, "serialization_round_trip": True,
               "chronological_cv": True},
               "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                               "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
               "input_manifest": source_manifest["files"],
               "artifact_manifest": file_manifest(sorted(output.glob("*")) + [artifact], root),
               "test_policy": "2021 yearly labels were previously inspected. No holdout scoring or category-driven feature tuning in Phase 3.",
               "cv_policy": "Refit a fresh complete model pipeline per raw-data fold; do not cross-validate cached X_train."}
    (report / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    write_findings(report, summary)
    return summary


def write_findings(report: Path, summary: dict) -> None:
    splits = "\n".join(f"| {role} | {info['year']} | {info['rows']:,} | {info['shape'][1]} | {info['speed_missing_rows']} |"
                       for role, info in summary["splits"].items())
    text = f"""# AccidentIQ — Phase 3: Preprocessing + Feature Engineering

Phase 3 tamamlandı. Classifier eğitilmedi, performans metriği hesaplanmadı.
Ham veri korundu. Kaynak dosyalarının Phase 2 SHA-256 snapshot'ına eşitliği kontrol edildi.

## Girdiler ve hedef

{len(summary['raw_features'])} ham girdi: {', '.join(summary['raw_features'])}.
Target raw 1/2/3 -> 0/Fatal, 1/Serious, 2/Slight. Bu sabit schema dönüşümü y'den bir şey öğrenmez.
Hedef P(severity | reported injury collision, X); kaza gerçekleşme riski değil.

## İşleme sözleşmesi

- Date: DD/MM/YYYY, ISO YYYY-MM-DD veya timezone-naive pandas datetime. Time: HH:MM.
  Tarih/saat zorunlu ve geçerli olmalı; hatalı tarih/saatle sessiz tahmin üretilmez.
- Hour (dakika bilgisi dahil), weekday ve month için sin/cos. Periyotlar 24, 7, 12.
  Season month'tan türediği için ilk feature setinde redundant ayrı alan olarak eklenmedi.
- Kategoriler resmi dictionary ile nominal label'a çevrilir. Missing, unknown, unrecognized ayrı token.
  Darkness - lighting unknown geçerli ışık kategorisi olarak kalır.
- Kavşak dışı detail=0 iken control uygulanamaz token olur; raw control=0 da aynı anlamdadır.
  Kavşakta -1 control ise missing kalır. Çelişkili kavşak dışı control kayıtları summary'de sayılır;
  ham veri değişmez, bu feature tasarımı için control bilgisinin uygulanamazlığı esas alınır.
- Speed limit: mph. 20/30/40/50/60/70 dışı veya null değer missing olur; satır silinmez.
  Missing indicator korunur. Train medyanıyla imputation, ardından train-fitted StandardScaler.
  Tüm train hızları eksikse sıfır medyan uydurulmaz, fit hata verir.
- Cyclic özellikler ve missing indicator zaten sınırlı aralıkta, ayrıca standardize edilmez.
- OneHotEncoder vocabulary yalnızca fit partition'dan öğrenilir, drop yok, nadir kategori birleştirme yok.
  Train'de görülmeyen kategori ilgili one-hot blokta sıfır üretir (handle_unknown=ignore).
  Bu 'bilinen bir sınıf' tahmini değildir; unseen sayıları operasyonel schema kontrolleri olarak raporlanır.
  Holdout'tan vocabulary genişletilmez. Sayısal olmayan, kesirli veya sonsuz code girişi reddedilir.
- CSR float32 çıktı: {summary['output_feature_count']} feature. Feature isimleri ve sıra artifact ile saklanır.

## Split ve leakage koruması

| Partition | Yıl | Satır | Çıktı feature | Missing speed |
| --- | --- | --- | --- | --- |
{splits}

Preprocessor yalnızca 2019'a fit edildi. Öğrenilen speed medyanı {summary['statistics']['speed_median_mph']:g} mph.
2020/2021 transform sırasında fit-state hash'inin değişmediği doğrulandı.
Her partition tarih/saat/collision_index sırasıyla deterministik sıralandı. Satır anahtarları ayrı sidecar'da;
target, yıl, kimlik ve koordinat gibi leakage/proxy alanları allowlist sayesinde X'e ulaşmaz.

İlk sürümde second_road_class, pedestrian crossing ve trunk flag eklenmedi: EDA'daki adaylar zorunlu
girdiler değildi. İlk bağlam baseline'ını küçük tutuyoruz. Bunlar daha sonra yalnız train içi ablation ile
değerlendirilebilir. Number of casualties/vehicles, adjusted/enhanced severity ve polis müdahalesi hariç.

## Cross-validation

2019 içinde üç expanding-window fold: Q1 -> Q2; Q1–Q2 -> Q3; Q1–Q3 -> Q4.
Aynı zaman noktasındaki kayıtlar train/validation sınırını paylaşmaz. Her fold'da yeni preprocessor fit
edilip sonraki çeyrek transform edildi. Farklı fold'larda vocabulary/dimension değişebilir; beklenen budur.
Bu bir CV skor raporu değil. Phase 4'te make_model_pipeline ile classifier da aynı pipeline'a bağlanacak.
Önceden full-train fit edilmiş X_train üzerinde CV yapılmamalı. Yerel zamanlar timezone/DST açısından
belirsiz olabilir; veride timezone yoktur, UTC gibi yorumlanmadı. Çeyrek sınırları geleceği taklit eden
zamansal genelleme kontrolüdür; spatial independence garantisi değildir.

## Artifact ve tekrar üretilebilirlik

- data/processed/context_v1/X_{{train,validation,test}}.npz: CSR float32.
- y_*.npy: sabit mapping ile int64 hedefler.
- rows_*.csv: aynı sıradaki kimlik/timestamp/row_position (model input değil).
- models/preprocessing_context_v1.joblib: preprocessor, classifier değil.
- reports/preprocessing/summary.json: train istatistikleri, fold'lar, schema kontrolleri, ortam ve hash'ler.

Raw data, processed row-level data ve joblib artifact GitHub'a eklenmez. Summary aggregate'dir.
Joblib/pickle güvenilmeyen dosyadan yüklenmemeli; yükleme Python/sklearn sürümleriyle uyum gerektirir.
İşlem rastgele örnekleme içermez; seed/shuffle gerektirmez. İleride modeller için seed=42 kullanılacak.

## Kontroller ve sınırlar

Training-only fit, finite values, aynı feature sırası/boyutu, transform-state sabitliği, kaynak snapshot,
zamansal CV ve artifact round-trip kontrolleri geçti. Model başarısı veya üretim uygunluğu iddia edilmez.
2021 target dağılımı Phase 2 öncesinde görüldüğünden test pristine değildir; burada target'a göre tuning yoktur.
Fold-fitted işlem leakage'i azaltır; gözlem yanlılığını, causal confounding'i veya reporting shift'i çözmez.

## Öğrenme notları

WHAT WE DID: ham girdilerden tutarlı sparse feature matrisleri ve tekrar kullanılabilir preprocessor ürettik.
WHY WE DID IT: aynı feature kurallarını eğitim ve inference'da kullanmak, held-out veriden öğrenmeyi engellemek.
WHAT I SHOULD LEARN: fit istatistik öğrenir; transform aynı istatistikleri uygular. Pipeline CV sırasında da
her fold'un training kısmına fit edilmelidir. One-hot kategori kodlarına sahte sayısal sıra yüklemez.
IMPORTANT CONCEPTS: preprocessing, feature engineering, imputation, scaling, cyclic encoding, one-hot,
schema contract, unseen categories, data leakage ve chronological cross-validation.

## Teknik referanslar

- [OneHotEncoder (scikit-learn 1.7)](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [ColumnTransformer](https://scikit-learn.org/1.7/modules/generated/sklearn.compose.ColumnTransformer.html)
- [SimpleImputer](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html)

Sonraki adım Phase 4: Dummy ve Logistic Regression baseline. Phase 3 hiçbir classifier eğitmez.
"""
    (report / "phase3_findings.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    summary = prepare_features(parser.parse_args().root)
    print(json.dumps({"phase": summary["phase"], "classifier_trained": summary["classifier_trained"],
                      "features": summary["output_feature_count"], "splits": {k: v["shape"] for k, v in summary["splits"].items()},
                      "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
