from __future__ import annotations

import json
import platform
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.eda import (ANALYSIS_COLUMNS, audit_tables, code_flags, cramers_v,
                          file_manifest, numeric_profile, profile_columns, severity_rates)
from src.data.load_stats19 import load_collision_data, read_codebook


SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
COLORS = ["#b44747", "#da9c3f", "#416c9c"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TITLES = {"hour": "Hour (local recorded time)", "weekday": "Weekday", "month": "Month",
          "season": "Season", "weather_conditions_label": "Weather",
          "road_surface_conditions_label": "Road surface", "light_conditions_label": "Light conditions",
          "road_type_label": "Road type", "junction_detail_historic_label": "Junction (historic codes)",
          "junction_control_label": "Junction control", "speed_limit": "Posted speed limit (mph)",
          "urban_or_rural_area_label": "Urban / rural"}


def markdown_table(frame: pd.DataFrame) -> str:
    names = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rates(table: pd.DataFrame, column: str, path: Path) -> None:
    ordered = table.sort_values(column)
    if column == "season":
        ordered = table.set_index(column).reindex(["Winter", "Spring", "Summer", "Autumn"]).reset_index()
    labels = [DAY_NAMES[int(v)] if column == "weekday" else str(int(v)) if isinstance(v, (float, np.floating)) else str(v)
              for v in ordered[column]]
    compact = column in {"hour", "weekday", "month", "season", "speed_limit"}
    if column == "speed_limit":
        labels = [f"{label} mph" if float(label) > 0 else f"Missing ({label})" for label in labels]
    labels = [(label + f" (n={n:,})") if compact else (textwrap.fill(label, 27) + f"\n(n={n:,})")
              for label, n in zip(labels, ordered["n"])]
    row_height = 0.42 if compact else 0.76
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(ordered) * row_height + 2)), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(ordered))
    left = np.zeros(len(ordered))
    for label, color in zip(("fatal", "serious", "slight"), COLORS):
        values = ordered[f"{label}_fraction"].to_numpy()
        axes[0].barh(y, values, left=left, label=label.title(), color=color)
        left += values
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xlabel("Within-group severity distribution")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    p = ordered["severe_fraction"].to_numpy()
    err = np.vstack((np.maximum(0, p - ordered["severe_ci_low"].to_numpy()),
                     np.maximum(0, ordered["severe_ci_high"].to_numpy() - p)))
    axes[1].errorbar(p, y, xerr=err, fmt="o", color="#333333", capsize=3)
    small = ordered["small_group"].to_numpy()
    axes[1].scatter(p[small], y[small], marker="x", color="#b44747", zorder=4, label="n < 100")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, min(1, max(0.45, float(ordered["severe_ci_high"].max()) + 0.04)))
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel("Serious + Fatal fraction (95% Wilson interval)")
    if small.any():
        axes[1].legend(loc="lower right")
    fig.suptitle(f"2019 training data: severity by {TITLES[column].lower()}")
    fig.text(0.5, 0.005, "Reported injury collisions only. These are not accident occurrence probabilities or causal effects.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_figure(fig, path)


def run_eda(root: Path) -> dict:
    root = Path(root).resolve()
    data_dir = root / "data" / "raw" / "stats19"
    guide_path = root / "config" / "stats19_data_guide_2025.xlsx"
    output = root / "reports" / "eda"
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    frame = load_collision_data(data_dir, guide_path=guide_path)
    codebook = read_codebook(guide_path)
    links, source_paths, raw_columns = audit_tables(data_dir)
    train = frame.loc[frame["collision_year"] == 2019].copy()

    target = pd.crosstab(frame["collision_year"], frame["collision_severity"]).reindex(columns=[1, 2, 3], fill_value=0)
    target.columns = SEVERITY_ORDER
    target["total"] = target.sum(axis=1)
    target.to_csv(tables / "target_by_year.csv")
    target.iloc[:, :3].div(target["total"], axis=0).to_csv(output / "severity_by_year.csv")
    profile = profile_columns(frame[raw_columns], codebook)
    profile.to_csv(output / "missingness.csv", index=False)
    profile_columns(train[raw_columns], codebook).to_csv(tables / "column_profile_2019.csv", index=False)
    yearly_profiles = pd.concat([profile_columns(part[raw_columns], codebook).assign(year=year)
                                for year, part in frame.groupby("collision_year")], ignore_index=True)
    yearly_profiles.to_csv(tables / "column_profile_by_year.csv", index=False)

    expected_day = ((frame["weekday"] + 1) % 7) + 1
    quality = {
        "duplicate_collision_index_count": int(frame["collision_index"].duplicated().sum()),
        "invalid_target_count": int((~frame["collision_severity"].isin([1, 2, 3])).sum()),
        "invalid_date_count": int(frame["date"].isna().sum()),
        "invalid_hour_count": int(frame["hour"].isna().sum()),
        "date_year_mismatch_count": int(frame["date"].dt.year.ne(frame["collision_year"]).sum()),
        "weekday_mismatch_count": int(frame["day_of_week"].ne(expected_day).sum()),
        "missing_coordinate_pairs": int(frame[["latitude", "longitude"]].isna().any(axis=1).sum()),
        "zero_coordinate_pairs": int((frame["latitude"].eq(0) & frame["longitude"].eq(0)).sum()),
        "out_of_world_coordinate_rows": int(((frame["latitude"].abs() > 90) | (frame["longitude"].abs() > 180)).sum()),
        "unmapped_label_counts": {c: int(frame[c].eq("Unmapped code").sum())
                                  for c in frame.columns if c.endswith("_label")},
    }
    checks = {k: v == 0 for k, v in quality.items() if k in {
        "duplicate_collision_index_count", "invalid_target_count", "invalid_date_count",
        "invalid_hour_count", "date_year_mismatch_count", "weekday_mismatch_count", "out_of_world_coordinate_rows"}}
    checks["labels_mapped"] = all(v == 0 for v in quality["unmapped_label_counts"].values())
    for year, audit in links.items():
        checks[f"links_{year}"] = all(value == 0 for name, value in audit.items()
                                       if name not in {"collision_rows", "vehicle_rows", "casualty_rows", "collision_schema_equal"}) and audit["collision_schema_equal"]
    (output / "quality_checks.json").write_text(json.dumps({"checks": checks, "details": quality, "tables": links}, indent=2), encoding="utf-8")
    (output / "link_audit.json").write_text(json.dumps(links, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise ValueError(f"Data integrity checks failed: {[k for k, v in checks.items() if not v]}; see quality_checks.json")

    numeric = numeric_profile(train, codebook)
    numeric.to_csv(tables / "numeric_profile_2019.csv", index=False)
    pd.crosstab(train["junction_detail_historic"], train["junction_control"]).to_csv(tables / "junction_control_context_2019.csv")
    train["is_severe"] = train["collision_severity"].isin([1, 2]).astype(int)
    speed = train["speed_limit"]
    valid_speed = speed.isin([20, 30, 40, 50, 60, 70])
    quality["nonstandard_speed_rows_2019"] = int((~valid_speed).sum())
    group_rates = {}
    for column in ANALYSIS_COLUMNS:
        scope = train.loc[valid_speed] if column == "speed_limit" else train
        rates = severity_rates(scope, column)
        group_rates[column] = rates
        rates.to_csv(tables / f"severity_by_{column}_2019.csv", index=False)
        plot_rates(rates, column, figures / f"severity_by_{column}_2019.png")

    distribution_rows = []
    categories = [c for c in ANALYSIS_COLUMNS if c.endswith("_label")]
    for column in categories:
        counts = train[column].value_counts(dropna=False)
        for label, n in counts.items():
            distribution_rows.append({"column": column, "category": label, "n": int(n), "fraction": n / len(train)})
    pd.DataFrame(distribution_rows).to_csv(tables / "categorical_distribution_2019.csv", index=False)

    correlation_data = train[["speed_limit", "number_of_vehicles", "latitude", "longitude"]].copy()
    correlation_data.loc[~valid_speed, "speed_limit"] = np.nan
    numeric_correlation = correlation_data.corr(method="spearman")
    numeric_correlation.to_csv(tables / "spearman_2019.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(numeric_correlation, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag", ax=ax)
    ax.set_title("2019: Spearman correlation of numeric quantities\nTarget and categorical codes excluded")
    fig.tight_layout()
    save_figure(fig, figures / "numeric_correlation_2019.png")

    association_columns = categories + ["collision_severity_label"]
    association = pd.DataFrame(index=association_columns, columns=association_columns, dtype=float)
    for i, left in enumerate(association_columns):
        for right in association_columns[i:]:
            value = 1.0 if left == right else cramers_v(train[left], train[right])
            association.loc[left, right] = association.loc[right, left] = value
    association.to_csv(tables / "cramers_v_2019.csv")
    short = [TITLES.get(c, "Severity") for c in association_columns]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(association, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues",
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_title("2019: bias-corrected Cramer's V\nCategorical association, not causality")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, figures / "categorical_association_2019.png")

    monthly = frame.groupby(["collision_year", "month"]).size().rename("n").reset_index()
    monthly.to_csv(tables / "monthly_counts_all_years.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for year, part in monthly.groupby("collision_year"):
        axes[0].plot(part["month"], part["n"], marker="o", label=str(year))
    axes[0].set(xlabel="Month", ylabel="Reported collisions", title="Monthly counts: 2019–2021")
    axes[0].set_xticks(range(1, 13))
    axes[0].legend()
    for label, color in zip(SEVERITY_ORDER, COLORS):
        axes[1].plot(target.index, target[label] / target["total"], marker="o", label=label, color=color)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xticks(target.index)
    axes[1].set(title="Previously inspected yearly target distribution", xlabel="Year", ylabel="Severity fraction")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, figures / "temporal_overview.png")

    train_counts = train["collision_severity_label"].value_counts().reindex(SEVERITY_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(SEVERITY_ORDER, train_counts, color=COLORS)
    for i, n in enumerate(train_counts):
        ax.text(i, n, f"{n:,}\n{n / len(train):.2%}", ha="center", va="bottom")
    ax.set(title="2019 training target: class imbalance", ylabel="Reported collisions")
    ax.set_ylim(0, float(train_counts.max()) * 1.2)
    fig.tight_layout()
    save_figure(fig, figures / "target_2019.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column in zip(axes, ["speed_limit", "number_of_vehicles", "number_of_casualties"]):
        clean = train[column].mask(code_flags(train[column], codebook.get(column, {})).any(axis=1)).dropna()
        ax.hist(clean, bins=min(30, clean.nunique()), color="#416c9c")
        ax.set(title=column.replace("_", " "), ylabel="Collisions")
        if column == "number_of_casualties":
            ax.set_title("Casualties (outcome; exclude from model)")
        elif column == "speed_limit":
            ax.set_title("Posted speed limit (mph)")
    fig.tight_layout()
    save_figure(fig, figures / "numeric_distributions_2019.png")

    spatial = train.dropna(subset=["longitude", "latitude"])
    fig, ax = plt.subplots(figsize=(6, 8))
    density = ax.hexbin(spatial["longitude"], spatial["latitude"], gridsize=65, mincnt=1, bins="log", cmap="Blues")
    fig.colorbar(density, ax=ax, label="Reported collision count (log scale)")
    ax.set(title="2019 collision locations (WGS84)\nDensity reflects exposure and reporting too", xlabel="Longitude", ylabel="Latitude")
    fig.tight_layout()
    save_figure(fig, figures / "spatial_density_2019.png")

    missing_train = profile_columns(train[raw_columns], codebook).query("unavailable_fraction > 0").head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(missing_train["column"], missing_train["unavailable_fraction"], color="#416c9c")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="2019: null + coded missing + unknown\nAvailability differs from pandas isna()", xlabel="Unavailable fraction")
    fig.tight_layout()
    save_figure(fig, figures / "missingness_2019.png")

    summary = {"rows": len(frame), "raw_columns": len(raw_columns), "analysis_year": 2019,
               "years": {str(y): int(n) for y, n in frame["collision_year"].value_counts().sort_index().items()},
               "target_distribution": {str(y): int(n) for y, n in frame["collision_severity"].value_counts().sort_index().items()},
               "train_target": {label: int(n) for label, n in train_counts.items()},
               "quality": quality, "all_integrity_checks_passed": all(checks.values()),
               "figure_count": len(ANALYSIS_COLUMNS) + 7,
               "interpretation": "P(severity | police-reported injury collision, X); not accident occurrence risk",
               "test_policy": "2021 yearly labels were inspected earlier. New feature/target analyses use only 2019. Test is not pristine."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "quality_checks.json").write_text(json.dumps({"checks": checks, "details": quality, "tables": links}, indent=2), encoding="utf-8")
    manifest = {"source": "https://www.gov.uk/government/statistical-data-sets/road-safety-open-data",
                "guide_url": "https://assets.publishing.service.gov.uk/media/6ab2a71d997a4b2950cced58/dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx",
                "files": file_manifest(source_paths + [guide_path], root),
                "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                                "numpy": np.__version__, "matplotlib": matplotlib.__version__, "seaborn": sns.__version__}}
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_feature_decisions(output, raw_columns)
    write_findings(output, summary, group_rates, numeric, profile, association)
    return summary


def write_feature_decisions(output: Path, columns: list[str]) -> None:
    keep = {"date", "time", "road_type", "speed_limit", "junction_detail_historic", "junction_control",
            "light_conditions", "weather_conditions", "road_surface_conditions", "urban_or_rural_area",
            "first_road_class", "second_road_class", "pedestrian_crossing_human_control_historic",
            "pedestrian_crossing_physical_facilities_historic", "trunk_road_flag"}
    outcome = {"number_of_casualties", "enhanced_severity_collision", "collision_injury_based",
               "collision_adjusted_severity_serious", "collision_adjusted_severity_slight",
               "did_police_officer_attend_scene_of_accident"}
    identifiers = {"collision_index", "collision_ref_no"}
    rows = []
    for column in columns:
        if column == "collision_severity":
            decision, available, reason = "target", "after outcome", "Reported severity label; never an input"
        elif column in {"date", "time"}:
            decision, available, reason = "candidate", "yes if supplied", "Derive calendar/cyclic features; do not feed raw timestamp identifiers"
        elif column in keep:
            decision, available, reason = "candidate", "if observed at prediction time", "Road/context input; verify deployment source. Nominal codes except speed_limit."
        elif column in outcome:
            decision, available, reason = "exclude", "after outcome", "Outcome or reporting process; leakage/proxy risk"
        elif column in identifiers:
            decision, available, reason = "key_only", "not a predictor", "Record identity; join/audit only"
        elif column == "collision_year":
            decision, available, reason = "split_only", "yes", "Temporal split; not a training predictor"
        elif column == "police_force":
            decision, available, reason = "group_audit_only", "location-dependent", "Recording practice and geographic confounding"
        elif column == "number_of_vehicles":
            decision, available, reason = "exclude_v1_context", "after collision assessment", "Unknown in a prospective environmental scenario"
        elif column in {"latitude", "longitude", "location_easting_osgr", "location_northing_osgr"}:
            decision, available, reason = "defer_spatial", "yes if supplied", "Spatial memorization; requires geographic holdout evaluation"
        elif column == "day_of_week":
            decision, available, reason = "derive_from_date", "yes", "Validated against date; avoid redundant raw/derived weekday inputs"
        elif column in {"junction_detail", "pedestrian_crossing", "carriageway_hazards"}:
            decision, available, reason = "exclude_converted_v1", "varies", "Converted 2024 schema; use source-era context rather than redundant aliases"
        elif column in {"carriageway_hazards_historic", "special_conditions_at_site"}:
            decision, available, reason = "exclude_pending_availability", "not guaranteed", "Police scene assessment may not be available before outcome"
        else:
            decision, available, reason = "exclude_v1", "varies", "Administrative/high-cardinality road/location code; not needed for first context model"
        rows.append({"feature": column, "decision": decision, "available_at_prediction": available, "reason": reason})
    pd.DataFrame(rows).to_csv(output / "feature_decisions.csv", index=False)


def write_findings(output: Path, summary: dict, rates: dict, numeric: pd.DataFrame,
                   profile: pd.DataFrame, association: pd.DataFrame) -> None:
    rows = []
    for column in ["weather_conditions_label", "road_surface_conditions_label", "light_conditions_label", "speed_limit", "urban_or_rural_area_label"]:
        table = rates[column].copy()
        table["group"] = table[column].astype(str)
        table["feature"] = TITLES[column]
        table["severe %"] = table["severe_fraction"].map(lambda x: f"{x:.2%}")
        table["95% interval"] = table.apply(lambda r: f"{r.severe_ci_low:.2%}–{r.severe_ci_high:.2%}", axis=1)
        rows.append(table[["feature", "group", "n", "severe %", "95% interval", "small_group"]])
    unavailable = profile.query("unavailable_fraction > 0").head(10).copy()
    unavailable["unavailable_fraction"] = unavailable["unavailable_fraction"].map(lambda x: f"{x:.2%}")
    counts = summary["train_target"]
    majority = counts["Slight"] / sum(counts.values())
    junction_profile = profile.loc[profile["column"] == "junction_control"].iloc[0]
    daylight = rates["light_conditions_label"].set_index("light_conditions_label").loc["Daylight", "severe_fraction"]
    unlit = rates["light_conditions_label"].set_index("light_conditions_label").loc["Darkness - no lighting", "severe_fraction"]
    text = f"""# AccidentIQ — Phase 2 EDA bulguları

## Kapsam ve doğrulama

2019–2021 resmi STATS19 verileri: {summary['rows']:,} collision, {summary['raw_columns']} ham collision kolonu.
Anahtar, full-row duplicate, yıl/tarih/gün, vehicle/casualty sayım ve hedef bütünlüğü kontrolleri geçti.
Dosya SHA-256 değerleri ve çalışma ortamı `data_manifest.json` içinde kaydedildi.
Veri temizlenmedi, satır silinmedi, imputation/scaling/encoder fit edilmedi.

Tüm yıllarda veri bütünlüğü kontrol edildi. Ayrıntılı feature–severity analizi yalnızca eğitim yılı 2019 kullanır.
2020 validation ve 2021 test olarak planlandı. Önceki EDA'da 2021'in hedef dağılımına bakıldığı için
bu test tamamen kör değildir; sonraki sonuç raporu bunu açıkça belirtecek.

## Hedef ve dengesizlik

2019: Fatal {counts['Fatal']:,}, Serious {counts['Serious']:,}, Slight {counts['Slight']:,}.
Sadece Slight tahmini yapan yaklaşımın eğitimdeki çoğunluk payı {majority:.2%}; bu bir eğitilmiş model sonucu değildir.
Fatal/Serious recall, macro F1, balanced accuracy ve confusion matrix gereklidir.
Üç sınıfı koruyacağız; Serious+Fatal birleşimi raporlama için ayrıca değerlendirilebilir.
Önce class weighting karşılaştırılacak. SMOTE otomatik uygulanmayacak.

## Eksik veri

`isna()` tek başına yetersiz. Kod sözlüğündeki 'Data missing', 'Unknown' ve 'Unallocated' ayrı sayıldı.
0 bütün kolonlarda missing değildir: kavşakta 0 geçerli 'kavşak dışında' kategorisidir.
Darkness–lighting unknown geçerli bir ışık kategorisi olarak korundu; aydınlatmanın alt bilgisi bilinmiyor.
Unused/deprecated/sonradan eklenen kolonlarda eksiklik, alanın tarihsel kapsamından kaynaklanabilir.
Kavşak dışındayken eksik junction_control yapısal bağlamdır. Tüm veri içinde
{int(junction_profile['missing_with_no_junction_count']):,} kayıt bu duruma giriyor;
2019 ayrımı `column_profile_2019.csv` ve `junction_control_context_2019.csv` içinde görülebilir.
Eksik -1 değerini her satırda otomatik 0'a çevirmeyeceğiz; gerçekten kavşakta olan kayıtlar ayrı tutulacak.

Tüm yılların ham availability özeti (2019 ayrıntısı ayrı dosyada):

{markdown_table(unavailable[['column', 'null_count', 'coded_missing_count', 'unknown_count', 'unavailable_fraction']])}

## Severity oranları (2019)

Severe = Serious + Fatal. Payda, aynı koşuldaki raporlanmış yaralanmalı kazaların sayısıdır.
Bu tablo kaza gerçekleşme olasılığı değildir. Trafik hacmi/maruziyet olmadan 'yağmur kaza riskini artırır'
sonucu çıkarılamaz. Wilson aralıkları tanımlayıcıdır; bağımsız gözlem varsayımıyla hesaplanır,
mekânsal bağımlılık ve polis raporlama yanlılığını düzeltmez. n<100 kategorileri ayrı işaretlendi.

{markdown_table(pd.concat(rows, ignore_index=True))}

Gündüz severe oranı {daylight:.2%}, aydınlatmasız karanlıkta {unlit:.2%}.
Weather/surface ilişkileri tek yönlü basit bir 'kötü hava = daha ağır sonuç' kuralını desteklemiyor;
yol türü, hız limiti, sürüş davranışı, maruziyet ve raporlama farklılıkları karıştırıcı olabilir.

## Zaman, ilişkiler ve sayısal değişkenler

Saat 0–23, gün Pazartesi–Pazar, ay 1–12 ve mevsim takvim sırasıyla incelendi.
2019–2021 aylık kayıt sayıları `temporal_overview.png` içinde. 2020 düşüşü gözlemleniyor;
bu analiz tek başına pandeminin nedensel etkisini ölçmez. Yıllık sınıf dağılımı kayması da raporlandı.

Weather–surface bias-corrected Cramer's V: {association.loc['weather_conditions_label', 'road_surface_conditions_label']:.3f}.
Kategorik kodlara Pearson/Spearman uygulamak sıralama varsayımı üretir; bu nedenle kategoriler için
Cramer's V, gerçek sayısal büyüklükler için Spearman kullanıldı. Konum density grafiği maruziyetten etkilenir.

{markdown_table(numeric.round(3))}

IQR dışındaki değerler inceleme bayrağıdır; 70 mph veya çok araçlı kaza otomatik silinmez.
speed_limit gerçek ölçülmüş araç hızı değildir. İngiltere hız limiti birimi mph'dir.
2019 speed_limit grafiği ve oran tablosu yalnızca geçerli 20–70 mph değerlerini kullanır;
{summary['quality']['nonstandard_speed_rows_2019']} unavailable/standart dışı kayıt bu tanımlayıcı alt analizden hariçtir,
ham veri ve diğer analizlerde korunmuştur. Ham kod dağılımı availability profiline yansır.
number_of_casualties yalnızca outcome analizi içindir ve predictor olmayacak.
Doğrudan visibility mesafesi veya trafik yoğunluğu alanı yoktur. Light conditions visibility metre ölçümü değildir.

## Phase 3'e aktarılan kararlar

- Tarih/saat: hour, weekday, month ve season; hour/month için döngüsel sin/cos dönüşümü adaydır.
- weather, surface, light, road_type, junction_control, urban/rural: nominal kategoriler olarak işlenecek.
- junction_control: kavşak dışında yapısal uygulanamaz durum ve kavşakta eksik bilgi ayrı ele alınacak.
- 2019–2021 kavşak bilgisi için junction_detail_historic korunacak. Yeni junction_detail eski sınıfları
  dönüştürdüğünden iki alan birlikte predictor yapılmayacak. `_historic` alanın label'ları resmi guide'dan alındı.
- speed_limit: 20–70 mph geçerli limitler; kodlanmış unavailable ve standart dışı değerler Phase 3'te ayrı ele alınacak.
- number_of_vehicles yalnızca olay sonrası, erken olay değerlendirmesi kapsamı seçilirse kullanılabilir;
  çevresel koşul senaryosu modeli için ilk feature listesinden çıkarılacak.
- casualty sayısı/şiddeti, enhanced/adjusted severity, injury-based flags, polis olay yeri müdahale bilgisi çıkarılacak.
- collision_index/ref_no anahtar; collision_year split; police_force kayıt sistemi/grup analizinde yardımcı alan.
  Bunlar ilk modelin girdisi olmayacak. Ham yüksek kardinaliteli yol numarası/LSOA/otorite kodları da hariç.
- latitude/longitude için mekânsal genelleme testi gerekir; ilk çevresel baseline'dan çıkarılacak.
- Eksik kodlar ham veride korunacak; dönüşümler training fold içinde fit edilecek.
- Nadir kategoriler otomatik birleşmeyecek. Birleştirme gerekiyorsa karar train verisiyle alınacak.

Tüm ham alanların tahmin anı/feature kararları `feature_decisions.csv` dosyasında.

## Öğrenme notları

WHAT WE DID: veri sözlüğü, eksik kodlar, duplicate/ilişki kontrolleri, dağılımlar, oranlar, aykırı değerler ve ilişkiler.
WHY WE DID IT: gerçek sorunları model eğitiminden önce bulmak ve hatalı performans yorumlarını önlemek.
WHAT I SHOULD LEARN: bir oranı paydasını bilerek yorumlamak; kategori kodunu sayısal büyüklük sanmamak;
train/validation/test rollerini ve tahmin anını netleştirmek.
IMPORTANT CONCEPTS: class imbalance, selection bias, exposure, data leakage, temporal shift,
conditional probability, Wilson interval, Cramer's V, IQR ve correlation vs causation.

## Kaynaklar

- [DfT Road Safety Open Data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)
- [DfT codebook](https://assets.publishing.service.gov.uk/media/6ab2a71d997a4b2950cced58/dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx)

Yaralanma temelli kayıt yöntemleri severity dağılımını değiştirebilir. Adjusted severity kolonları
bireysel sınıf etiketi/predictor olarak kullanılmadı. Hedef, raporlanmış collision severity'dir.
EDA ilişkileri neden-sonuç ilişkisi veya üretimde doğrulanmış model performansı değildir.

Önceki tüm-yıl grafikleri `legacy/` klasöründe korunur; güncel rapor bu çalışmanın 19 grafiğini kullanır.
"""
    (output / "phase2_findings.md").write_text(text, encoding="utf-8")
