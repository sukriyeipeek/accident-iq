from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


DISPLAY_NAMES = {"dummy_prior": "Dummy (majority/prior)", "logistic": "Logistic Regression",
                 "logistic_balanced": "Logistic Regression (balanced)"}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "average_precision_ovr_macro", "roc_auc_ovr_macro")


def metric_table(results: dict) -> str:
    rows = ["| Model | Accuracy | Macro F1 | Balanced accuracy | Fatal recall | Serious recall | Log-loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in results.items():
        metric = result["validation"]
        rows.append(f"| {DISPLAY_NAMES[name]} | {metric['accuracy']:.4f} | {metric['macro_f1']:.4f} | "
                    f"{metric['balanced_accuracy']:.4f} | {metric['per_class']['Fatal']['recall']:.4f} | "
                    f"{metric['per_class']['Serious']['recall']:.4f} | {metric['log_loss']:.4f} |")
    return "\n".join(rows)


def plot_reports(report: Path, summary: dict, y_valid: np.ndarray, probabilities: dict) -> None:
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = ["#728092", "#2f6f9c", "#bb6d37"]
    labels = ["Fatal", "Serious", "Slight"]
    names = list(summary["results"])
    for name, result in summary["results"].items():
        matrix = np.array(result["validation"]["confusion_matrix"])
        totals = matrix.sum(axis=1, keepdims=True)
        rates = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, values, title, decimals in zip(axes, [matrix, rates], ["Counts", "Row-normalized (recall)"], [0, 3]):
            artist = ax.imshow(values, cmap="Blues", vmin=0, vmax=None if decimals == 0 else 1)
            for row in range(3):
                for col in range(3):
                    text = f"{values[row, col]:,}" if decimals == 0 else f"{values[row, col]:.1%}"
                    ax.text(col, row, text, ha="center", va="center",
                            color="white" if values[row, col] > values.max() * 0.55 else "#222222")
            ax.set_xticks(range(3), labels)
            ax.set_yticks(range(3), labels)
            ax.set(xlabel="Predicted", ylabel="True", title=title)
            fig.colorbar(artist, ax=ax, fraction=0.045)
        fig.suptitle(f"2020 validation: {DISPLAY_NAMES[name]}")
        fig.tight_layout()
        fig.savefig(figures / f"confusion_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for index, (ax, label) in enumerate(zip(axes, labels)):
        binary = y_valid == index
        for name, color in zip(names, colors):
            precision, recall, _ = precision_recall_curve(binary, probabilities[name][:, index])
            ap = summary["results"][name]["validation"]["per_class"][label]["average_precision_ovr"]
            ax.step(recall, precision, where="post", label=f"{DISPLAY_NAMES[name]}\nAP={ap:.3f}", color=color)
        ax.axhline(binary.mean(), color="#999999", linestyle=":", label=f"Prevalence={binary.mean():.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", title=f"{label} vs rest", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("2020 validation: precision-recall (AP is not trapezoidal PR-AUC)")
    fig.tight_layout()
    fig.savefig(figures / "precision_recall_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    short = ["Dummy", "Logistic", "Balanced LR"]
    for ax, metric in zip(axes, ["accuracy", "macro_f1", "balanced_accuracy"]):
        values = [summary["results"][name]["validation"][metric] for name in names]
        ax.bar(short, values, color=colors)
        for i, value in enumerate(values):
            ax.text(i, value + 0.01, f"{value:.3f}", ha="center")
        ax.set(title=metric.replace("_", " ").title(), ylim=(0, 1))
    fig.suptitle("2020 validation: accuracy does not describe rare-class performance")
    fig.tight_layout()
    fig.savefig(figures / "validation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, color in zip(names, colors):
        values = [fold["metrics"]["macro_f1"] for fold in summary["cv_results"][name]]
        ax.plot([2, 3, 4], values, marker="o", label=DISPLAY_NAMES[name], color=color)
    ax.set(xticks=[2, 3, 4], xticklabels=["Q2", "Q3", "Q4"], xlabel="2019 validation quarter",
           ylabel="Macro F1", title="Chronological CV: fresh full pipeline per fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "cv_macro_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(names))
    groups = [("Train 2019 (in-sample)", [summary["results"][n]["train"]["macro_f1"] for n in names]),
              ("2019 CV (mean)", [summary["cv_summary"][n]["macro_f1_mean"] for n in names]),
              ("Validation 2020", [summary["results"][n]["validation"]["macro_f1"] for n in names])]
    for index, (label, values) in enumerate(groups):
        ax.bar(x + (index - 1) * 0.25, values, width=0.25, label=label)
    ax.set_xticks(x, short)
    ax.set(ylabel="Macro F1", ylim=(0, 1), title="Apparent vs temporal generalization; folds overlap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "generalization_macro_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_findings(report: Path, summary: dict) -> None:
    selected = summary["selected_baseline"]
    cv_rows = "\n".join(f"| {DISPLAY_NAMES[n]} | {m['macro_f1_mean']:.4f} | {m['macro_f1_std']:.4f} |"
                        for n, m in summary["cv_summary"].items())
    text = f"""# AccidentIQ — Phase 4: ML Baseline

Phase 4 tamamlandı: Dummy + iki sabit Logistic Regression deneyi. Advanced ML/DL/API henüz yok.
Kaynak snapshot Phase 2 hash'leriyle eşleşti. Classifier ve preprocessing birlikte artifact olarak saklandı.

## Deney protokolü

- Train: 2019 ({summary['train_rows']:,} kayıt), validation: 2020 ({summary['validation_rows']:,} kayıt).
- 2021 collision label/feature dosyası bu aşamada yüklenmedi ve skorlanmadı. Yalnız kaynak fingerprint'i
  bütün snapshot doğrulaması için okundu. Daha önce yıllık hedef dağılımı görüldüğünden test pristine değildir.
- 2019 Q1→Q2, Q1–Q2→Q3, Q1–Q3→Q4: her fold'a raw girdiden yeni tam model pipeline'ı fit edildi.
- Hiperparametreler önceden sabit: C=1, L2, lbfgs, tol=1e-6, max_iter=2000, seed=42.
  Random/grid search veya validation'a göre C/threshold ayarlama yapılmadı. CPU thread limit=1.
- Dummy strategy=prior: hard prediction çoğunluk sınıfı, probability training class prior.
  Dolayısıyla yalnızca 0/1 probability veren most_frequent yerine anlamlı log-loss referansı var.
- Logistic unweighted ve balanced karşılaştırıldı. Balanced ağırlıklar her fold'un kendi y_train'inden
  n/(3*n_class) ile öğrenilir. SMOTE/sampling yok. Karar standart multiclass argmax.
- Primary seçim: üç CV fold'unda eşit ağırlıklı mean macro F1, eşitlikte model adı.
  Validation sonucu seçimde kullanılmadı. Bu seçim deployment onayı değildir.

## 2019 kronolojik CV

| Model | Mean macro F1 | Fold std (ddof=1) |
| --- | ---: | ---: |
{cv_rows}

Seçilen baseline: **{DISPLAY_NAMES[selected]}**. Fold std güven aralığı değildir;
eğitim pencereleri örtüşür ve yalnız üç zaman penceresi vardır. Pooled OOF skoruyla karıştırılmamalı.

## 2020 validation sonuçları

{metric_table(summary['results'])}

Tüm sınıfların precision/recall/F1/support, prediction count ve OVR AUC/AP değerleri metrics.json içinde.
Confusion matrix sırası Fatal, Serious, Slight; satır=true, sütun=predicted. Model hiçbir sınıfı tahmin
etmiyorsa precision tanımsızdır: sayısal raporda 0 kullanılır ve precision_defined=false olarak belirtilir.
Yüksek accuracy, Fatal/Serious recall yokken yeterli başarı değildir.

AP (average precision), trapezoidal PR-AUC değildir. Rare class AP'si sınıf prevalence referansıyla
karşılaştırılmalıdır. OVR ROC-AUC tek başına düşük prevalence'de yanıltıcı iyimserlik verebilir.
Log-loss ve multiclass Brier de raporlanır; Brier = mean(sum_class((p-y_onehot)^2)), aralığı [0,2].
Balanced sınıf maliyetlerini değiştirir; probability'leri doğal prevalence'e kalibre edilmiş confidence
gibi gösterilmemeli. Calibration/threshold seçimi ayrı değerlendirme aşamasının konusudur.

## Genelleme, karmaşıklık ve maliyet

Train in-sample, 2019 CV ve 2020 validation ayrı raporlandı. 2020'deki fark hem temporal shift hem
model genellemesiyle ilgili olabilir; train/validation farkı tek başına overfitting'in nedensel kanıtı değildir.
Logistic karar sınırı engineered özellikler üzerinde lineerdir; etkileşimleri sınırlı öğrenir.
Dummy feature kullanmaz. Logistic daha esnektir fakat düşük sınıf recall/precision trade-off'u sürer.

Training süresi pipeline preprocessing dahil; latency warm-up sonrası 128'e kadar satırlık batch üzerinde
5 tekrar end-to-end predict_proba medyanıdır. CPU, batch ve sklearn ortamına bağımlıdır; production SLA değil.
Süreler ve artifact byte boyutları summary.json içinde. Bu aşamada SHAP/causal explanation eklenmedi.

## Artifact ve güvenlik

models/baseline_v1/{{model}}.joblib: üç tam model pipeline'ı.
models/baseline_v1/selected.joblib: yalnız 2019 CV ile seçilen pipeline.
config/baseline_protocol.json: sabit deney protokolü.
data/processed/baseline_v1/: validation probabilities, target ve collision key sidecar; GitHub'a eklenmez.
reports/baselines/: metrics, summary, feature names, yedi grafik ve bu öğrenme raporu.

Artifact reload sonrası raw-input probability/prediction eşitliği kontrol edildi. Güvenilmeyen joblib
dosyasını yüklemeyin. Request input sözleşmesi Phase 3 allowlist'i; prediction target veya identity istemez.
API entegrasyonu henüz yapılmadı; occurrence risk veya güvenli sürüş tavsiyesi üretmez.

## Öğrenme notları

WHAT WE DID: Dummy ve Logistic baseline'larını leakage-aware pipeline ile eğittik, kronolojik CV ve 2020
validation sonuçlarını sınıf bazlı karşılaştırdık.
WHY WE DID IT: advanced ML/DL'nin gerçekten fayda sağlayıp sağlamadığını kıyaslayacağımız ölçülebilir referans kurmak.
WHAT I SHOULD LEARN: classification argmax kararları ile probability kalitesi farklıdır. Accuracy çoğunluk
sınıfına duyarlıdır; macro F1 sınıflara eşit ağırlık verir. Class weighting recall/precision ve calibration'ı değiştirir.
IMPORTANT CONCEPTS: classification, regularization (C ters güç), cross-validation, class imbalance,
precision/recall/F1, confusion matrix, OVR ROC-AUC, average precision, log-loss, model persistence.

## Referanslar

- [LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [DummyClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.dummy.DummyClassifier.html)
- [Average precision](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html)

Sonraki aşama Phase 5: kontrollü advanced ML. 2021 final evaluation için saklı tutulur.
"""
    (report / "phase4_findings.md").write_text(text, encoding="utf-8")
