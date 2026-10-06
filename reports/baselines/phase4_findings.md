# AccidentIQ — Phase 4: ML Baseline

Phase 4 tamamlandı: Dummy + iki sabit Logistic Regression deneyi. Advanced ML/DL/API henüz yok.
Kaynak snapshot Phase 2 hash'leriyle eşleşti. Classifier ve preprocessing birlikte artifact olarak saklandı.

## Deney protokolü

- Train: 2019 (117,536 kayıt), validation: 2020 (91,199 kayıt).
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
| Dummy (majority/prior) | 0.2936 | 0.0021 |
| Logistic Regression | 0.2936 | 0.0021 |
| Logistic Regression (balanced) | 0.3244 | 0.0192 |

Seçilen baseline: **Logistic Regression (balanced)**. Fold std güven aralığı değildir;
eğitim pencereleri örtüşür ve yalnız üç zaman penceresi vardır. Pooled OOF skoruyla karıştırılmamalı.

## 2020 validation sonuçları

| Model | Accuracy | Macro F1 | Balanced accuracy | Fatal recall | Serious recall | Log-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy (majority/prior) | 0.7835 | 0.2929 | 0.3333 | 0.0000 | 0.0000 | 0.5778 |
| Logistic Regression | 0.7835 | 0.2929 | 0.3333 | 0.0000 | 0.0000 | 0.5611 |
| Logistic Regression (balanced) | 0.4747 | 0.3193 | 0.4721 | 0.6175 | 0.2758 | 1.0208 |

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

models/baseline_v1/{model}.joblib: üç tam model pipeline'ı.
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
