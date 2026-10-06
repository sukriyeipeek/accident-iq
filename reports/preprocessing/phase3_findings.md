# AccidentIQ — Phase 3: Preprocessing + Feature Engineering

Phase 3 tamamlandı. Classifier eğitilmedi, performans metriği hesaplanmadı.
Ham veri korundu. Kaynak dosyalarının Phase 2 SHA-256 snapshot'ına eşitliği kontrol edildi.

## Girdiler ve hedef

11 ham girdi: date, time, speed_limit, weather_conditions, road_surface_conditions, light_conditions, road_type, junction_detail_historic, junction_control, urban_or_rural_area, first_road_class.
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
- CSR float32 çıktı: 63 feature. Feature isimleri ve sıra artifact ile saklanır.

## Split ve leakage koruması

| Partition | Yıl | Satır | Çıktı feature | Missing speed |
| --- | --- | --- | --- | --- |
| train | 2019 | 117,536 | 63 | 80 |
| validation | 2020 | 91,199 | 63 | 12 |
| test | 2021 | 101,087 | 63 | 0 |

Preprocessor yalnızca 2019'a fit edildi. Öğrenilen speed medyanı 30 mph.
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

- data/processed/context_v1/X_{train,validation,test}.npz: CSR float32.
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
