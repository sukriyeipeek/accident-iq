# AccidentIQ — Phase 2 EDA bulguları

## Kapsam ve doğrulama

2019–2021 resmi STATS19 verileri: 309,822 collision, 44 ham collision kolonu.
Anahtar, full-row duplicate, yıl/tarih/gün, vehicle/casualty sayım ve hedef bütünlüğü kontrolleri geçti.
Dosya SHA-256 değerleri ve çalışma ortamı `data_manifest.json` içinde kaydedildi.
Veri temizlenmedi, satır silinmedi, imputation/scaling/encoder fit edilmedi.

Tüm yıllarda veri bütünlüğü kontrol edildi. Ayrıntılı feature–severity analizi yalnızca eğitim yılı 2019 kullanır.
2020 validation ve 2021 test olarak planlandı. Önceki EDA'da 2021'in hedef dağılımına bakıldığı için
bu test tamamen kör değildir; sonraki sonuç raporu bunu açıkça belirtecek.

## Hedef ve dengesizlik

2019: Fatal 1,658, Serious 22,980, Slight 92,898.
Sadece Slight tahmini yapan yaklaşımın eğitimdeki çoğunluk payı 79.04%; bu bir eğitilmiş model sonucu değildir.
Fatal/Serious recall, macro F1, balanced accuracy ve confusion matrix gereklidir.
Üç sınıfı koruyacağız; Serious+Fatal birleşimi raporlama için ayrıca değerlendirilebilir.
Önce class weighting karşılaştırılacak. SMOTE otomatik uygulanmayacak.

## Eksik veri

`isna()` tek başına yetersiz. Kod sözlüğündeki 'Data missing', 'Unknown' ve 'Unallocated' ayrı sayıldı.
0 bütün kolonlarda missing değildir: kavşakta 0 geçerli 'kavşak dışında' kategorisidir.
Darkness–lighting unknown geçerli bir ışık kategorisi olarak korundu; aydınlatmanın alt bilgisi bilinmiyor.
Unused/deprecated/sonradan eklenen kolonlarda eksiklik, alanın tarihsel kapsamından kaynaklanabilir.
Kavşak dışındayken eksik junction_control yapısal bağlamdır. Tüm veri içinde
128,527 kayıt bu duruma giriyor;
2019 ayrımı `column_profile_2019.csv` ve `junction_control_context_2019.csv` içinde görülebilir.
Eksik -1 değerini her satırda otomatik 0'a çevirmeyeceğiz; gerçekten kavşakta olan kayıtlar ayrı tutulacak.

Tüm yılların ham availability özeti (2019 ayrıntısı ayrı dosyada):

| column | null_count | coded_missing_count | unknown_count | unavailable_fraction |
| --- | --- | --- | --- | --- |
| junction_control | 0 | 129721 | 6436 | 43.95% |
| second_road_number | 0 | 0 | 128744 | 41.55% |
| trunk_road_flag | 0 | 23914 | 0 | 7.72% |
| pedestrian_crossing_human_control_historic | 0 | 551 | 10577 | 3.59% |
| pedestrian_crossing_physical_facilities_historic | 0 | 515 | 9223 | 3.14% |
| weather_conditions | 0 | 13 | 8605 | 2.78% |
| pedestrian_crossing | 0 | 500 | 6761 | 2.34% |
| road_type | 0 | 0 | 6756 | 2.18% |
| special_conditions_at_site | 0 | 809 | 4805 | 1.81% |
| junction_detail | 0 | 4968 | 0 | 1.60% |

## Severity oranları (2019)

Severe = Serious + Fatal. Payda, aynı koşuldaki raporlanmış yaralanmalı kazaların sayısıdır.
Bu tablo kaza gerçekleşme olasılığı değildir. Trafik hacmi/maruziyet olmadan 'yağmur kaza riskini artırır'
sonucu çıkarılamaz. Wilson aralıkları tanımlayıcıdır; bağımsız gözlem varsayımıyla hesaplanır,
mekânsal bağımlılık ve polis raporlama yanlılığını düzeltmez. n<100 kategorileri ayrı işaretlendi.

| feature | group | n | severe % | 95% interval | small_group |
| --- | --- | --- | --- | --- | --- |
| Weather | Fine + high winds | 1230 | 24.96% | 22.62%–27.45% | False |
| Weather | Fine no high winds | 92316 | 21.43% | 21.17%–21.70% | False |
| Weather | Fog or mist | 394 | 24.37% | 20.39%–28.84% | False |
| Weather | Other | 3043 | 15.51% | 14.27%–16.84% | False |
| Weather | Raining + high winds | 1430 | 24.76% | 22.59%–27.06% | False |
| Weather | Raining no high winds | 15294 | 20.10% | 19.47%–20.74% | False |
| Weather | Snowing + high winds | 40 | 15.00% | 7.06%–29.07% | True |
| Weather | Snowing no high winds | 353 | 17.56% | 13.95%–21.88% | False |
| Weather | Unknown | 3436 | 14.00% | 12.88%–15.20% | False |
| Road surface | Data missing or out of range | 299 | 8.03% | 5.45%–11.67% | False |
| Road surface | Dry | 82321 | 20.82% | 20.54%–21.10% | False |
| Road surface | Flood over 3cm. deep | 177 | 21.47% | 16.06%–28.09% | False |
| Road surface | Frost or ice | 1185 | 21.10% | 18.87%–23.51% | False |
| Road surface | Snow | 245 | 17.14% | 12.94%–22.36% | False |
| Road surface | Wet or damp | 32259 | 21.87% | 21.43%–22.33% | False |
| Road surface | unknown (self reported) | 1050 | 8.48% | 6.94%–10.32% | False |
| Light conditions | Darkness - lighting unknown | 2634 | 13.97% | 12.70%–15.35% | False |
| Light conditions | Darkness - lights lit | 24407 | 22.21% | 21.69%–22.73% | False |
| Light conditions | Darkness - lights unlit | 891 | 23.34% | 20.68%–26.23% | False |
| Light conditions | Darkness - no lighting | 6092 | 31.83% | 30.67%–33.01% | False |
| Light conditions | Data missing or out of range | 1 | 0.00% | 0.00%–79.35% | True |
| Light conditions | Daylight | 83511 | 20.00% | 19.73%–20.27% | False |
| Posted speed limit (mph) | 20.0 | 11747 | 16.19% | 15.54%–16.87% | False |
| Posted speed limit (mph) | 30.0 | 69305 | 19.18% | 18.89%–19.48% | False |
| Posted speed limit (mph) | 40.0 | 10021 | 22.47% | 21.67%–23.30% | False |
| Posted speed limit (mph) | 50.0 | 4716 | 23.05% | 21.87%–24.27% | False |
| Posted speed limit (mph) | 60.0 | 14514 | 31.45% | 30.70%–32.21% | False |
| Posted speed limit (mph) | 70.0 | 7153 | 21.43% | 20.50%–22.40% | False |
| Urban / rural | Rural | 37975 | 26.20% | 25.76%–26.64% | False |
| Urban / rural | Unallocated | 28 | 14.29% | 5.70%–31.49% | True |
| Urban / rural | Urban | 79533 | 18.47% | 18.20%–18.74% | False |

Gündüz severe oranı 20.00%, aydınlatmasız karanlıkta 31.83%.
Weather/surface ilişkileri tek yönlü basit bir 'kötü hava = daha ağır sonuç' kuralını desteklemiyor;
yol türü, hız limiti, sürüş davranışı, maruziyet ve raporlama farklılıkları karıştırıcı olabilir.

## Zaman, ilişkiler ve sayısal değişkenler

Saat 0–23, gün Pazartesi–Pazar, ay 1–12 ve mevsim takvim sırasıyla incelendi.
2019–2021 aylık kayıt sayıları `temporal_overview.png` içinde. 2020 düşüşü gözlemleniyor;
bu analiz tek başına pandeminin nedensel etkisini ölçmez. Yıllık sınıf dağılımı kayması da raporlandı.

Weather–surface bias-corrected Cramer's V: 0.404.
Kategorik kodlara Pearson/Spearman uygulamak sıralama varsayımı üretir; bu nedenle kategoriler için
Cramer's V, gerçek sayısal büyüklükler için Spearman kullanıldı. Konum density grafiği maruziyetten etkilenir.

| column | valid_n | min | q1 | median | q3 | max | mean | std | iqr_lower | iqr_upper | iqr_flagged_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| speed_limit | 117456 | 20.0 | 30.0 | 30.0 | 40.0 | 70.0 | 36.799 | 14.036 | 15.0 | 55.0 | 21667 |
| number_of_vehicles | 117536 | 1.0 | 1.0 | 2.0 | 2.0 | 17.0 | 1.841 | 0.708 | -0.5 | 3.5 | 2786 |
| number_of_casualties | 117536 | 1.0 | 1.0 | 1.0 | 1.0 | 52.0 | 1.303 | 0.754 | 1.0 | 1.0 | 23945 |
| latitude | 117508 | 49.918 | 51.466 | 51.853 | 53.357 | 60.389 | 52.395 | 1.36 | 48.629 | 56.195 | 1290 |
| longitude | 117508 | -7.525 | -2.181 | -1.123 | -0.135 | 1.757 | -1.249 | 1.392 | -5.248 | 2.932 | 373 |

IQR dışındaki değerler inceleme bayrağıdır; 70 mph veya çok araçlı kaza otomatik silinmez.
speed_limit gerçek ölçülmüş araç hızı değildir. İngiltere hız limiti birimi mph'dir.
2019 speed_limit grafiği ve oran tablosu yalnızca geçerli 20–70 mph değerlerini kullanır;
80 unavailable/standart dışı kayıt bu tanımlayıcı alt analizden hariçtir,
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
