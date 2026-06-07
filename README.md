# Eğiticili Makine Öğrenmesi Yöntemleri ile Deepvoice Tahmini

Yıldız Teknik Üniversitesi — Bilgisayar Mühendisliği Bölümü, Bilgisayar Projesi.

Bu çalışmada, yapay zekâ tabanlı metinden-sese (TTS) ve ses dönüşümü sistemleriyle
üretilmiş sentetik konuşmanın gerçek insan sesinden ayırt edilmesi için geleneksel
makine öğrenmesi (Random Forest, SVM) ve derin öğrenme (CNN) modelleri geliştirilip
aynı protokol altında karşılaştırılmaktadır.

## Özet

İki ses öznitelik ailesi ve dört model değerlendirilmektedir:

- **Geleneksel ML:** 194 boyutlu öznitelik vektörü (MFCC istatistikleri, ZCR, spektral
  centroid/bandwidth/rolloff, RMS, chroma) üzerinde Random Forest ve SVM.
- **Derin öğrenme:** Mel-spektrogram ve standart spektrogram girdileri üzerinde,
  residual bloklar içeren bir CNN.

Modeller iki değerlendirme protokolüyle ölçülür: havuzlanmış 5 katlı çapraz doğrulama
(konuşmacıya göre GroupKFold) ve standart protokol (train→dev→eval). Ana metrik EER'dir;
sınıf dengesizliği nedeniyle dengeli doğruluk ve AUC de raporlanır.

## Sonuçlar (ASVspoof 2019 LA, EER)

| Yöntem | Havuzlanmış k-fold | Standart protokol (eval) |
|---|---|---|
| Random Forest | 0.0902 | 0.1437 |
| SVM | 0.0454 | 0.1306 |
| CNN (Mel) | 0.0063 | 0.0453 |
| CNN (Spec) | 0.0041 | 0.1146 |

Ayrıntılı tablolar: [`asvspoof_results.md`](asvspoof_results.md).

## Klasör yapısı

```
.
├── app.py                          # Streamlit arayüzü (ses yükle → tahmin)
├── asvspoof_extract_features.py    # ASVspoof öznitelik çıkarımı (ML / mel / spec)
├── asvspoof_train_ml.py            # RF + SVM eğitimi/değerlendirmesi
├── asvspoof_train_cnn.py           # Mel-spektrogram CNN
├── asvspoof_train_spectrogram_cnn.py  # Standart spektrogram CNN
├── asvspoof_cnn_common.py          # CNN ortak altyapısı
├── asvspoof_results.py             # Sonuç kayıt/özet modülü
├── extract_*_features.py           # FoR öznitelik çıkarımı
├── train_*.py                      # FoR eğitim scriptleri
├── experiments/                    # Yan analizler
├── models/                         # Eğitilmiş modeller (yalnız arayüzün kullandıkları)
├── requirements.txt                # Eğitim/çıkarım bağımlılıkları
└── requirements_gui.txt            # Arayüz bağımlılıkları
```

Büyük dosyalar (veri setleri, çıkarılan `.npy` öznitelikleri, sanal ortam) depoya
dâhil değildir; bkz. `.gitignore`.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Veri setleri

- **ASVspoof 2019 LA:** [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)
  (`LA.zip`). Açıldıktan sonra `Dataset/ASVspoof2019_LA/` altına yerleştirilir.
- **The Fake-or-Real (FoR):** `Dataset/for-original/` altına yerleştirilir.

## Çalıştırma

Öznitelik çıkarımı ve eğitim:

```bash
# ASVspoof
python asvspoof_extract_features.py --ml-only     # veya --mel-only / --spec-only
python asvspoof_train_ml.py
python asvspoof_train_cnn.py
python asvspoof_train_spectrogram_cnn.py
```

Arayüz:

```bash
pip install -r requirements_gui.txt
streamlit run app.py
```

Arayüzde model seçilir, ses dosyası (`.wav/.mp3/.flac`) yüklenir; sistem gerçek/sentetik
kararını, güven skorunu ve dalga formu / mel-spektrogram görselini gösterir.
