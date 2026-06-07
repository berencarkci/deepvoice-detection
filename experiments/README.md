# Deneysel Scriptler — Proje Kapsamı Dışı

Bu klasör, **The Fake-or-Real (FoR) veri setine** dayanan ana proje akışının
**dışında** kalan deneysel scriptleri içerir. Rapor ve değerlendirme
sonuçlarına dahil edilmemelidir.

## İçerik

### `extract_hf_features.py`
Hugging Face üzerinden **DFADD** veri setinden Mel-Spektrogram çıkarımı yapar.
Streaming modu ile çalışır, tüm seti diske indirmeden işler.

### `train_dfadd_cnn.py`
DFADD veri seti üzerinde eski mimari ile bir CNN eğitir. Şu sebeplerle ana
akıştan ayrıldı:

- DFADD veri seti (proje şartnamesinde **FoR** belirtilmiş)
- Eski CNN mimarisi (`BCELoss + Sigmoid`, sabit `Linear(64*16*27)`, 3 epoch)
- Hardcoded `total_samples = 200465`
- AdaptiveAvgPool2d içermez → giriş boyutu sabittir, taşınabilir değil

> **Not:** Asıl projeyi yansıtmak için lütfen kök dizindeki
> `train_cnn_model.py` (Mel-CNN) ve `train_spectrogram_cnn.py` (Spec-CNN)
> scriptlerini kullanın.
