"""
ASVspoof 2019 LA — Mel-Spektrogram CNN eğitimi ve değerlendirmesi.

Ortak eğitim/değerlendirme altyapısı asvspoof_cnn_common.py içinde.

Girdi (asvspoof_extract_features.py --mel-only ile üretilir):
  ASV_mel_x_{train,dev,eval}.npy, ASV_mel_y_*

Değerlendirme: saldırı-gruplu 5 katlı çapraz doğrulama (görülmemiş saldırı türleri);
ardından tüm havuzla arayüz için nihai model eğitilir.

Çıktı:
  logs/asvspoof_train_cnn_mel.log
  models/asvspoof_cnn_mel_fold{1..5}.pth, ..._final.pth (+ ..._final.norm.json)
  results/asvspoof_results.csv (+md)

Çalıştırma:
  python asvspoof/asvspoof_train_cnn.py
  ASV_SKIP_FINAL=1 python asvspoof/asvspoof_train_cnn.py     # sadece çapraz doğrulama
"""

from asvspoof_cnn_common import run_experiment

if __name__ == "__main__":
    run_experiment({
        "name": "mel",
        "result_name": "Mel-CNN",
        "title": "Mel-Spektrogram CNN",
        "x_prefix": "ASV_mel",
        "freq_mask": 20,                       # SpecAugment frekans maskesi genişliği
        "log_file": "asvspoof_train_cnn_mel.log",
        "model_tag": "cnn_mel",
        "batch_env": "ASV_MEL_BATCH_SIZE",
    })
