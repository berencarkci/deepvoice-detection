"""
ASVspoof 2019 LA — Standart Spektrogram CNN eğitimi ve değerlendirmesi.

Ortak eğitim/değerlendirme altyapısı asvspoof_cnn_common.py içinde.

Girdi (asvspoof_extract_features.py --spec-only ile üretilir):
  ASV_spec_x_{train,dev,eval}.npy (257×216), ASV_spec_y_*

Değerlendirme: saldırı-gruplu 5 katlı çapraz doğrulama (görülmemiş saldırı türleri);
ardından tüm havuzla arayüz için nihai model eğitilir.

Çıktı:
  logs/asvspoof_train_cnn_spec.log
  models/asvspoof_cnn_spec_fold{1..5}.pth, ..._final.pth (+ ..._final.norm.json)
  results/asvspoof_results.csv (+md)

Çalıştırma:
  python asvspoof/asvspoof_train_spectrogram_cnn.py
  ASV_SKIP_FINAL=1 python asvspoof/asvspoof_train_spectrogram_cnn.py     # sadece çapraz doğrulama
"""

from asvspoof_cnn_common import run_experiment

if __name__ == "__main__":
    run_experiment({
        "name": "spec",
        "result_name": "Spec-CNN",
        "title": "Standart Spektrogram CNN",
        "x_prefix": "ASV_spec",
        "freq_mask": 30,                       # SpecAugment frekans maskesi genişliği
        "log_file": "asvspoof_train_cnn_spec.log",
        "model_tag": "cnn_spec",
        "batch_env": "ASV_SPEC_BATCH_SIZE",
    })
