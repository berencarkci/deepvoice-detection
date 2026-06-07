"""
ASVspoof 2019 LA — Standart Spektrogram CNN eğitimi ve değerlendirmesi.

Ortak eğitim/değerlendirme altyapısı asvspoof_cnn_common.py içinde.

Girdi (asvspoof_extract_features.py --spec-only ile üretilir):
  ASV_spec_x_{train,dev,eval}.npy (257×216), ASV_spec_y_*, ASV_spec_groups_*

Çıktı:
  asvspoof_train_cnn_spec.log
  models/asvspoof_cnn_spec_fold{1..5}.pth, ..._official.pth (+ ..._final.pth opsiyonel)
  asvspoof_cnn_spec_kfold_roc_confusion.png, asvspoof_cnn_spec_kfold_accuracy_bars.png

Çalıştırma:
  python asvspoof_train_spectrogram_cnn.py
  ASV_SKIP_OFFICIAL=1 python asvspoof_train_spectrogram_cnn.py     # sadece k-fold
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
        "plot_prefix": "asvspoof_cnn_spec_kfold",
        "batch_env": "ASV_SPEC_BATCH_SIZE",
    })
