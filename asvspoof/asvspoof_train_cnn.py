"""
ASVspoof 2019 LA — Mel-Spektrogram CNN eğitimi ve değerlendirmesi.

Ortak eğitim/değerlendirme altyapısı asvspoof_cnn_common.py içinde.

Girdi (asvspoof_extract_features.py --mel-only ile üretilir):
  ASV_mel_x_{train,dev,eval}.npy, ASV_mel_y_*, ASV_mel_groups_*

Çıktı:
  asvspoof_train_cnn_mel.log
  models/asvspoof_cnn_mel_fold{1..5}.pth, ..._official.pth (+ ..._final.pth opsiyonel)
  asvspoof_cnn_mel_kfold_roc_confusion.png, asvspoof_cnn_mel_kfold_accuracy_bars.png

Çalıştırma:
  python asvspoof_train_cnn.py
  ASV_SKIP_OFFICIAL=1 python asvspoof_train_cnn.py     # sadece k-fold
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
        "plot_prefix": "asvspoof_cnn_mel_kfold",
        "batch_env": "ASV_MEL_BATCH_SIZE",
    })
