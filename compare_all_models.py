"""
=============================================================================
  DeepVoice / Deepfake Ses Tespiti — Birleşik Model Karşılaştırma Raporu
=============================================================================
  Amaç:
    Eğitilmiş tüm sınıflandırıcıları (RF, SVM, CNN-Mel, CNN-Spec) AYNI test
    seti üzerinde değerlendirmek ve sonuçları yan yana kıyaslamak.

  Üretilen Çıktılar:
    1. comparison_report.csv         — Tüm metriklerin DataFrame tablosu
    2. comparison_confmats.png       — 2x2 confusion matrix gridi
    3. comparison_roc_curves.png     — Birleşik ROC eğrileri (4 model)

  ML model dosyaları (train_ml_models.py — K-fold sonrası final):
    - models/random_forest_model.pkl + models/random_forest_scaler.pkl
    - models/svm_model.pkl           + models/svm_scaler.pkl
    Eski adlar (rf_model.pkl + scaler.pkl) varsa onlar da otomatik kullanılır.

  CNN ağırlıkları:
    - Önce models/deepvoice_cnn_best.pth / deepvoice_spec_cnn_best.pth
    - Yoksa train_*_cnn.py K-fold çıktısı: deepvoice_*_fold{N}_best.pth (en yüksek N)

  Notlar:
    - EER, scipy.optimize.brentq ile interpolasyonlu olarak hesaplanır
      (ASVspoof literatürü standardı).
    - AUC, sklearn.metrics.roc_auc_score ile hesaplanır.
    - Eksik model dosyaları varsa uyarı verilir ve atlanır.
=============================================================================
"""

import glob
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=UserWarning)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 0. DONANIM
# ─────────────────────────────────────────────
device = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)
print(f"Kullanılan donanım birimi: {device}\n")

# CUDA'da inference hızlandırma; MPS/CPU'da etkisiz.
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

MODELS_DIR = "models"


def resolve_rf_paths():
    """train_ml_models.py (K-fold) çıktıları veya eski tek scaler düzeni."""
    new_m = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    new_s = os.path.join(MODELS_DIR, "random_forest_scaler.pkl")
    old_m = os.path.join(MODELS_DIR, "rf_model.pkl")
    old_s = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.isfile(new_m) and os.path.isfile(new_s):
        return new_m, new_s
    if os.path.isfile(old_m) and os.path.isfile(old_s):
        return old_m, old_s
    return None, None


def resolve_svm_paths():
    """Her model kendi scaler'ı ile (yeni); eski düzende svm + ortak scaler.pkl."""
    new_m = os.path.join(MODELS_DIR, "svm_model.pkl")
    new_s = os.path.join(MODELS_DIR, "svm_scaler.pkl")
    old_s = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.isfile(new_m) and os.path.isfile(new_s):
        return new_m, new_s
    if os.path.isfile(new_m) and os.path.isfile(old_s):
        return new_m, old_s
    return None, None


def resolve_cnn_checkpoint(prefix: str):
    """
    prefix: 'deepvoice_cnn' veya 'deepvoice_spec_cnn'.
    Önce *_best.pth; yoksa *_fold{N}_best.pth içinden en büyük N.
    """
    best = os.path.join(MODELS_DIR, f"{prefix}_best.pth")
    if os.path.isfile(best):
        return best
    pattern = os.path.join(MODELS_DIR, f"{prefix}_fold*_best.pth")
    paths = glob.glob(pattern)

    def fold_num(p):
        m = re.search(r"fold(\d+)_best", os.path.basename(p))
        return int(m.group(1)) if m else -1

    if paths:
        return max(paths, key=fold_num)
    return None


# ─────────────────────────────────────────────
# 1. METRİK YARDIMCILARI
# ─────────────────────────────────────────────
def compute_eer(y_true, y_score):
    """
    Equal Error Rate (EER) — ASVspoof literatürü ile uyumlu,
    scipy.optimize.brentq ile interpolasyonlu hesaplama.
    fpr ve fnr eğrilerinin kesişim noktasıdır.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        # brentq başarısız olursa argmin yedek yöntemine düş
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return float(eer)


def compute_metrics(model_name, y_true, y_pred, y_prob):
    """Tüm performans metriklerini tek bir sözlükte döndürür."""
    return {
        "Model"     : model_name,
        "Accuracy"  : round(accuracy_score (y_true, y_pred), 4),
        "Precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"    : round(recall_score   (y_true, y_pred, zero_division=0), 4),
        "F1-Score"  : round(f1_score       (y_true, y_pred, zero_division=0), 4),
        "AUC"       : round(roc_auc_score  (y_true, y_prob), 4),
        "EER"       : round(compute_eer    (y_true, y_prob), 4),
    }


# ─────────────────────────────────────────────
# 2. CNN MİMARİSİ — train_cnn_model.py ile aynı
#    (AdaptiveAvgPool2d sayesinde hem mel hem spec girişlerle çalışır)
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.block(x) + x)


class DeepVoiceCNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()

        def conv_bn_lrelu(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.stem = nn.Sequential(
            conv_bn_lrelu(1, 32),
            nn.MaxPool2d(2, 2),
        )
        self.stage1 = nn.Sequential(
            conv_bn_lrelu(32, 64),
            ResBlock(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        self.stage2 = nn.Sequential(
            conv_bn_lrelu(64, 128),
            ResBlock(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        self.stage3 = nn.Sequential(
            conv_bn_lrelu(128, 256),
            ResBlock(256),
            nn.AdaptiveAvgPool2d((4, 9)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 9, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# 3. CNN INFERENCE FONKSİYONU
# ─────────────────────────────────────────────
def cnn_inference(model_path, x_test_path, y_test_path, train_x_path,
                  batch_size=64):
    """
    Pre-extracted .npy test setleri üzerinde CNN inference yapar.
    Global Z-score normalizasyonu için training istatistikleri tekrar hesaplanır
    (memory-mapped okumayla, RAM-dostu).
    """
    print(f"  Training istatistikleri hesaplanıyor (mmap)...")
    train_arr = np.load(train_x_path, mmap_mode="r")
    mean = float(train_arr.mean())
    std  = float(train_arr.std())
    del train_arr

    print(f"    mean={mean:.4f} | std={std:.4f}")

    print(f"  Test seti yükleniyor...")
    x_test = np.load(x_test_path).astype(np.float32)
    y_test = np.load(y_test_path).astype(np.int8).flatten()

    # Z-score normalizasyon
    x_test = (x_test - mean) / (std + 1e-8)

    # (N, H, W) → (N, 1, H, W)
    x_test = np.expand_dims(x_test, axis=1)

    print(f"  Model yükleniyor: {model_path}")
    model = DeepVoiceCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"  Inference yapılıyor...")
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch = torch.tensor(x_test[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(batch).cpu().numpy()
            all_logits.append(logits)

    all_logits = np.concatenate(all_logits, axis=0).flatten()
    all_probs  = 1 / (1 + np.exp(-all_logits))   # sigmoid
    all_preds  = (all_probs > 0.5).astype(int)

    return y_test, all_preds, all_probs


# ─────────────────────────────────────────────
# 4. ML INFERENCE FONKSİYONU
# ─────────────────────────────────────────────
def ml_inference(model_path, scaler_path, x_test_path, y_test_path):
    """RF veya SVM inference."""
    print(f"  Scaler yükleniyor: {scaler_path}")
    scaler = joblib.load(scaler_path)

    print(f"  Model yükleniyor: {model_path}")
    model = joblib.load(model_path)

    print(f"  Test seti yükleniyor...")
    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path).flatten()

    x_test_scaled = scaler.transform(x_test)
    preds = model.predict(x_test_scaled)
    probs = model.predict_proba(x_test_scaled)[:, 1]

    return y_test, preds, probs


# ─────────────────────────────────────────────
# 5. MODELLER ÜZERİNDE DEĞERLENDİRME
# ─────────────────────────────────────────────
results       = []
predictions   = {}   # model_name → (y_true, y_pred, y_prob)

# — Random Forest —
print("=" * 65)
print("[1/4] Random Forest değerlendiriliyor...")
print("=" * 65)
rf_path, rf_scaler_path = resolve_rf_paths()
if rf_path and rf_scaler_path:
    print(f"  Model: {rf_path}")
    print(f"  Scaler: {rf_scaler_path}")
    y_t, y_p, y_pr = ml_inference(rf_path, rf_scaler_path,
                                  "EMF2-x_testing_mfcc.npy",
                                  "EMF2-y_testing_labels.npy")
    results.append(compute_metrics("Random Forest", y_t, y_p, y_pr))
    predictions["Random Forest"] = (y_t, y_p, y_pr)
    print(f"  ✔ Tamam | Accuracy={results[-1]['Accuracy']:.4f}")
else:
    print("  ⚠ Atlandı (random_forest_model.pkl + random_forest_scaler.pkl "
          "veya rf_model.pkl + scaler.pkl bulunamadı)")

# — SVM —
print("\n" + "=" * 65)
print("[2/4] SVM değerlendiriliyor...")
print("=" * 65)
svm_path, svm_scaler_path = resolve_svm_paths()
if svm_path and svm_scaler_path:
    print(f"  Model: {svm_path}")
    print(f"  Scaler: {svm_scaler_path}")
    y_t, y_p, y_pr = ml_inference(svm_path, svm_scaler_path,
                                  "EMF2-x_testing_mfcc.npy",
                                  "EMF2-y_testing_labels.npy")
    results.append(compute_metrics("SVM", y_t, y_p, y_pr))
    predictions["SVM"] = (y_t, y_p, y_pr)
    print(f"  ✔ Tamam | Accuracy={results[-1]['Accuracy']:.4f}")
else:
    print("  ⚠ Atlandı (svm_model.pkl + svm_scaler.pkl veya svm_model.pkl + scaler.pkl "
          "bulunamadı)")

# — CNN (Mel-Spektrogram) —
print("\n" + "=" * 65)
print("[3/4] CNN-Mel değerlendiriliyor...")
print("=" * 65)
mel_cnn_path = resolve_cnn_checkpoint("deepvoice_cnn")
if mel_cnn_path and os.path.exists("ECF0-x_training_mel.npy"):
    print(f"  Ağırlık dosyası: {mel_cnn_path}")
    y_t, y_p, y_pr = cnn_inference(
        mel_cnn_path,
        "ECF2-x_testing_mel.npy",
        "ECF2-y_testing_mel.npy",
        "ECF0-x_training_mel.npy",
    )
    results.append(compute_metrics("CNN-Mel", y_t, y_p, y_pr))
    predictions["CNN-Mel"] = (y_t, y_p, y_pr)
    print(f"  ✔ Tamam | Accuracy={results[-1]['Accuracy']:.4f}")
else:
    print("  ⚠ Atlandı (deepvoice_cnn_best.pth / deepvoice_cnn_fold*_best.pth "
          "veya ECF0-x_training_mel.npy bulunamadı)")

# — CNN (Standart Spektrogram) —
print("\n" + "=" * 65)
print("[4/4] CNN-Spec değerlendiriliyor...")
print("=" * 65)
spec_cnn_path = resolve_cnn_checkpoint("deepvoice_spec_cnn")
if spec_cnn_path and os.path.exists("ESF0-x_training_spec.npy"):
    print(f"  Ağırlık dosyası: {spec_cnn_path}")
    y_t, y_p, y_pr = cnn_inference(
        spec_cnn_path,
        "ESF2-x_testing_spec.npy",
        "ESF2-y_testing_spec.npy",
        "ESF0-x_training_spec.npy",
    )
    results.append(compute_metrics("CNN-Spec", y_t, y_p, y_pr))
    predictions["CNN-Spec"] = (y_t, y_p, y_pr)
    print(f"  ✔ Tamam | Accuracy={results[-1]['Accuracy']:.4f}")
else:
    print("  ⚠ Atlandı (deepvoice_spec_cnn_best.pth / deepvoice_spec_cnn_fold*_best.pth "
          "veya ESF0-x_training_spec.npy bulunamadı)")

# ─────────────────────────────────────────────
# 6. RAPOR — Pandas DataFrame
# ─────────────────────────────────────────────
if not results:
    print("\n❌ Hiçbir model değerlendirilemedi. Önce modelleri eğitin.")
    raise SystemExit(1)

df = pd.DataFrame(results).set_index("Model")

print("\n\n" + "=" * 75)
print("        BİRLEŞİK MODEL KARŞILAŞTIRMA RAPORU (FoR Test Seti)")
print("=" * 75)
print(df.to_string())
print("=" * 75)

# CSV kaydı
df.to_csv("comparison_report.csv", index=True)
print("\n✓ comparison_report.csv kaydedildi.")

# ─────────────────────────────────────────────
# 7. CONFUSION MATRIX GRİDİ
# ─────────────────────────────────────────────
n_models  = len(predictions)
n_cols    = 2
n_rows    = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
axes      = np.array(axes).flatten()

cmaps = ["Blues", "Oranges", "Purples", "Greens", "Reds"]

for ax, ((name, (y_t, y_p, _)), cmap) in zip(axes, zip(predictions.items(), cmaps)):
    cm   = confusion_matrix(y_t, y_p)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Sahte (0)", "Gerçek (1)"])
    disp.plot(ax=ax, cmap=cmap, values_format="d", colorbar=False)
    acc = accuracy_score(y_t, y_p)
    ax.set_title(f"{name}\nAccuracy: {acc * 100:.2f}%")

# Boş kalan eksenleri gizle
for ax in axes[n_models:]:
    ax.axis("off")

fig.suptitle("Tüm Modeller — Confusion Matrix Karşılaştırması",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "comparison_confmats.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✓ figures/comparison_confmats.png kaydedildi.")

# ─────────────────────────────────────────────
# 8. BİRLEŞİK ROC EĞRİSİ
# ─────────────────────────────────────────────
plt.figure(figsize=(9, 8))
colors = ["royalblue", "darkorange", "darkorchid", "forestgreen", "crimson"]

for (name, (y_t, _, y_pr)), color in zip(predictions.items(), colors):
    fpr, tpr, _ = roc_curve(y_t, y_pr)
    auc_val     = roc_auc_score(y_t, y_pr)
    eer_val     = compute_eer(y_t, y_pr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name}  (AUC={auc_val:.4f}, EER={eer_val:.4f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Rastgele tahmin")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Birleşik ROC Eğrileri — Tüm Modeller", fontsize=13, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "comparison_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✓ figures/comparison_roc_curves.png kaydedildi.")

# ─────────────────────────────────────────────
# 9. EN İYİ MODELİ BELİRLE (F1 ve EER üzerinden)
# ─────────────────────────────────────────────
best_f1  = df["F1-Score"].idxmax()
best_eer = df["EER"     ].idxmin()

print("\n" + "=" * 65)
print("                        ÖZET YORUM")
print("=" * 65)
print(f"  En yüksek F1-Score : {best_f1}  ({df.loc[best_f1, 'F1-Score']:.4f})")
print(f"  En düşük EER       : {best_eer} ({df.loc[best_eer, 'EER']:.4f})")
print("=" * 65)
