"""
Makine öğrenmesi (RF + SVM) K-fold eğitimi ve final test raporu.

Çıktılar:
  - train_ml.log : Tüm stdout (fold satırları + final tablo) konsola ve bu dosyaya yazılır.
  - models/random_forest_model.pkl, random_forest_scaler.pkl
  - models/svm_model.pkl, svm_scaler.pkl
  - ml_kfold_confusion_matrices.png (karmaşıklık matrisleri)
"""

import sys
import atexit
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(_BASE_DIR, "features")
LOGS_DIR = os.path.join(_BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(_BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
_MPL_DIR = os.path.join(_BASE_DIR, ".mplconfig")
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    _venv_py = os.path.join(_BASE_DIR, ".venv", "bin", "python")
    print(
        "HATA: matplotlib yüklü değil (muhtemelen sistem Python'u kullanılıyor).\n"
        f"  Şu anki Python: {sys.executable}\n"
        "  Çözüm — terminalde:\n"
        f"    cd {_BASE_DIR}\n"
        "    source .venv/bin/activate\n"
        "    pip install -r requirements.txt\n"
        "    python train_ml_models.py\n"
        "  Cursor: Ctrl+Shift+P → 'Python: Select Interpreter' → .venv/bin/python",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

LOG_FILE = os.path.join(LOGS_DIR, "train_ml.log")

class _TeeIO:
    """stdout'u konsola ve train_ml.log dosyasına çift yazar."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            try:
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

_orig_stdout = sys.stdout
_log_fp = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = _TeeIO(_orig_stdout, _log_fp)

def _close_ml_training_log():
    sys.stdout = _orig_stdout
    try:
        _log_fp.close()
    except Exception:
        pass

atexit.register(_close_ml_training_log)

import joblib
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
    return float(eer)


# 1. Tüm Klasörlerdeki Verileri Yükle ve TEK BİR HAVUZDA Birleştir
print("Tüm veriler (Train + Val + Test) yüklenip birleştiriliyor...")
x1 = np.load(os.path.join(FEATURES_DIR, "EMF0-x_training_mfcc.npy"))
y1 = np.load(os.path.join(FEATURES_DIR, "EMF0-y_training_labels.npy"))

x2 = np.load(os.path.join(FEATURES_DIR, "EMF1-x_validation_mfcc.npy"))
y2 = np.load(os.path.join(FEATURES_DIR, "EMF1-y_validation_labels.npy"))

x3 = np.load(os.path.join(FEATURES_DIR, "EMF2-x_testing_mfcc.npy"))
y3 = np.load(os.path.join(FEATURES_DIR, "EMF2-y_testing_labels.npy"))

X_TOTAL = np.concatenate((x1, x2, x3), axis=0)
Y_TOTAL = np.concatenate((y1, y2, y3), axis=0)

groups1 = np.load(os.path.join(FEATURES_DIR, "EMF_groups_training.npy"))
groups2 = np.load(os.path.join(FEATURES_DIR, "EMF_groups_validation.npy"))
groups3 = np.load(os.path.join(FEATURES_DIR, "EMF_groups_testing.npy"))

if len(groups1) != len(x1):
    raise ValueError(
        f"EMF_groups_training.npy uzunluğu eğitimin özellik sayısıyla eşleşmiyor: "
        f"groups={len(groups1)} vs x={len(x1)}"
    )
if len(groups2) != len(x2):
    raise ValueError(
        f"EMF_groups_validation.npy uzunluğu doğrulama verisiyle eşleşmiyor: "
        f"groups={len(groups2)} vs x={len(x2)}"
    )
if len(groups3) != len(x3):
    raise ValueError(
        f"EMF_groups_testing.npy uzunluğu test verisiyle eşleşmiyor: "
        f"groups={len(groups3)} vs x={len(x3)}"
    )

GROUPS_TOTAL = np.concatenate((groups1, groups2, groups3), axis=0)

print(f"Toplam Birleşik Veri Seti Boyutu: {X_TOTAL.shape[0]} örnek")

# 2. K-Fold Yapılandırması (5 Parça)
K_FOLDS = 5
gkf = GroupKFold(n_splits=K_FOLDS)

def run_kfold_training(model_obj, model_name):
    print(f"\n>>> {model_name} için {K_FOLDS}-Fold Eğitim Başladı (GroupKFold)...")

    fold_test_accs = []
    fold_test_eers = []

    # Tüm veri setinin K-Fold sonrası birleştirilmiş test sonuçlarını tutacak diziler
    oof_preds = np.zeros(len(Y_TOTAL))
    oof_probs = np.zeros(len(Y_TOTAL))

    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(X_TOTAL, Y_TOTAL, groups=GROUPS_TOTAL)):
        # 1. Adım: Tüm verinin %20'si bu fold için TEST setidir.
        x_test = X_TOTAL[test_idx]
        y_test = Y_TOTAL[test_idx]
        
        # Geriye kalan 4 parça (%80) Train+Val havuzudur.
        x_train_val = X_TOTAL[train_val_idx]
        y_train_val = Y_TOTAL[train_val_idx]
        groups_train_val = GROUPS_TOTAL[train_val_idx]

        # 2. Adım: Geriye kalan 4 parçayı %80 Train, %20 Val olarak böl (sızıntı olmaması için yine gruplu split)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(x_train_val, y_train_val, groups=groups_train_val))
        
        x_train = x_train_val[train_idx]
        y_train = y_train_val[train_idx]
        x_val = x_train_val[val_idx]
        y_val = y_train_val[val_idx]

        # 3. Adım: Standardizasyon (Sızıntıyı önlemek için SADECE Train setinden fit edilir)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled   = scaler.transform(x_val)
        x_test_scaled  = scaler.transform(x_test)

        # 4. Adım: Modeli Eğit (Sadece Train verisiyle)
        model_obj.fit(x_train_scaled, y_train)

        # 5. Adım: Validation ve Test sonuçlarını al
        val_probs = model_obj.predict_proba(x_val_scaled)[:, 1]
        val_preds = model_obj.predict(x_val_scaled)
        val_acc = accuracy_score(y_val, val_preds)

        test_probs = model_obj.predict_proba(x_test_scaled)[:, 1]
        test_preds = model_obj.predict(x_test_scaled)
        test_acc = accuracy_score(y_test, test_preds)
        test_eer = compute_eer(y_test, test_probs)

        fold_test_accs.append(test_acc)
        fold_test_eers.append(test_eer)

        # Test edilen parça sonuçlarını genel havuzdaki indekslerine kaydet
        oof_preds[test_idx] = test_preds
        oof_probs[test_idx] = test_probs

        print(
            f"  Fold {fold + 1}/{K_FOLDS} | "
            f"Veri Dağılımı: [Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}] | "
            f"Val ACC: {val_acc:.4f} -> Test ACC: {test_acc:.4f}, Test EER: {test_eer:.4f}"
        )

    print(f"  {model_name} K-Fold Ortalama Test ACC: {np.mean(fold_test_accs):.4f} ± {np.std(fold_test_accs):.4f}")
    print(f"  {model_name} K-Fold Ortalama Test EER: {np.mean(fold_test_eers):.4f} ± {np.std(fold_test_eers):.4f}")

    # İleride dışarıdan tekil ses dosyalarıyla test edebilmeniz için
    # tüm verilerle nihai (production) modelin eğitilip kaydedilmesi:
    print(f"  Nihai {model_name} modeli diske kaydediliyor...")
    final_scaler = StandardScaler()
    X_TOTAL_scaled = final_scaler.fit_transform(X_TOTAL)
    model_obj.fit(X_TOTAL_scaled, Y_TOTAL)

    joblib.dump(
        model_obj,
        os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_model.pkl"),
    )
    joblib.dump(
        final_scaler,
        os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_scaler.pkl"),
    )

    return oof_preds, oof_probs


# Modelleri Çalıştır
rf_final_preds, rf_final_probs = run_kfold_training(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Random Forest",
)

svm_final_preds, svm_final_probs = run_kfold_training(
    SVC(kernel="rbf", probability=True, random_state=42),
    "SVM",
)

def get_stats(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred), 4),
        "EER": round(compute_eer(y_true, y_prob), 4),
    }

final_results = {
    "Random Forest": get_stats(Y_TOTAL, rf_final_preds, rf_final_probs),
    "SVM": get_stats(Y_TOTAL, svm_final_preds, svm_final_probs),
}

print("\n" + "=" * 65)
print("--- TÜM VERİSETİ K-FOLD FİNAL TEST PERFORMANSI ---")
print("=" * 65)
print(pd.DataFrame(final_results).T.to_string())
print("=" * 65)

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ConfusionMatrixDisplay.from_predictions(
        Y_TOTAL,
        rf_final_preds,
        display_labels=["Sahte (0)", "Gerçek (1)"],
        cmap=plt.cm.Blues,
        ax=axes[0],
    )
    axes[0].set_title("Random Forest (Tüm K-Fold Sonuçları)")

    ConfusionMatrixDisplay.from_predictions(
        Y_TOTAL,
        svm_final_preds,
        display_labels=["Sahte (0)", "Gerçek (1)"],
        cmap=plt.cm.Oranges,
        ax=axes[1],
    )
    axes[1].set_title("SVM (Tüm K-Fold Sonuçları)")

    plt.tight_layout()
    _cm_path = os.path.join(FIGURES_DIR, "ml_kfold_confusion_matrices.png")
    plt.savefig(_cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nKarmaşıklık matrisi grafiği kaydedildi: {_cm_path}")
except Exception as _plot_err:
    print(f"\nUyarı: ML karmaşıklık grafiği kaydedilemedi ({type(_plot_err).__name__}: {_plot_err})")