"""
ASVspoof 2019 LA — geleneksel ML modelleri (Random Forest + SVM) eğitimi ve değerlendirmesi.

Giriş (.npy — asvspoof_extract_features.py tarafından üretilir):
  ASV_ml_x_{train,dev,eval}.npy, ASV_ml_y_*, ASV_ml_groups_*  (groups = konuşmacı ID)

İki değerlendirme uygulanır:
  1) Konuşmacıya göre havuzlanmış 5 katlı çapraz doğrulama (train+dev+eval birleşik);
     hem RF hem SVM. SVM (RBF) varsayılan olarak tüm eğitim verisiyle çalışır; çok yavaşsa
     ASV_SVM_KFOLD_CAP=N ile eğitim alt-örneklemi sınırlanabilir (test fold'u tam kalır).
     SVM skoru decision_function ile alınır.
  2) Standart protokol: train ile eğit, dev ile izle, eval ile test.

Sınıf dengesizliği (~1:9) için class_weight="balanced"; EER ve dengeli doğruluk accuracy
ile birlikte raporlanır.

Çıktılar:
  asvspoof_train_ml.log
  models/asvspoof_random_forest_model.pkl (+scaler), models/asvspoof_svm_model.pkl (+scaler)
  asvspoof_ml_confusion_matrices.png, asvspoof_ml_roc_curves.png
  asvspoof_ml_kfold_roc_confusion.png, asvspoof_ml_report.csv
"""

import sys
import atexit
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MPL_DIR = os.path.join(_BASE_DIR, ".mplconfig")
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("HATA: matplotlib yüklü değil.", file=sys.stderr)
    raise SystemExit(1)

LOG_FILE = os.path.join(_BASE_DIR, "asvspoof_train_ml.log")

class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            try: s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass

_orig_stdout = sys.stdout
_log_fp = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = _TeeIO(_orig_stdout, _log_fp)

def _close_log():
    sys.stdout = _orig_stdout
    try: _log_fp.close()
    except Exception: pass

atexit.register(_close_log)

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, ConfusionMatrixDisplay,
    balanced_accuracy_score,
)

MODELS_DIR = os.path.join(_BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(_BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
K_FOLDS = 5
SPLITS = ["train", "dev", "eval"]


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
    return float(eer)


def metrics(y_true, scores, thr=0.5):
    """scores: olasılık (thr=0.5) ya da decision_function (thr=0.0). EER/AUC eşikten bağımsız."""
    preds = (scores > thr).astype(int)
    return {
        "ACC": accuracy_score(y_true, preds),
        "BalACC": balanced_accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "AUC": roc_auc_score(y_true, scores),
        "EER": compute_eer(y_true, scores),
    }, preds


# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME (+ konuşmacı grupları, havuz)
# ─────────────────────────────────────────────
print("ASVspoof 2019 LA — ML Model Eğitimi")
print("=" * 65)
print("\nVeriler yükleniyor...")

xs, ys, gs, split_id = {}, {}, {}, {}
for k, s in enumerate(SPLITS):
    xs[s] = np.load(os.path.join(_BASE_DIR, f"ASV_ml_x_{s}.npy"))
    ys[s] = np.load(os.path.join(_BASE_DIR, f"ASV_ml_y_{s}.npy"))
    gs[s] = np.load(os.path.join(_BASE_DIR, f"ASV_ml_groups_{s}.npy"))
    print(f"  {s:<6}: {xs[s].shape[0]:>6} örnek  "
          f"(bonafide={int(np.sum(ys[s]==1))}, spoof={int(np.sum(ys[s]==0))})  "
          f"konuşmacı={len(np.unique(gs[s]))}")

x_train, y_train = xs["train"], ys["train"]
x_dev, y_dev = xs["dev"], ys["dev"]
x_eval, y_eval = xs["eval"], ys["eval"]

X_TOTAL = np.concatenate([xs[s] for s in SPLITS])
Y_TOTAL = np.concatenate([ys[s] for s in SPLITS])
G_TOTAL = np.concatenate([gs[s] for s in SPLITS])
print(f"  HAVUZ : {X_TOTAL.shape[0]:>6} örnek  konuşmacı={len(np.unique(G_TOTAL))}")


# ═══════════════════════════════════════════════════════════════
# 2. HAVUZLANMIŞ 5-FOLD GroupKFold (KONUŞMACI) — RF + SVM
# ═══════════════════════════════════════════════════════════════
_cap_env = os.environ.get("ASV_SVM_KFOLD_CAP")
SVM_KFOLD_CAP = int(_cap_env) if _cap_env else None   # None = tüm eğitim verisi kullanılır


def _stratified_subsample(idx, y, cap, seed=42):
    """idx'ten sınıf oranını koruyarak en çok `cap` örnek seçer (RBF SVM'i tractable yapmak için)."""
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        k = max(1, int(round(cap * len(ci) / len(idx))))
        out.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.concatenate(out)


def _kfold_scores(make_estimator, score_kind, subsample_cap=None):
    """Bir model için havuzlanmış GroupKFold çalıştırır; OOF skor + fold metriklerini döner."""
    gkf = GroupKFold(n_splits=K_FOLDS)
    oof = np.zeros(len(Y_TOTAL))
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_TOTAL, Y_TOTAL, groups=G_TOTAL)):
        fit_idx = tr_idx if subsample_cap is None else _stratified_subsample(tr_idx, Y_TOTAL, subsample_cap)
        scaler = StandardScaler().fit(X_TOTAL[fit_idx])
        clf = make_estimator()
        clf.fit(scaler.transform(X_TOTAL[fit_idx]), Y_TOTAL[fit_idx])
        Xte = scaler.transform(X_TOTAL[te_idx])
        if score_kind == "proba":
            score = clf.predict_proba(Xte)[:, 1]
            thr = 0.5
        else:                                  # decision_function (SVM)
            score = clf.decision_function(Xte)
            thr = 0.0
        oof[te_idx] = score
        m, _ = metrics(Y_TOTAL[te_idx], score, thr=thr)
        rows.append(m)
        extra = "" if subsample_cap is None else f" (eğitim alt-örneklem={len(fit_idx)})"
        print(f"  Fold {fold+1}/{K_FOLDS} | Train {len(tr_idx)}{extra} / Test {len(te_idx)} | "
              f"ACC {m['ACC']:.4f} | BalACC {m['BalACC']:.4f} | EER {m['EER']:.4f}")
    return oof, rows


def run_kfold():
    print("\n" + "=" * 70)
    print(f"  1) {K_FOLDS}-FOLD GroupKFold (konuşmacı) — havuzlanmış train+dev+eval")
    print("=" * 70)

    configs = [
        ("Random Forest",
         lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                        class_weight="balanced"),
         "proba", None),
        ("SVM",
         lambda: SVC(kernel="rbf", random_state=42, class_weight="balanced",
                     cache_size=1000),                       # probability YOK; cache büyük
         "decision", SVM_KFOLD_CAP),
    ]
    results = {}
    for name, make, score_kind, cap in configs:
        note = "  [tüm veri]" if cap is None else f"  [eğitim ≤{cap} alt-örneklem]"
        print(f"\n  >>> {name}{note}")
        oof, rows = _kfold_scores(make, score_kind, cap)
        print(f"  --- {name} havuzlanmış {K_FOLDS}-fold ortalama ---")
        for key in ["ACC", "BalACC", "Precision", "Recall", "F1", "AUC", "EER"]:
            v = [r[key] for r in rows]
            print(f"      {key:<10}: {np.mean(v):.4f} ± {np.std(v):.4f}")
        results[name] = {"oof": oof, "rows": rows}
    return results


# ═══════════════════════════════════════════════════════════════
# 3. Standart protokol — train ile eğit, eval ile test
# ═══════════════════════════════════════════════════════════════
def train_and_evaluate(model_obj, model_name):
    print(f"\n{'='*65}\n  {model_name} — STANDART PROTOKOL (train→eğit, eval→test)\n{'='*65}")
    scaler = StandardScaler().fit(x_train)          # scaler yalnızca train'den fit edilir
    model_obj.fit(scaler.transform(x_train), y_train)

    dev_probs = model_obj.predict_proba(scaler.transform(x_dev))[:, 1]
    dev_m, _ = metrics(y_dev, dev_probs)
    print(f"  Dev  → ACC: {dev_m['ACC']:.4f} | EER: {dev_m['EER']:.4f}")

    eval_probs = model_obj.predict_proba(scaler.transform(x_eval))[:, 1]
    eval_m, eval_preds = metrics(y_eval, eval_probs)
    print(f"  Eval → ACC: {eval_m['ACC']:.4f} | BalACC: {eval_m['BalACC']:.4f} | "
          f"Recall: {eval_m['Recall']:.4f} | EER: {eval_m['EER']:.4f}")

    tag = model_name.lower().replace(" ", "_")
    joblib.dump(model_obj, os.path.join(MODELS_DIR, f"asvspoof_{tag}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"asvspoof_{tag}_scaler.pkl"))
    print(f"  ✔ models/asvspoof_{tag}_model.pkl")

    return {
        "Model": model_name,
        "Dev ACC": round(dev_m["ACC"], 4), "Dev EER": round(dev_m["EER"], 4),
        "Eval ACC": round(eval_m["ACC"], 4), "Eval BalACC": round(eval_m["BalACC"], 4),
        "Eval Precision": round(eval_m["Precision"], 4), "Eval Recall": round(eval_m["Recall"], 4),
        "Eval F1": round(eval_m["F1"], 4), "Eval AUC": round(eval_m["AUC"], 4),
        "Eval EER": round(eval_m["EER"], 4),
    }, (y_eval, eval_preds, eval_probs)


# ─────────────────────────────────────────────
# ÇALIŞTIR
# ─────────────────────────────────────────────
kfold_results = run_kfold()

rf_results, rf_preds = train_and_evaluate(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Random Forest",
)
# RBF SVM eğitim süresi örnek sayısıyla O(n²) ölçeklenir; alternatif olarak LinearSVC kullanılabilir.
svm_results, svm_preds = train_and_evaluate(
    SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    "SVM",
)

# ─────────────────────────────────────────────
# RAPOR
# ─────────────────────────────────────────────
df = pd.DataFrame([rf_results, svm_results]).set_index("Model")
print("\n" + "=" * 80)
print("     STANDART PROTOKOL — EVAL SETİ")
print("=" * 80)
print(df.to_string())
print("=" * 80)
df.to_csv(os.path.join(_BASE_DIR, "asvspoof_ml_report.csv"), index=True)
print("✔ asvspoof_ml_report.csv kaydedildi.")

official_eer = {"Random Forest": rf_results["Eval EER"], "SVM": svm_results["Eval EER"]}
print("\n" + "=" * 70)
print("  ÖZET — Havuzlanmış k-fold vs Standart protokol (EER)")
print("=" * 70)
print(f"  {'Model':<16}{'5-fold EER':>14}{'Standart EER':>14}")
for name in ["Random Forest", "SVM"]:
    kf = np.mean([r["EER"] for r in kfold_results[name]["rows"]])
    print(f"  {name:<16}{kf:>14.4f}{official_eer[name]:>14.4f}")
print("=" * 70)

# ─── Sonuçları ortak asvspoof_results.csv/md'ye otomatik kaydet ───
from asvspoof_results import record_result
_name_map = {"Random Forest": "RF", "SVM": "SVM"}
for name, official in [("Random Forest", rf_results), ("SVM", svm_results)]:
    rws = kfold_results[name]["rows"]
    record_result(
        _name_map[name], "kfold_pooled",
        eer=float(np.mean([r["EER"] for r in rws])),
        balacc=float(np.mean([r["BalACC"] for r in rws])),
        recall=float(np.mean([r["Recall"] for r in rws])),
        auc=float(np.mean([r["AUC"] for r in rws])),
        eer_std=float(np.std([r["EER"] for r in rws])),
    )
    record_result(
        _name_map[name], "official_eval",
        eer=official["Eval EER"], balacc=official["Eval BalACC"],
        recall=official["Eval Recall"], auc=official["Eval AUC"],
    )

# ─────────────────────────────────────────────
# GRAFİKLER
# ─────────────────────────────────────────────
try:
    # Resmî protokol — confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ASVspoof 2019 LA — ML Confusion (Resmî Protokol, Eval)",
                 fontsize=13, fontweight="bold")
    for ax, (name, (y_t, y_p, _)), cmap in zip(
        axes, [("Random Forest", rf_preds), ("SVM", svm_preds)], ["Blues", "Oranges"]):
        ConfusionMatrixDisplay.from_predictions(
            y_t, y_p, display_labels=["Spoof (0)", "Bonafide (1)"],
            cmap=cmap, ax=ax, colorbar=False)
        ax.set_title(f"{name}\nACC: {accuracy_score(y_t, y_p)*100:.2f}% | "
                     f"BalACC: {balanced_accuracy_score(y_t, y_p)*100:.2f}%")
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, "asvspoof_ml_confusion_matrices.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"✔ {p}")

    # Resmî protokol — ROC
    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, (y_t, _, y_pr)), color in zip(
        [("Random Forest", rf_preds), ("SVM", svm_preds)], ["royalblue", "darkorange"]):
        fpr, tpr, _ = roc_curve(y_t, y_pr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc_score(y_t, y_pr):.4f}, EER={compute_eer(y_t, y_pr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ASVspoof 2019 LA — ROC (Resmî Protokol, Eval)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, "asvspoof_ml_roc_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"✔ {p}")

    # Havuzlanmış k-fold — OOF ROC (RF+SVM) + confusion (RF, SVM)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.suptitle(f"ASVspoof 2019 LA — Havuzlanmış {K_FOLDS}-Fold (OOF)",
                 fontsize=12, fontweight="bold")
    for name, color in [("Random Forest", "royalblue"), ("SVM", "darkorange")]:
        oof = kfold_results[name]["oof"]
        fpr, tpr, _ = roc_curve(Y_TOTAL, oof)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} (AUC={roc_auc_score(Y_TOTAL, oof):.4f}, "
                           f"EER={compute_eer(Y_TOTAL, oof):.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC (OOF)")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)
    for ax, (name, thr, cmap) in zip(
        axes[1:], [("Random Forest", 0.5, plt.cm.Blues), ("SVM", 0.0, plt.cm.Oranges)]):
        preds = (kfold_results[name]["oof"] > thr).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            Y_TOTAL, preds, display_labels=["Spoof (0)", "Bonafide (1)"],
            cmap=cmap, ax=ax, colorbar=False)
        ax.set_title(f"{name} Confusion (OOF)")
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, "asvspoof_ml_kfold_roc_confusion.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"✔ {p}")

except Exception as e:
    print(f"\nUyarı: Grafik kaydedilemedi ({type(e).__name__}: {e})")
