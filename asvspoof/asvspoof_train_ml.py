"""
ASVspoof 2019 LA — geleneksel ML modelleri (Random Forest + SVM) eğitimi ve değerlendirmesi.

Giriş (.npy — asvspoof_extract_features.py tarafından üretilir):
  ASV_ml_x_{train,dev,eval}.npy, ASV_ml_y_*  (train+dev+eval havuzlanır)

Değerlendirme: saldırı-gruplu 5 katlı çapraz doğrulama. Grup ataması spoof→saldırı türü,
bonafide→konuşmacı; böylece her fold'un test seti eğitimde yer almayan saldırılardan oluşur
ve görülmemiş sentez sistemlerine genelleme ölçülür. SVM (RBF) varsayılan olarak tüm eğitim
verisiyle çalışır; çok yavaşsa ASV_SVM_KFOLD_CAP=N ile eğitim alt-örneklemi sınırlanabilir
(test fold'u tam kalır). SVM skoru decision_function ile alınır.

Sınıf dengesizliği (~1:9) için class_weight="balanced"; EER ve dengeli doğruluk accuracy
ile birlikte raporlanır.

Çıktılar:
  logs/asvspoof_train_ml.log
  models/asvspoof_random_forest_model.pkl (+scaler), models/asvspoof_svm_model.pkl (+scaler)
    — arayüz için tüm havuzla eğitilmiş nihai modeller
  results/asvspoof_results.csv (+md) — saldırı-gruplu tam metrik tablosu
"""

import sys
import atexit
import os
import time

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(_BASE_DIR, "features")
LOGS_DIR = os.path.join(_BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "asvspoof_train_ml.log")


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
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, balanced_accuracy_score,
)

from asvspoof_extract_features import build_attack_groups
from asvspoof_results import record_result

MODELS_DIR = os.path.join(_BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
K_FOLDS = 5
SPLITS = ["train", "dev", "eval"]
PROTO = "attack_kfold"


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
        "acc": accuracy_score(y_true, preds),
        "balacc": balanced_accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auc": roc_auc_score(y_true, scores),
        "eer": compute_eer(y_true, scores),
    }


# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME (havuz) + saldırı grupları
# ─────────────────────────────────────────────
print("ASVspoof 2019 LA — ML Model Eğitimi (saldırı-gruplu)")
print("=" * 65)
print("\nVeriler yükleniyor...")

xs, ys = [], []
for s in SPLITS:
    xs.append(np.load(os.path.join(FEATURES_DIR, f"ASV_ml_x_{s}.npy")))
    ys.append(np.load(os.path.join(FEATURES_DIR, f"ASV_ml_y_{s}.npy")))
X = np.concatenate(xs)
Y = np.concatenate(ys).astype(np.int64)

G, Y_derived = build_attack_groups()
if len(Y) != len(Y_derived) or not np.array_equal(Y, Y_derived):
    raise SystemExit("❌ HİZALAMA HATASI: türetilen saldırı etiketleri kayıtlı y ile eşleşmiyor.")

n_attack = len([g for g in set(G) if not g.startswith("bona_")])
print(f"  HAVUZ : {X.shape[0]} örnek × {X.shape[1]} öznitelik | "
      f"bonafide={int((Y==1).sum())}, spoof={int((Y==0).sum())}")
print(f"  Gruplar: {n_attack} saldırı grubu + {len(set(G)) - n_attack} bonafide-konuşmacı grubu")


def _stratified_subsample(idx, cap, seed=42):
    """idx'ten sınıf oranını koruyarak en çok `cap` örnek seçer (RBF SVM'i tractable yapmak için)."""
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(Y[idx]):
        ci = idx[Y[idx] == c]
        k = max(1, int(round(cap * len(ci) / len(idx))))
        out.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.concatenate(out)


# ─────────────────────────────────────────────
# 2. SALDIRI-GRUPLU 5-FOLD ÇAPRAZ DOĞRULAMA
# ─────────────────────────────────────────────
def run_attack_kfold():
    _cap_env = os.environ.get("ASV_SVM_KFOLD_CAP")
    svm_cap = int(_cap_env) if _cap_env else None

    configs = [
        ("RF",
         lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                        class_weight="balanced"),
         "proba", None),
        ("SVM",
         lambda: SVC(kernel="rbf", random_state=42, class_weight="balanced", cache_size=1000),
         "decision", svm_cap),
    ]

    gkf = GroupKFold(n_splits=K_FOLDS)
    for name, make, kind, cap in configs:
        note = "[tüm veri]" if cap is None else f"[eğitim ≤{cap}]"
        print(f"\n{'='*70}\n  {name} — SALDIRI-GRUPLU {K_FOLDS}-FOLD  {note}\n{'='*70}")
        rows = []
        for fold, (tr, te) in enumerate(gkf.split(X, Y, groups=G)):
            t0 = time.time()
            fit = tr if cap is None else _stratified_subsample(tr, cap)
            scaler = StandardScaler().fit(X[fit])
            clf = make().fit(scaler.transform(X[fit]), Y[fit])
            Xte = scaler.transform(X[te])
            if kind == "proba":
                score, thr = clf.predict_proba(Xte)[:, 1], 0.5
            else:
                score, thr = clf.decision_function(Xte), 0.0
            m = metrics(Y[te], score, thr)
            rows.append(m)
            test_atks = sorted(set(G[te][Y[te] == 0]))
            print(f"  Fold {fold+1}/{K_FOLDS} | Train {len(fit)} / Test {len(te)} | "
                  f"ACC {m['acc']:.4f} BalACC {m['balacc']:.4f} EER {m['eer']:.4f} "
                  f"| görülmemiş saldırı: {test_atks} ({time.time()-t0:.0f}s)")

        print(f"\n  --- {name} {K_FOLDS}-fold ortalama ---")
        for key, lbl in [("acc", "Accuracy"), ("balacc", "Bal.Accuracy"), ("precision", "Precision"),
                         ("recall", "Recall"), ("f1", "F1-Score"), ("auc", "AUC"), ("eer", "EER")]:
            v = [r[key] for r in rows]
            print(f"      {lbl:<13}: {np.mean(v):.4f} ± {np.std(v):.4f}")

        def avg(k):
            return float(np.mean([r[k] for r in rows]))
        record_result(
            name, PROTO,
            eer=avg("eer"), balacc=avg("balacc"), recall=avg("recall"), auc=avg("auc"),
            eer_std=float(np.std([r["eer"] for r in rows])),
            acc=avg("acc"), precision=avg("precision"), f1=avg("f1"),
        )


# ─────────────────────────────────────────────
# 3. NİHAİ MODELLER (tüm havuz) — arayüz için
# ─────────────────────────────────────────────
def train_final_models():
    print(f"\n{'='*70}\n  NİHAİ MODELLER — tüm havuzla eğitilir (arayüz için kaydedilir)\n{'='*70}")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    finals = [
        ("random_forest",
         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                class_weight="balanced")),
        ("svm",
         SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")),
    ]
    for tag, clf in finals:
        t0 = time.time()
        clf.fit(Xs, Y)
        joblib.dump(clf, os.path.join(MODELS_DIR, f"asvspoof_{tag}_model.pkl"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"asvspoof_{tag}_scaler.pkl"))
        print(f"  ✔ models/asvspoof_{tag}_model.pkl (+scaler) ({time.time()-t0:.0f}s)")


# ─────────────────────────────────────────────
# ÇALIŞTIR
# ─────────────────────────────────────────────
run_attack_kfold()
train_final_models()
print("\n✓ Tamamlandı. Sonuçlar: results/asvspoof_results.md (attack_kfold)")
