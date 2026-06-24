"""
ASVspoof 2019 LA — Saldırı-gruplu (leave-attack-out) 5 katlı çapraz doğrulama.

Havuzlanmış train+dev+eval üzerinde GroupKFold(5) uygulanır; grup ataması:
  * spoof    → saldırı türü (A01..A19)
  * bonafide → konuşmacı (bona_<speaker>)

Böylece her fold'un test seti, eğitimde yer almayan GÖRÜLMEMİŞ saldırılardan oluşur;
modelin görülmemiş sentez sistemlerine genelleme başarımı ölçülür.

Saldırı etiketi öznitelik .npy'lerinde tutulmaz; protokol dosyalarından, öznitelik
çıkarımıyla AYNI sırada yeniden türetilir ve kayıtlı y ile birebir hizalandığı
doğrulanır.

Sonuçlar results/asvspoof_results.csv (+md) tablosuna "attack_kfold" protokolü olarak
yazılır.

Kullanım (proje kökünden):
  python experiments/asvspoof_attack_grouped_kfold.py            # ml + cnn
  python experiments/asvspoof_attack_grouped_kfold.py --ml-only
  python experiments/asvspoof_attack_grouped_kfold.py --cnn-only
  python experiments/asvspoof_attack_grouped_kfold.py --mel-only
  python experiments/asvspoof_attack_grouped_kfold.py --spec-only

Ortam değişkenleri:
  ASV_SVM_KFOLD_CAP=N   SVM eğitim fold'unu N örnekle sınırla (test fold'u tam kalır)
"""

import os
import sys
import time
import argparse

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "asvspoof"))

import numpy as np

import asvspoof_extract_features as ex
from asvspoof_extract_features import parse_protocol, FEATURES_DIR
from asvspoof_results import record_result

SPLITS = ["train", "dev", "eval"]
K = 5
PROTO = "attack_kfold"


# ─────────────────────────────────────────────
# Saldırı-grup dizisi (öznitelik sırasıyla hizalı)
# ─────────────────────────────────────────────
def build_attack_groups():
    """spoof→saldırı, bonafide→bona_<konuşmacı>. (gruplar, türetilen_y) döner."""
    groups, ys = [], []
    for s in SPLITS:
        cfg = ex.SPLITS[s]
        flac = cfg["flac_dir"]
        rows = [(aid, spk, lab, atk)
                for (aid, spk, lab, atk) in parse_protocol(cfg["protocol"])
                if os.path.exists(os.path.join(flac, f"{aid}.flac"))]
        for _aid, spk, lab, atk in rows:
            groups.append(atk if lab == 0 else f"bona_{spk}")
            ys.append(lab)
    return np.array(groups), np.array(ys, dtype=np.int64)


def _load_split_concat(prefix, mmap=False):
    """{prefix}_x/y_{split}.npy dosyalarını havuzlar."""
    xs, ys = [], []
    for s in SPLITS:
        xs.append(np.load(os.path.join(FEATURES_DIR, f"{prefix}_x_{s}.npy"),
                          mmap_mode="r" if mmap else None))
        ys.append(np.load(os.path.join(FEATURES_DIR, f"{prefix}_y_{s}.npy")).astype(np.float32))
    return xs, np.concatenate(ys)


def _check_alignment(y_pooled, y_derived, tag):
    if len(y_pooled) != len(y_derived) or not np.array_equal(
            y_pooled.astype(np.int64), y_derived):
        raise SystemExit(f"❌ HİZALAMA HATASI ({tag}): türetilen saldırı etiketleri "
                         f"kayıtlı y ile eşleşmiyor.")


# ─────────────────────────────────────────────
# Metrikler
# ─────────────────────────────────────────────
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
    except Exception:
        return float(fpr[np.nanargmin(np.abs((1 - tpr) - fpr))])


def full_metrics(y_true, scores, thr):
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


def _avg(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)), float(np.std(v))


def _summarize_and_record(method, rows):
    print("\n" + "=" * 58)
    print(f"   {method} — SALDIRI-GRUPLU 5-FOLD ORTALAMASI")
    print("=" * 58)
    for key, lbl in [("acc", "Accuracy"), ("balacc", "Bal.Accuracy"), ("precision", "Precision"),
                     ("recall", "Recall"), ("f1", "F1-Score"), ("auc", "AUC"), ("eer", "EER")]:
        mu, sd = _avg(rows, key)
        print(f"  {lbl:<13}: {mu:.4f} ± {sd:.4f}")
    print("=" * 58)
    record_result(
        method, PROTO,
        eer=_avg(rows, "eer")[0], balacc=_avg(rows, "balacc")[0],
        recall=_avg(rows, "recall")[0], auc=_avg(rows, "auc")[0], eer_std=_avg(rows, "eer")[1],
        acc=_avg(rows, "acc")[0], precision=_avg(rows, "precision")[0], f1=_avg(rows, "f1")[0],
    )


# ─────────────────────────────────────────────
# ML (RF + SVM)
# ─────────────────────────────────────────────
def run_ml(G_atk, Y_derived):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    print("\n" + "#" * 70)
    print("# ML — Random Forest + SVM (saldırı-gruplu 5-fold)")
    print("#" * 70)
    xs, Y = _load_split_concat("ASV_ml")
    X = np.concatenate(xs)
    _check_alignment(Y, Y_derived, "ASV_ml")
    print(f"Havuz: {X.shape[0]} örnek × {X.shape[1]} öznitelik | "
          f"bonafide={int((Y==1).sum())}, spoof={int((Y==0).sum())} | "
          f"grup sayısı={len(np.unique(G_atk))}")

    cap_env = os.environ.get("ASV_SVM_KFOLD_CAP")
    svm_cap = int(cap_env) if cap_env else None
    rng = np.random.default_rng(42)

    configs = [
        ("RF", lambda: RandomForestClassifier(n_estimators=200, random_state=42,
                                              n_jobs=-1, class_weight="balanced"),
         "proba", None),
        ("SVM", lambda: SVC(kernel="rbf", random_state=42, class_weight="balanced",
                            cache_size=1000), "decision", svm_cap),
    ]

    gkf = GroupKFold(n_splits=K)
    for name, make, kind, cap in configs:
        print(f"\n>>> {name} {'[tüm veri]' if cap is None else f'[fold≤{cap}]'}")
        rows = []
        for fold, (tr, te) in enumerate(gkf.split(X, Y, groups=G_atk)):
            t0 = time.time()
            fit = tr
            if cap is not None and len(tr) > cap:               # sınıf oranını koruyan alt-örnek
                out = []
                for c in np.unique(Y[tr]):
                    ci = tr[Y[tr] == c]
                    k = max(1, int(round(cap * len(ci) / len(tr))))
                    out.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
                fit = np.concatenate(out)
            sc = StandardScaler().fit(X[fit])
            clf = make().fit(sc.transform(X[fit]), Y[fit])
            Xte = sc.transform(X[te])
            if kind == "proba":
                score, thr = clf.predict_proba(Xte)[:, 1], 0.5
            else:
                score, thr = clf.decision_function(Xte), 0.0
            m = full_metrics(Y[te], score, thr)
            rows.append(m)
            test_atks = sorted(set(G_atk[te][Y[te] == 0]))
            print(f"  Fold {fold+1}/{K} | Train {len(fit)} / Test {len(te)} | "
                  f"ACC {m['acc']:.4f} BalACC {m['balacc']:.4f} EER {m['eer']:.4f} "
                  f"| görülmemiş saldırı: {test_atks} ({time.time()-t0:.0f}s)")
        _summarize_and_record(name, rows)


# ─────────────────────────────────────────────
# CNN (Mel + Spec) — cnn_common altyapısını yeniden kullanır
# ─────────────────────────────────────────────
def run_cnn(G_atk, y_derived, which):
    import gc
    import torch
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    import asvspoof_cnn_common as cc

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    batch_size = 64 if device.type == "cuda" else 32
    epochs = int(os.environ.get("ASV_EPOCHS", "50"))
    common = dict(device=device, batch_size=batch_size, epochs=epochs, patience=10, pin=pin)

    cfgs = []
    if which in ("mel", "both"):
        cfgs.append(dict(prefix="ASV_mel", tag="cnn_mel_attack", name="Mel-CNN", freq_mask=20))
    if which in ("spec", "both"):
        cfgs.append(dict(prefix="ASV_spec", tag="cnn_spec_attack", name="Spec-CNN", freq_mask=30))

    for cfg in cfgs:
        print("\n" + "#" * 70)
        print(f"# {cfg['name']} (saldırı-gruplu 5-fold) | Donanım: {device} | batch: {batch_size}")
        print("#" * 70)
        xs, Y = _load_split_concat(cfg["prefix"], mmap=True)
        X = cc.MmapConcatArray(xs)
        _check_alignment(Y, y_derived, cfg["prefix"])
        all_idx = np.arange(len(Y))
        gkf = GroupKFold(n_splits=K)
        rows = []
        for fold, (tv, te) in enumerate(gkf.split(all_idx, Y, groups=G_atk)):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_rel, val_rel = next(gss.split(tv, Y[tv], groups=G_atk[tv]))
            train_idx, val_idx = tv[tr_rel], tv[val_rel]
            test_atks = sorted(set(G_atk[te][Y[te] == 0]))
            print(f"\n--- FOLD {fold+1}/{K} | Train {len(train_idx)} / Val {len(val_idx)} / "
                  f"Test {len(te)} | görülmemiş saldırı: {test_atks} ---")
            t0 = time.time()
            mean, std = cc.compute_mean_std(X, train_idx)
            ckpt = os.path.join(cc.MODELS_DIR, f"asvspoof_{cfg['tag']}_fold{fold+1}.pth")
            model = cc.train_model(X, Y, train_idx, val_idx, mean, std,
                                   ckpt_path=ckpt, freq_mask=cfg["freq_mask"], **common)
            labels, probs = cc.evaluate(model, X, Y, te, mean, std,
                                        device=device, batch_size=batch_size, pin=pin)
            m = cc._metrics(labels, probs)[0]
            rows.append(m)
            print(f"Fold {fold+1} → ACC {m['acc']:.4f} BalACC {m['balacc']:.4f} "
                  f"EER {m['eer']:.4f} ({time.time()-t0:.0f}s)")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _summarize_and_record(cfg["name"], rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-only", action="store_true")
    ap.add_argument("--cnn-only", action="store_true")
    ap.add_argument("--mel-only", action="store_true")
    ap.add_argument("--spec-only", action="store_true")
    args = ap.parse_args()

    G_atk, G_atk_y = build_attack_groups()
    n_attack_groups = len([g for g in set(G_atk) if not g.startswith("bona_")])
    print(f"Saldırı-grup dizisi: {len(G_atk)} örnek | {n_attack_groups} saldırı grubu + "
          f"{len(set(G_atk)) - n_attack_groups} bonafide-konuşmacı grubu")

    do_ml = args.ml_only or not (args.cnn_only or args.mel_only or args.spec_only)
    do_mel = args.mel_only or args.cnn_only or not (args.ml_only or args.spec_only)
    do_spec = args.spec_only or args.cnn_only or not (args.ml_only or args.mel_only)

    if do_ml:
        run_ml(G_atk, G_atk_y)
    if do_mel and do_spec:
        run_cnn(G_atk, G_atk_y, "both")
    elif do_mel:
        run_cnn(G_atk, G_atk_y, "mel")
    elif do_spec:
        run_cnn(G_atk, G_atk_y, "spec")

    print("\n✓ Saldırı-gruplu 5-fold tamamlandı. Sonuçlar: results/asvspoof_results.md (attack_kfold)")
