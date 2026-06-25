"""
ASVspoof 2019 LA — mel ve spektrogram CNN modelleri için ortak eğitim/değerlendirme altyapısı.

Değerlendirme: saldırı-gruplu 5 katlı çapraz doğrulama. train+dev+eval birleşik havuz
üzerinde GroupKFold uygulanır; grup ataması spoof→saldırı türü, bonafide→konuşmacı.
Böylece her fold'un test seti eğitimde yer almayan saldırılardan oluşur ve görülmemiş
sentez sistemlerine genelleme ölçülür. Çapraz doğrulamadan sonra tüm havuzla bir nihai
model eğitilip arayüz için kaydedilir (asvspoof_<tag>_final.pth + .norm.json).

Sınıf dengesizliği (~1:9) için BCEWithLogitsLoss'ta pos_weight kullanılır; değer her
eğitim setinden hesaplanır. EER ve dengeli doğruluk, accuracy ile birlikte raporlanır.

Ortam değişkenleri:
  ASV_<MEL|SPEC>_BATCH_SIZE   batch boyutu (varsayılan: CUDA 64, diğer 32)
  ASV_EPOCHS                  epoch sayısı (varsayılan 50)
  ASV_SKIP_FINAL=1            çapraz doğrulamadan sonra nihai modeli eğitmeyi atla
"""

import os
import gc
import sys
import json
import atexit

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from asvspoof_extract_features import build_attack_groups
from asvspoof_results import record_result

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(_BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
FEATURES_DIR = os.path.join(_BASE_DIR, "features")
LOGS_DIR = os.path.join(_BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

K_FOLDS = 5
SPLITS = ["train", "dev", "eval"]


# ─────────────────────────────────────────────
# LOG (terminal + dosya)
# ─────────────────────────────────────────────
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


def _init_log(log_path):
    orig = sys.stdout
    fp = open(log_path, "w", encoding="utf-8")
    sys.stdout = _TeeIO(orig, fp)
    def _close():
        sys.stdout = orig
        try: fp.close()
        except Exception: pass
    atexit.register(_close)


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
    return float(eer)


# ─────────────────────────────────────────────
# RAM dostu mmap birleştirici + chunked mean/std
# ─────────────────────────────────────────────
class MmapConcatArray:
    """Birden çok disk-üstü memmap diziyi tek sanal dizi gibi birleştirir."""
    def __init__(self, arrays):
        self.arrays = arrays
        self.lengths = [len(a) for a in arrays]
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.shape = (sum(self.lengths),) + arrays[0].shape[1:]
        self.dtype = arrays[0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx += len(self)
            arr_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
            local = idx if arr_idx == 0 else idx - self.cumulative_lengths[arr_idx - 1]
            return self.arrays[arr_idx][local]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            idx = np.arange(start, stop, step)
        if isinstance(idx, (list, np.ndarray)):
            return np.array([self[int(i)] for i in idx], dtype=self.dtype)
        raise TypeError(f"Desteklenmeyen indeks tipi: {type(idx)}")


def compute_mean_std(x_total, indices, chunk_size=2000):
    """Büyük mmap dizilerde RAM doldurmadan ortalama/std (iki geçiş)."""
    total_sum, total_count = 0.0, 0
    for i in range(0, len(indices), chunk_size):
        chunk = x_total[indices[i:i + chunk_size]]
        total_sum += chunk.sum(); total_count += chunk.size
    mean = total_sum / total_count
    sq = 0.0
    for i in range(0, len(indices), chunk_size):
        chunk = x_total[indices[i:i + chunk_size]]
        sq += ((chunk - mean) ** 2).sum()
    return float(mean), float(np.sqrt(sq / total_count))


# ─────────────────────────────────────────────
# DATASET (indis tabanlı, z-score + SpecAugment)
# ─────────────────────────────────────────────
class SpecDataset(Dataset):
    def __init__(self, x_data, y_data, indices, mean, std,
                 augment=False, freq_mask_param=30, time_mask_param=40):
        self.x = x_data
        self.y = y_data
        self.indices = np.asarray(indices)
        self.mean = mean; self.std = std
        self.augment = augment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __len__(self):
        return len(self.indices)

    def _spec_augment(self, x):
        _, n_freq, t_steps = x.shape
        f = np.random.randint(0, self.freq_mask_param + 1)
        f0 = np.random.randint(0, max(1, n_freq - f))
        x[:, f0:f0 + f, :] = 0.0
        t = np.random.randint(0, self.time_mask_param + 1)
        t0 = np.random.randint(0, max(1, t_steps - t))
        x[:, :, t0:t0 + t] = 0.0
        return x

    def __getitem__(self, idx):
        actual = int(self.indices[idx])
        row = self.x[actual]
        if not row.flags.c_contiguous:
            row = np.ascontiguousarray(row)
        row = row.astype(np.float32)
        x_val = torch.from_numpy(row).unsqueeze(0)
        x_val = (x_val - self.mean) / (self.std + 1e-8)
        if self.augment:
            x_val = self._spec_augment(x_val)
        y_val = torch.tensor([float(self.y[actual])], dtype=torch.float32)
        return x_val, y_val


# ─────────────────────────────────────────────
# MODEL MİMARİSİ (adaptive pooling farklı giriş boyutlarını kabul eder)
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.act(self.block(x) + x)


class DeepVoiceCNN(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        def cbl(i, o, k=3, s=1, p=1):
            return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=False),
                                 nn.BatchNorm2d(o), nn.LeakyReLU(0.1, inplace=True))
        self.stem = nn.Sequential(cbl(1, 32), nn.MaxPool2d(2, 2))
        self.stage1 = nn.Sequential(cbl(32, 64), ResBlock(64), nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate))
        self.stage2 = nn.Sequential(cbl(64, 128), ResBlock(128), nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate))
        self.stage3 = nn.Sequential(cbl(128, 256), ResBlock(256), nn.AdaptiveAvgPool2d((4, 9)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4 * 9, 256),
            nn.LeakyReLU(0.1, inplace=True), nn.Dropout(dropout_rate + 0.1),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        return self.classifier(self.stage3(self.stage2(self.stage1(self.stem(x)))))


# ─────────────────────────────────────────────
# EĞİTİM / DEĞERLENDİRME
# ─────────────────────────────────────────────
def _pos_weight(Y, idx, device):
    """pos_weight = #spoof / #bonafide  (azınlık olan bonafide=1'i yukarı ağırlıklar)."""
    yt = np.asarray(Y)[idx]
    n_pos = int((yt == 1).sum()); n_neg = int((yt == 0).sum())
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)


def train_model(X, Y, train_idx, val_idx, mean, std, *, device, freq_mask,
                batch_size, epochs, patience, ckpt_path, pin):
    train_ds = SpecDataset(X, Y, train_idx, mean, std, augment=True, freq_mask_param=freq_mask)
    val_ds = SpecDataset(X, Y, val_idx, mean, std, augment=False, freq_mask_param=freq_mask)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    model = DeepVoiceCNN(dropout_rate=0.4).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(Y, train_idx, device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val, patience_ctr = float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=pin), labels.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=pin), labels.to(device, non_blocking=pin)
                vloss += criterion(model(inputs), labels).item()
        vloss /= len(val_loader)
        scheduler.step()
        if vloss < best_val:
            best_val, patience_ctr = vloss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Epoch {epoch:02d} ⚠ Early stopping (val_loss={vloss:.4f})")
                break

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def evaluate(model, X, Y, test_idx, mean, std, *, device, batch_size, pin):
    model.eval()
    ds = SpecDataset(X, Y, test_idx, mean, std, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    logits, labels = [], []
    for inputs, lab in loader:
        logits.extend(model(inputs.to(device, non_blocking=pin)).cpu().numpy())
        labels.extend(lab.numpy())
    labels = np.array(labels).flatten()
    probs = expit(np.array(logits).flatten())          # sayısal kararlı sigmoid (taşma yok)
    return labels, probs


def _metrics(labels, probs):
    preds = (probs > 0.5).astype(int)
    return {
        "acc": accuracy_score(labels, preds),
        "balacc": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
        "eer": compute_eer(labels, probs),
    }, preds


# ─────────────────────────────────────────────
# DENEYİ ÇALIŞTIR
# ─────────────────────────────────────────────
def run_experiment(cfg):
    """
    cfg: dict(name, result_name, title, x_prefix, freq_mask, log_file, model_tag, batch_env)
      x_prefix: "ASV_mel" veya "ASV_spec"  → {prefix}_x_{split}.npy, {prefix}_y_{split}.npy
    """
    _init_log(os.path.join(LOGS_DIR, cfg["log_file"]))

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin = device.type == "cuda"
    default_bs = 64 if device.type == "cuda" else 32
    batch_size = int(os.environ.get(cfg["batch_env"], str(default_bs)))
    epochs = int(os.environ.get("ASV_EPOCHS", "50"))
    patience = 10
    skip_final = os.environ.get("ASV_SKIP_FINAL", "0") == "1"

    print(f"ASVspoof 2019 LA — {cfg['title']} (saldırı-gruplu)")
    print(f"Donanım: {device} | batch: {batch_size} | epochs: {epochs}")
    print("=" * 70)

    # ── Veri (mmap) ──
    xs, ys = [], []
    for split in SPLITS:
        xp = os.path.join(FEATURES_DIR, f"{cfg['x_prefix']}_x_{split}.npy")
        yp = os.path.join(FEATURES_DIR, f"{cfg['x_prefix']}_y_{split}.npy")
        if not (os.path.exists(xp) and os.path.exists(yp)):
            print(f"\n❌ Eksik dosya ({split}): {xp} / {yp}")
            print("   Önce: python asvspoof/asvspoof_extract_features.py "
                  f"--{'mel' if 'mel' in cfg['x_prefix'] else 'spec'}-only")
            sys.exit(1)
        xs.append(np.load(xp, mmap_mode="r"))
        ys.append(np.load(yp).astype(np.float32))

    X = MmapConcatArray(xs)
    Y = np.concatenate(ys)

    # ── Saldırı grupları (öznitelik sırasıyla hizalı; kayıtlı y ile doğrulanır) ──
    G, Y_derived = build_attack_groups()
    if len(Y) != len(Y_derived) or not np.array_equal(Y.astype(np.int64), Y_derived):
        print("\n❌ HİZALAMA HATASI: türetilen saldırı etiketleri kayıtlı y ile eşleşmiyor.")
        sys.exit(1)
    n_attack = len([g for g in set(G) if not g.startswith("bona_")])
    print(f"Toplam: {len(Y)} örnek | bonafide={int((Y==1).sum())}, spoof={int((Y==0).sum())} | "
          f"{n_attack} saldırı grubu + {len(set(G)) - n_attack} bonafide-konuşmacı grubu")

    common = dict(device=device, freq_mask=cfg["freq_mask"], batch_size=batch_size,
                  epochs=epochs, patience=patience, pin=pin)

    # ════════════════ Saldırı-gruplu 5 katlı çapraz doğrulama ════════════════
    print("\n" + "=" * 70)
    print(f"  {K_FOLDS}-FOLD GroupKFold (saldırı-gruplu) — görülmemiş saldırı türleri")
    print("=" * 70)

    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_metrics = []
    all_idx = np.arange(len(Y))
    oof_probs = np.zeros(len(Y))                       # out-of-fold (her örnek tam 1 kez test)

    for fold, (tv_idx, test_idx) in enumerate(gkf.split(all_idx, Y, groups=G)):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_rel, val_rel = next(gss.split(tv_idx, Y[tv_idx], groups=G[tv_idx]))
        train_idx, val_idx = tv_idx[tr_rel], tv_idx[val_rel]
        test_atks = sorted(set(G[test_idx][Y[test_idx] == 0]))

        print(f"\n--- FOLD {fold + 1}/{K_FOLDS} | Train {len(train_idx)} / Val {len(val_idx)} / "
              f"Test {len(test_idx)} | görülmemiş saldırı: {test_atks} ---")
        mean, std = compute_mean_std(X, train_idx)
        ckpt = os.path.join(MODELS_DIR, f"asvspoof_{cfg['model_tag']}_fold{fold+1}.pth")
        model = train_model(X, Y, train_idx, val_idx, mean, std, ckpt_path=ckpt, **common)
        labels, probs = evaluate(model, X, Y, test_idx, mean, std,
                                 device=device, batch_size=batch_size, pin=pin)
        oof_probs[test_idx] = probs
        m, _ = _metrics(labels, probs)
        fold_metrics.append(m)
        print(f"Fold {fold+1} → Test ACC: {m['acc']:.4f} | BalACC: {m['balacc']:.4f} | EER: {m['eer']:.4f}")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _avg(key):
        v = [fm[key] for fm in fold_metrics]
        return np.mean(v), np.std(v)

    print("\n" + "=" * 60)
    print(f"   SALDIRI-GRUPLU {K_FOLDS}-FOLD SONUÇLARI")
    print("=" * 60)
    for key, lbl in [("acc", "Accuracy"), ("balacc", "Bal.Accuracy"), ("precision", "Precision"),
                     ("recall", "Recall"), ("f1", "F1-Score"), ("auc", "AUC"), ("eer", "EER")]:
        mu, sd = _avg(key)
        print(f"  {lbl:<13}: {mu:.4f} ± {sd:.4f}")
    print("=" * 60)

    record_result(
        cfg["result_name"], "attack_kfold",
        eer=_avg("eer")[0], balacc=_avg("balacc")[0],
        recall=_avg("recall")[0], auc=_avg("auc")[0], eer_std=_avg("eer")[1],
        acc=_avg("acc")[0], precision=_avg("precision")[0], f1=_avg("f1")[0],
    )

    _save_plots(cfg, Y, oof_probs, fold_metrics)

    # ════════════════ Nihai model (tüm havuz) — arayüz için ════════════════
    if not skip_final:
        _train_final(cfg, X, Y, G, all_idx, common, device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"kfold": fold_metrics}


def _save_plots(cfg, Y, oof_probs, fold_metrics):
    """Saldırı-gruplu OOF tahminlerinden ROC+confusion ve fold bazlı metrik grafikleri."""
    try:
        # ROC + Confusion (OOF)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(f"{cfg['title']} — Saldırı-gruplu OOF", fontsize=12.5, fontweight="bold")
        fpr, tpr, _ = roc_curve(Y, oof_probs)
        auc_v = roc_auc_score(Y, oof_probs); eer_v = compute_eer(Y, oof_probs)
        axes[0].plot(fpr, tpr, color="darkorchid", lw=2, label=f"AUC = {auc_v:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[0].set_title(f"ROC (OOF)\nGenel EER ≈ {eer_v:.3f}")
        axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)
        preds = (oof_probs > 0.5).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            Y.astype(int), preds, display_labels=["Spoof (0)", "Bonafide (1)"],
            cmap=plt.cm.Purples, ax=axes[1], colorbar=False)
        axes[1].set_title(f"Confusion (OOF)\nBalACC {balanced_accuracy_score(Y, preds)*100:.2f}%")
        plt.tight_layout()
        p1 = os.path.join(FIGURES_DIR, f"asvspoof_{cfg['model_tag']}_roc_confusion.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  ✔ {os.path.basename(p1)}")

        # Fold bazlı Accuracy / EER
        fig_b, ax_b = plt.subplots(figsize=(8, 4.5))
        xs = list(range(1, len(fold_metrics) + 1))
        ax_b.bar(xs, [fm["acc"] for fm in fold_metrics], color="steelblue", alpha=0.85, label="Accuracy")
        ax_b.plot(xs, [fm["eer"] for fm in fold_metrics], color="crimson", marker="o", lw=2, label="EER")
        ax_b.set_xticks(xs); ax_b.set_xlabel("Fold"); ax_b.set_ylim(0, 1.0)
        ax_b.set_title(f"{cfg['title']} — Fold bazlı Accuracy / EER")
        ax_b.legend(); ax_b.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p2 = os.path.join(FIGURES_DIR, f"asvspoof_{cfg['model_tag']}_fold_metrics.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig_b)
        print(f"  ✔ {os.path.basename(p2)}")
    except Exception as e:
        print(f"\nUyarı: grafik kaydedilemedi ({type(e).__name__}: {e})")


def _train_final(cfg, X, Y, G, all_idx, common, device):
    """Tüm havuzu %90/%10 (gruba göre) bölerek arayüz için nihai model eğitir;
    ağırlıkları ve normalizasyon istatistiklerini (.norm.json) kaydeder."""
    print("\n" + "=" * 70)
    print("  NİHAİ MODEL — tüm havuzla eğitilir (arayüz için kaydedilir)")
    print("=" * 70)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_rel, val_rel = next(gss.split(all_idx, Y, groups=G))
    train_idx, val_idx = all_idx[tr_rel], all_idx[val_rel]
    mean, std = compute_mean_std(X, train_idx)
    ckpt = os.path.join(MODELS_DIR, f"asvspoof_{cfg['model_tag']}_final.pth")
    train_model(X, Y, train_idx, val_idx, mean, std, ckpt_path=ckpt, **common)
    norm_path = os.path.splitext(ckpt)[0] + ".norm.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std}, f)
    print(f"  ✔ Kaydedildi: {ckpt} (+ .norm.json | mean={mean:.4f}, std={std:.4f})")
