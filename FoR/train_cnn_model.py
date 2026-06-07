import sys
import atexit
import gc
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
_MPL_DIR = os.path.join(_BASE_DIR, ".mplconfig")
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

# GUI backend hatası (Tk/Qt yok) önlenir — grafikler dosyaya yazılır
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# ── Terminal + dosya: her çalıştırmada train_cnn_mel.log güncellenir ──────────
LOG_FILE = os.path.join(_BASE_DIR, "train_cnn_mel.log")

class _TeeIO:
    """stdout'u konsola ve log dosyasına çift yazar."""
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

def _close_mel_training_log():
    sys.stdout = _orig_stdout
    try:
        _log_fp.close()
    except Exception:
        pass

atexit.register(_close_mel_training_log)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
    return float(eer)

# ─────────────────────────────────────────────
# 0. DONANIM SEÇİMİ
# ─────────────────────────────────────────────
device = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"Kullanılan donanım birimi: {device}")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. TÜM VERİLERİ YÜKLEME VE BİRLEŞTİRME
# ─────────────────────────────────────────────
print("\nTüm veriler (Train + Val + Test) yüklenip birleştiriliyor...")
x1 = np.load("ECF0-x_training_mel.npy").astype(np.float32)
y1 = np.load("ECF0-y_training_mel.npy").astype(np.float32)

x2 = np.load("ECF1-x_validation_mel.npy").astype(np.float32)
y2 = np.load("ECF1-y_validation_mel.npy").astype(np.float32)

x3 = np.load("ECF2-x_testing_mel.npy").astype(np.float32)
y3 = np.load("ECF2-y_testing_mel.npy").astype(np.float32)

X_TOTAL = np.concatenate((x1, x2, x3), axis=0)
Y_TOTAL = np.concatenate((y1, y2, y3), axis=0)

del x1, y1, x2, y2, x3, y3  # RAM temizliği

groups1 = np.load("EMF_groups_training.npy")
groups2 = np.load("EMF_groups_validation.npy")
groups3 = np.load("EMF_groups_testing.npy")
GROUPS_TOTAL = np.concatenate((groups1, groups2, groups3), axis=0)

print(f"Toplam Birleşik Veri Seti Boyutu: {X_TOTAL.shape[0]} örnek")

# ─────────────────────────────────────────────
# 2. DATASET SINIFI (Array Tabanlı)
# ─────────────────────────────────────────────
class MelSpectrogramDataset(Dataset):
    def __init__(self, x_data, y_data, mean: float, std: float,
                 augment: bool = False, freq_mask_param: int = 20, time_mask_param: int = 40):
        self.x = np.asarray(x_data, dtype=np.float32)
        self.y = np.asarray(y_data, dtype=np.float32)
        self.mean = mean
        self.std = std
        self.augment = augment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __len__(self):
        return len(self.x)

    def _apply_spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        _, n_mels, time_steps = x.shape
        f = np.random.randint(0, self.freq_mask_param + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        x[:, f0:f0 + f, :] = 0.0

        t = np.random.randint(0, self.time_mask_param + 1)
        t0 = np.random.randint(0, max(1, time_steps - t))
        x[:, :, t0:t0 + t] = 0.0
        return x

    def __getitem__(self, idx):
        row = self.x[idx]
        if not row.flags.c_contiguous:
            row = np.ascontiguousarray(row)
        x_val = torch.from_numpy(row).unsqueeze(0)
        x_val = (x_val - self.mean) / (self.std + 1e-8)
        if self.augment:
            x_val = self._apply_spec_augment(x_val)
        y_val = torch.tensor([float(self.y[idx])], dtype=torch.float32)
        return x_val, y_val

# ─────────────────────────────────────────────
# 3. MİMARİ
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)

class DeepVoiceCNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()
        def conv_bn_lrelu(in_ch, out_ch, kernel=3, stride=1, pad=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        self.stem = nn.Sequential(conv_bn_lrelu(1, 32), nn.MaxPool2d(2, 2))
        self.stage1 = nn.Sequential(conv_bn_lrelu(32, 64), ResBlock(64), nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate))
        self.stage2 = nn.Sequential(conv_bn_lrelu(64, 128), ResBlock(128), nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate))
        self.stage3 = nn.Sequential(conv_bn_lrelu(128, 256), ResBlock(256), nn.AdaptiveAvgPool2d((4, 9)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 9, 256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)

# ─────────────────────────────────────────────
# 4. İÇ İÇE (NESTED) K-FOLD EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────
K_FOLDS = 5
EPOCHS = 50
_default_bs = 64 if device.type == "cuda" else 32
BATCH_SIZE = int(os.environ.get("DEEPVOICE_MEL_BATCH_SIZE", str(_default_bs)))
PATIENCE = 10
_pin = device.type == "cuda"

print(f"Batch boyutu: {BATCH_SIZE} (DEEPVOICE_MEL_BATCH_SIZE ile değiştirilebilir)")

kfold = GroupKFold(n_splits=K_FOLDS)

fold_results = {
    'accuracy': [], 'precision': [], 'recall': [], 
    'f1': [], 'auc': [], 'eer': []
}

# Tüm verisetinin test sonuçlarını toplamak için OOF (Out-of-Fold) dizileri
oof_preds = np.zeros(len(Y_TOTAL))
oof_probs = np.zeros(len(Y_TOTAL))

print(f"\n{K_FOLDS}-Fold Cross Validation Başlıyor (GroupKFold)...")
print("=" * 65)

for fold, (train_val_idx, test_idx) in enumerate(kfold.split(X_TOTAL, Y_TOTAL, groups=GROUPS_TOTAL)):
    print(f"\n--- FOLD {fold + 1}/{K_FOLDS} ---")
    
    # 1. Adım: Tüm verinin %20'si bu fold için TEST setidir.
    X_test_fold, Y_test_fold = X_TOTAL[test_idx], Y_TOTAL[test_idx]
    
    # Geriye kalan 4 parça (%80) Train+Val havuzudur.
    X_train_val, Y_train_val = X_TOTAL[train_val_idx], Y_TOTAL[train_val_idx]
    groups_train_val = GROUPS_TOTAL[train_val_idx]
    
    # 2. Adım: Kalan kısmı %80 Train, %20 Validation olarak böl
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X_train_val, Y_train_val, groups=groups_train_val))
    
    X_train_fold = X_train_val[train_idx]
    Y_train_fold = Y_train_val[train_idx]
    X_val_fold   = X_train_val[val_idx]
    Y_val_fold   = Y_train_val[val_idx]
    
    print(f"Veri Dağılımı -> Train: {len(X_train_fold)}, Val: {len(X_val_fold)}, Test: {len(X_test_fold)}")
    
    # 3. Global Mean/Std Hesabı (SADECE Train fold verisinden)
    fold_mean = float(X_train_fold.mean())
    fold_std  = float(X_train_fold.std())
    
    # 4. Dataset ve DataLoader Oluştur
    train_dataset = MelSpectrogramDataset(X_train_fold, Y_train_fold, mean=fold_mean, std=fold_std, augment=True)
    val_dataset   = MelSpectrogramDataset(X_val_fold, Y_val_fold, mean=fold_mean, std=fold_std, augment=False)
    test_dataset  = MelSpectrogramDataset(X_test_fold, Y_test_fold, mean=fold_mean, std=fold_std, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=_pin)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=_pin)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=_pin)
    
    # 5. Modeli Sıfırla
    model = DeepVoiceCNN(dropout_rate=0.4).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(MODELS_DIR, f"deepvoice_cnn_mel_fold{fold+1}_best.pth")
    
    # 6. Eğitim Döngüsü
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=_pin), labels.to(device, non_blocking=_pin)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=_pin), labels.to(device, non_blocking=_pin)
                val_loss += criterion(model(inputs), labels).item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Epoch [{epoch:02d}] ⚠ Early Stopping (Val Loss: {avg_val_loss:.4f})")
                break
                
    # 7. O Fold'un TEST Seti Üzerinde Değerlendirme
    try:
        _state = torch.load(best_model_path, map_location=device, weights_only=False)
    except TypeError:
        _state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(_state)
    model.eval()
    
    all_logits, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=_pin)
            logits = model(inputs).cpu().numpy()
            all_logits.extend(logits)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels).flatten()
    all_probs  = 1 / (1 + np.exp(-np.array(all_logits).flatten()))
    all_preds  = (all_probs > 0.5).astype(int)

    # Fold Metriklerini Kaydet
    test_acc = accuracy_score(all_labels, all_preds)
    test_eer = compute_eer(all_labels, all_probs)
    
    fold_results['accuracy'].append(test_acc)
    fold_results['precision'].append(precision_score(all_labels, all_preds, zero_division=0))
    fold_results['recall'].append(recall_score(all_labels, all_preds, zero_division=0))
    fold_results['f1'].append(f1_score(all_labels, all_preds, zero_division=0))
    fold_results['auc'].append(roc_auc_score(all_labels, all_probs))
    fold_results['eer'].append(test_eer)

    # OOF tahmini genel havuzdaki indekslerine kaydet
    oof_preds[test_idx] = all_preds
    oof_probs[test_idx] = all_probs
    
    print(f"Fold {fold+1} Test Tamamlandı -> Test ACC: {test_acc:.4f} | Test EER: {test_eer:.4f}")

    del train_dataset, val_dataset, test_dataset
    del train_loader, val_loader, test_loader
    del X_train_fold, X_val_fold, X_test_fold
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# 5. FİNAL ÜRETİM (PRODUCTION) MODELİ EĞİTİMİ
# ─────────────────────────────────────────────
print("\n" + "-" * 65)
print("Nihai CNN modeli eğitiliyor (Tüm veri havuzunun %90'ı Train, %10'u Val)...")
gss_final = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
final_train_idx, final_val_idx = next(gss_final.split(X_TOTAL, Y_TOTAL, groups=GROUPS_TOTAL))
X_final_train = X_TOTAL[final_train_idx]
Y_final_train = Y_TOTAL[final_train_idx]
X_final_val   = X_TOTAL[final_val_idx]
Y_final_val   = Y_TOTAL[final_val_idx]

final_mean = float(X_final_train.mean())
final_std  = float(X_final_train.std())

final_train_dataset = MelSpectrogramDataset(X_final_train, Y_final_train, mean=final_mean, std=final_std, augment=True)
final_val_dataset   = MelSpectrogramDataset(X_final_val, Y_final_val, mean=final_mean, std=final_std, augment=False)

final_train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=_pin)
final_val_loader   = DataLoader(final_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=_pin)

final_model = DeepVoiceCNN(dropout_rate=0.4).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_final_val_loss = float("inf")
patience_counter = 0
final_model_path = os.path.join(MODELS_DIR, "deepvoice_cnn_mel_final.pth")

for epoch in range(1, EPOCHS + 1):
    final_model.train()
    for inputs, labels in final_train_loader:
        inputs, labels = inputs.to(device, non_blocking=_pin), labels.to(device, non_blocking=_pin)
        optimizer.zero_grad(set_to_none=True)
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()
        
    final_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in final_val_loader:
            inputs, labels = inputs.to(device, non_blocking=_pin), labels.to(device, non_blocking=_pin)
            val_loss += criterion(final_model(inputs), labels).item()
            
    avg_val_loss = val_loss / len(final_val_loader)
    scheduler.step()
    
    if avg_val_loss < best_final_val_loss:
        best_final_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(final_model.state_dict(), final_model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break
            
print(f"Nihai model kaydedildi: {final_model_path} (Final Mean: {final_mean:.4f}, Std: {final_std:.4f})")

print("\n" + "=" * 55)
print(f"          TÜM VERİSETİ K-FOLD (K={K_FOLDS}) FİNAL SONUÇLARI")
print("=" * 55)
print(f"  Accuracy  : {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")
print(f"  Precision : {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}")
print(f"  Recall    : {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}")
print(f"  F1-Score  : {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
print(f"  AUC       : {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
print(f"  EER       : {np.mean(fold_results['eer']):.4f} ± {np.std(fold_results['eer']):.4f}")
print("=" * 55)

# ── Özet grafikler (Tüm Verisetini Kapsayan K-Fold) ──────────────────────────────
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"CNN-Mel K-Fold — Tüm Veriseti Çapraz Doğrulama Özeti",
        fontsize=12,
        fontweight="bold",
    )

    fpr, tpr, _thr = roc_curve(Y_TOTAL, oof_probs)
    auc_v = roc_auc_score(Y_TOTAL, oof_probs)
    eer_v = compute_eer(Y_TOTAL, oof_probs)
    
    ax = axes[0]
    ax.plot(fpr, tpr, color="darkorchid", lw=2, label=f"AUC = {auc_v:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC (Tüm K-Fold Sonuçları)\nGenel EER ≈ {eer_v:.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ConfusionMatrixDisplay.from_predictions(
        Y_TOTAL,
        oof_preds,
        display_labels=["Sahte (0)", "Gerçek (1)"],
        cmap=plt.cm.Purples,
        ax=axes[1],
        colorbar=False,
    )
    axes[1].set_title("Confusion Matrix (Tüm K-Fold Sonuçları)")

    plt.tight_layout()
    _roc_cm_path = os.path.join(FIGURES_DIR, "cnn_mel_kfold_roc_confusion.png")
    plt.savefig(_roc_cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nROC + karmaşıklık matrisi kaydedildi: {_roc_cm_path}")

    fig_b, ax_b = plt.subplots(figsize=(8, 4))
    xs = list(range(1, K_FOLDS + 1))
    ax_b.bar(xs, fold_results["accuracy"], color="steelblue", alpha=0.85)
    ax_b.axhline(np.mean(fold_results["accuracy"]), color="crimson", ls="--", lw=2,
                 label=f"Ort. Acc = {np.mean(fold_results['accuracy']):.4f}")
    ax_b.set_xticks(xs)
    ax_b.set_xlabel("Fold")
    ax_b.set_ylabel("Test Accuracy (fold modeli)")
    ax_b.set_title("CNN-Mel K-Fold — Fold bazlı test doğruluğu")
    ax_b.legend()
    ax_b.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    
    _bar_path = os.path.join(FIGURES_DIR, "cnn_mel_kfold_accuracy_bars.png")
    plt.savefig(_bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print(f"Fold doğruluk çubukları kaydedildi: {_bar_path}")

except Exception as _plot_err:
    print(f"\nUyarı: Grafik kaydedilemedi ({type(_plot_err).__name__}: {_plot_err})")