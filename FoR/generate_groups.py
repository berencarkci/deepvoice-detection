"""
Grup dizisi oluşturma scripti (GroupKFold için).

Her ses dosyasına bir grup ID'si atar. Aynı grup ID'sine sahip örnekler
GroupKFold tarafından DAİMA aynı fold'a yerleştirilir → veri sızıntısı önlenir.

Grup ID formatı: "{split}_{base_name}"
  Örnek: "training_file1514"

Bu sayede:
  - Aynı split'teki real/file1514 ve fake/file1514 AYNI gruba düşer
    (aynı kaynak konuşmacının gerçek ve sahte versiyonu birlikte tutulur)
  - Farklı split'lerdeki aynı isimli dosyalar FARKLI gruplara düşer
    (training_file1514 ≠ testing_file1514 → cross-split çakışma önlenir)

ÖNEMLİ: İterasyon sırası ve başarı/başarısızlık kriterleri,
extract_ml_features.py ve extract_cnn_features.py ile BİREBİR aynıdır.
Multiprocessing KULLANILMAZ (sıra garantisi için).
"""

import os
import librosa
import numpy as np
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(_BASE_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

BASE_DIR = "Dataset/for-original"
SPLITS = ["training", "validation", "testing"]
SR = 22050
DURATION = 5.0  # Feature extraction ile aynı süre (0.1 DEĞİL)


def process_split(split):
    print(f"\nProcessing {split}...")
    real_dir = os.path.join(BASE_DIR, split, "real")
    fake_dir = os.path.join(BASE_DIR, split, "fake")

    groups = []

    # ─── Real dosyalar önce (extract_ml/cnn_features.py ile aynı sıra) ───
    if os.path.exists(real_dir):
        valid_files = [f for f in os.listdir(real_dir) if f.endswith(('.wav', '.mp3'))]
        for f in tqdm(valid_files, desc=f"{split}/real"):
            file_path = os.path.join(real_dir, f)
            base_name = os.path.splitext(f)[0]
            try:
                # Feature extraction scriptleri ile AYNI parametreler
                y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
                groups.append(f"{split}_{base_name}")
            except Exception:
                pass  # Başarısız dosyalar atlanır (feature extraction ile aynı davranış)

    # ─── Fake dosyalar sonra (extract_ml/cnn_features.py ile aynı sıra) ───
    if os.path.exists(fake_dir):
        valid_files = [f for f in os.listdir(fake_dir) if f.endswith(('.wav', '.mp3'))]
        for f in tqdm(valid_files, desc=f"{split}/fake"):
            file_path = os.path.join(fake_dir, f)
            base_name = os.path.splitext(f)[0]
            try:
                y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
                groups.append(f"{split}_{base_name}")
            except Exception:
                pass

    groups = np.array(groups)
    save_path = os.path.join(FEATURES_DIR, f"EMF_groups_{split}.npy")
    np.save(save_path, groups)
    print(f"Saved {save_path} with shape {groups.shape}")

    unique_count = len(np.unique(groups))
    print(f"  Unique groups: {unique_count}")
    print(f"  Real-Fake çiftleri (aynı grup ID): {len(groups) - unique_count}")


if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)
