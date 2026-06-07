import os
import librosa
import numpy as np
from tqdm import tqdm

BASE_DIR = "Dataset/for-original"
SPLITS = ["training", "validation", "testing"]

# ─────────────────────────────────────────────
# Parametreler
# ─────────────────────────────────────────────
DURATION   = 5.0     # Saniye (mel ile aynı)
SR         = 22050   # Örnekleme hızı (mel ile aynı — adil karşılaştırma)
N_FFT      = 512     # FFT pencere boyutu → 257 frekans kovası verir
HOP_LENGTH = 512     # Aynı hop → zaman çözünürlüğü mel ile aynı (216 frame)

MAX_PAD_LEN = int((SR * DURATION) / HOP_LENGTH) + 1   # 216
N_FREQ_BINS = (N_FFT // 2) + 1                        # 257

print(f"Spektrogram konfigürasyonu:")
print(f"  SR={SR}, n_fft={N_FFT}, hop_length={HOP_LENGTH}")
print(f"  Çıktı şekli: ({N_FREQ_BINS}, {MAX_PAD_LEN})")
print(f"  Mel ile karşılaştırılabilir → her ikisi de 216 zaman frame'i\n")

# Mel-Spektogramdan farkı y eksenindeki frekans çözünürlüğü logaritmik değil lineer yani yüksek frekanslar sıkıştırılmıyo.
def extract_standard_spectrogram(file_path):
    """
    Ses dosyasından (257, 216) boyutunda standart STFT-tabanlı
    log-genlik spektrogramı üretir.
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)

        # STFT → karmaşık matris
        stft_matrix = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Genlik spektrogramı (mutlak değer)
        spec_mag = np.abs(stft_matrix)

        # dB ölçeğine çevir (insan algısına yakın, modelin öğrenmesi kolay)
        spec_db = librosa.amplitude_to_db(spec_mag, ref=np.max)

        # Sabit zaman boyutu (Padding/Truncating)
        if spec_db.shape[1] > MAX_PAD_LEN:
            spec_db = spec_db[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - spec_db.shape[1]
            spec_db = np.pad(spec_db, pad_width=((0, 0), (0, pad_width)),
                             mode="constant")

        return spec_db.astype(np.float32)

    except Exception:
        return None


def process_directory(directory, label):
    features, labels = [], []
    valid_files = [f for f in os.listdir(directory)
                   if f.endswith((".wav", ".mp3"))]

    for file_name in tqdm(valid_files, desc=f"Etiket: {label}"):
        data = extract_standard_spectrogram(os.path.join(directory, file_name))
        if data is not None:
            features.append(data)
            labels.append(label)

    return features, labels


for split in SPLITS:
    print(f"\n--- {split.upper()} Klasörü İşleniyor (Standart Spektrogram) ---")
    real_dir = os.path.join(BASE_DIR, split, "real")
    fake_dir = os.path.join(BASE_DIR, split, "fake")

    print("Gerçek sesler işleniyor...")
    real_features, real_labels = process_directory(real_dir, label=1)

    print("Sahte sesler işleniyor...")
    fake_features, fake_labels = process_directory(fake_dir, label=0)

    x = np.array(real_features + fake_features, dtype=np.float32)
    y = np.array(real_labels + fake_labels, dtype=np.int8)

    # ESF prefix = Extract Spectrogram Features (mel için ECF, mfcc için EMF)
    prefix = {"training": "ESF0", "validation": "ESF1", "testing": "ESF2"}[split]
    x_path = f"{prefix}-x_{split}_spec.npy"
    y_path = f"{prefix}-y_{split}_spec.npy"

    np.save(x_path, x)
    np.save(y_path, y)

    print(f"  → {x_path} kaydedildi | Şekil: {x.shape}")
    print(f"  → {y_path} kaydedildi | Şekil: {y.shape}")
    print(f"  → Disk boyutu: {x.nbytes / (1024**3):.2f} GB")
