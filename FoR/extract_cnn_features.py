import os
import librosa
import numpy as np
from tqdm import tqdm

# Temel dizin ve ayrılmış klasörler
BASE_DIR = "Dataset/for-original"
SPLITS = ["training", "validation", "testing"]

# CNN için sabit parametreler
DURATION = 5.0 # Saniye (mel ile aynı)
SR = 22050 # Örnekleme hızı (mel ile aynı)
N_MELS = 128 # Mel bant sayısı (insan kulağı 128 banda kadar ayırt edebiliyor)
HOP_LENGTH = 512 # Hop boyutu (5 saniyelik sesi 216 zaman parçasına böler)
MAX_PAD_LEN = int((SR * DURATION) / HOP_LENGTH) + 1  # 216 (mel ile aynı)


# Mel-Spektorgam = Y ekseni: Frekans (alttan yukarıya, 128 mel bandı)
#                  X ekseni: Zaman (soldan sağa, 216 frame)
#                  Renk    : Güç (dB cinsinden, koyu = sessiz, açık = yüksek ses)
#                  İnsan kulağı logaritmik algılar 100 Hz ile 200 Hz farkı çok belirginken 8000 Hz ile 8100 Hz fark edilmez Mel ölçeği bunu modeller, düşük frekansları sıkıştırmadan, yüksek frekansları gruplandırarak verir. Standart STFT spektogramında bu yapılmaz.
def extract_mel_spectrogram(file_path):
    """Ses dosyasını 128x216 boyutunda Mel-Spektrogram matrisine çevirir."""
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)

        # Mel-Spektrogram çıkar (güç ölçeğinde)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)

        # dB ölçeğine çevir (insan kulağına yakın)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Boyut sabitleme (Padding/Truncating)
        if mel_spec_db.shape[1] > MAX_PAD_LEN:
            mel_spec_db = mel_spec_db[:, :MAX_PAD_LEN] # uzunsa kes
        else:
            pad_width = MAX_PAD_LEN - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        return mel_spec_db # şekil 128,216
    except Exception as e:
        return None

def process_directory(directory, label):
    features, labels = [], []
    valid_files = [f for f in os.listdir(directory) if f.endswith(('.wav', '.mp3'))]
    
    for file_name in tqdm(valid_files, desc=f"Etiket: {label}"):
        data = extract_mel_spectrogram(os.path.join(directory, file_name))
        if data is not None:
            features.append(data)
            labels.append(label)
            
    return features, labels

# ECF prefix = "Extract CNN Features" (EMF=ml, ESF=std-spec ile uyumlu)
PREFIX_MAP = {"training": "ECF0", "validation": "ECF1", "testing": "ECF2"}

# Klasörleri sırasıyla gez ve verileri diske kaydet
for split in SPLITS:
    print(f"\n--- {split.upper()} Klasörü İşleniyor (CNN İçin Mel-Spektrogram) ---")
    real_dir = os.path.join(BASE_DIR, split, "real")
    fake_dir = os.path.join(BASE_DIR, split, "fake")
    
    print("Gerçek sesler çıkarılıyor...")
    real_features, real_labels = process_directory(real_dir, label=1)
    
    print("Sahte sesler çıkarılıyor...")
    fake_features, fake_labels = process_directory(fake_dir, label=0)
    
    # RAM tasarrufu için uygun veri tiplerine çeviriyoruz (float32 ve int8)
    x = np.array(real_features + fake_features, dtype=np.float32)
    y = np.array(real_labels + fake_labels, dtype=np.int8)
    
    prefix = PREFIX_MAP[split]
    x_path = f"{prefix}-x_{split}_mel.npy"
    y_path = f"{prefix}-y_{split}_mel.npy"
    np.save(x_path, x)
    np.save(y_path, y)
    
    print(f"  → {x_path} kaydedildi | Şekil: {x.shape}")
    print(f"  → {y_path} kaydedildi | Şekil: {y.shape}")