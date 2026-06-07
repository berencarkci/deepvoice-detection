import os
import librosa
import numpy as np
from tqdm import tqdm

BASE_DIR = "Dataset/for-original"
SPLITS = ["training", "validation", "testing"]

# CNN ile adil karşılaştırma için aynı SR ve süre kullanılır
SR       = 22050
DURATION = 5.0

def extract_advanced_features(file_path):
    """
    Bir ses dosyasından MFCC, Zero Crossing Rate, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, RMS Energy, Chroma STFT özniteliklerini çıkarır ve bu özniteliklerin ortalama, standart sapma, min, max değerlerini çıkarır.
    Bu özniteliklerin toplamı 194 adet öznitelik olur.
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
        features = []

        # 1. MFCC (40 Katsayı) x 4 öznitelik -> Ortalama, Standart Sapma, Min, Max (Toplam 160 Öznitelik)
        # Sesin tını yapısı
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.extend(np.min(mfccs, axis=1))
        features.extend(np.max(mfccs, axis=1))

        # 2. Zero Crossing Rate -> Sinyalin işaret değiştirme hızı (2 adet)
        # Sinyalin işaret değiştirme hızı
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])

        # 3. Spectral Centroid -> Sesin frekans merkez ağırlığı (2 Öznitelik)
        # Sesin frekans merkez ağırlığı
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(cent), np.std(cent)])

        # 4. Spectral Bandwidth -> Frekans bant genişliği (2 Öznitelik)
        # Frekans bant genişliği
        band = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([np.mean(band), np.std(band)])

        # 5. Spectral Rolloff -> Yüksek frekansların kesim noktası (2 Öznitelik)
        # Yüksek frekansların kesim noktası
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(rolloff), np.std(rolloff)])

        # 6. RMS Energy -> Sinyalin güç/enerji seviyesi (2 Öznitelik)
        # Sinyalin güç/enerji seviyesi
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])

        # 7. Chroma STFT -> Notaların/Tonların enerji dağılımı (24 Öznitelik)
        # Notaların/Tonların enerji dağılımı
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Toplam: 160 + 2 + 2 + 2 + 2 + 2 + 24 = 194 Boyutlu Vektör
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        return None

# Her klasörde tüm .wav ve .mp3 dosyaları işlenir
def process_directory(directory, label):
    features, labels = [], []
    valid_files = [f for f in os.listdir(directory) if f.endswith(('.wav', '.mp3'))]
    
    for file_name in tqdm(valid_files, desc=f"Etiket: {label}"):
        data = extract_advanced_features(os.path.join(directory, file_name))
        if data is not None:
            features.append(data)
            labels.append(label)
            
    return features, labels

# EMF prefix = "Extract ML Features" (ECF=mel-spec, ESF=std-spec ile uyumlu)
PREFIX_MAP = {"training": "EMF0", "validation": "EMF1", "testing": "EMF2"}

for split in SPLITS:
    print(f"\n--- {split.upper()} Klasörü İşleniyor ---")
    real_dir = os.path.join(BASE_DIR, split, "real")
    fake_dir = os.path.join(BASE_DIR, split, "fake")
    
    print("Gerçek sesler çıkarılıyor...")
    real_features, real_labels = process_directory(real_dir, label=1)
    
    print("Sahte sesler çıkarılıyor...")
    fake_features, fake_labels = process_directory(fake_dir, label=0)
    
    x = np.array(real_features + fake_features, dtype=np.float32)
    y = np.array(real_labels + fake_labels, dtype=np.int8)
    
    prefix = PREFIX_MAP[split]
    x_path = f"{prefix}-x_{split}_mfcc.npy"
    y_path = f"{prefix}-y_{split}_labels.npy"
    np.save(x_path, x)
    np.save(y_path, y)
    
    print(f"  → {x_path} kaydedildi | Şekil: {x.shape}")
    print(f"  → {y_path} kaydedildi | Şekil: {y.shape}")