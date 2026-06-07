import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def find_first_audio(directory, base_name):
    """
    Veri setinde aynı dosya adı hem .wav hem .mp3 olarak bulunabildiği için
    her iki uzantıyı da deneyen yardımcı fonksiyon.
    """
    for ext in (".wav", ".mp3", ".flac", ".ogg"):
        candidate = os.path.join(directory, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    # Hâlâ bulunamadıysa glob ile dene
    matches = glob.glob(os.path.join(directory, base_name + ".*"))
    return matches[0] if matches else None


# Veri setindeki training/real ve training/fake klasörlerinden birer örnek dosya çektik
REAL_AUDIO_PATH = find_first_audio("Dataset/for-original/training/real", "file1")
FAKE_AUDIO_PATH = find_first_audio("Dataset/for-original/training/fake", "file172")

def plot_audio_features(file_path, title_prefix):
    # Sesi yükle (sr=None orijinal örnekleme hızını korur)
    y, sr = librosa.load(file_path, sr=None) 
    
    plt.figure(figsize=(14, 10))

    # 1. Dalga Formu (Waveform) - Zaman düzlemindeki ham ses
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"{title_prefix} - Dalga Formu")

    # 2. Standart Spektrogram - Frekansların zamana göre değişimi
    plt.subplot(4, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title_prefix} - Standart Spektrogram")

    # 3. Mel-Spektrogram - İnsan kulağının duyumuna göre ölçeklendirilmiş spektrogram
    plt.subplot(4, 1, 3)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    M_db = librosa.power_to_db(M, ref=np.max)
    librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title_prefix} - Mel-Spektrogram (128 Bant)")

    # 4. MFCC - Sesin tınısını (timbre) belirleyen öznitelikler
    plt.subplot(4, 1, 4)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"{title_prefix} - MFCC (40 Katsayı)")

    plt.tight_layout()
    plt.show()

# Dosyalar mevcutsa işlemleri çalıştır
if REAL_AUDIO_PATH and os.path.exists(REAL_AUDIO_PATH):
    print(f"{REAL_AUDIO_PATH} işleniyor...")
    plot_audio_features(REAL_AUDIO_PATH, "Gerçek Ses")
else:
    print("HATA: Gerçek ses dosyası bulunamadı (.wav / .mp3 / .flac / .ogg).")

if FAKE_AUDIO_PATH and os.path.exists(FAKE_AUDIO_PATH):
    print(f"{FAKE_AUDIO_PATH} işleniyor...")
    plot_audio_features(FAKE_AUDIO_PATH, "Deepvoice (Sahte) Ses")
else:
    print("HATA: Sahte ses dosyası bulunamadı (.wav / .mp3 / .flac / .ogg).")