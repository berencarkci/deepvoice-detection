import numpy as np
import librosa
from datasets import load_dataset
import torch

# Kaydedilecek dosyalar
X_SAVE_PATH = "X_train_dfadd_mel.npy"
Y_SAVE_PATH = "y_train_dfadd_mel.npy"

# CNN için sabit parametreler
DURATION = 5.0
SR = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_PAD_LEN = int((SR * DURATION) / HOP_LENGTH) + 1 # 216

def process_audio_array(audio_array, orig_sr):
    """Hugging Face'den gelen ham ses dizisini Mel-Spektrogram'a çevirir."""
    try:
        # Örnekleme hızını librosa standardına (22050) eşitle
        if orig_sr != SR:
            audio_array = librosa.resample(y=audio_array, orig_sr=orig_sr, target_sr=SR)
        
        # Sadece ilk 5 saniyeyi al
        max_samples = int(SR * DURATION)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
            
        # Mel-Spektrogram çıkar
        mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Boyut Sabitleme (Padding veya Truncating)
        if mel_spec_db.shape[1] > MAX_PAD_LEN:
            mel_spec_db = mel_spec_db[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mel_spec_db
    except Exception as e:
        return None

print("Hugging Face üzerinden DFADD veri seti akışı (streaming) başlatılıyor...")
# streaming=True parametresi veriyi diske indirmeden anlık olarak okur
dataset = load_dataset("isjwdu/DFADD", split="train", streaming=True)

features = []
labels = []
processed_count = 0

print("İnternet üzerinden sesler çekilip işleniyor. (Ağ hızına bağlı olarak vakit alabilir)...")

# Akış üzerinden gelen her bir ses verisini döngüyle yakala
for item in dataset:
    # Hugging Face 'audio' sözlüğünde ham veri (array) ve örnekleme hızını (sampling_rate) tutar
    audio_array = item['audio']['array']
    orig_sr = item['audio']['sampling_rate']
    
    # Etiketi metin olarak al ve küçük harfe çevir
    raw_label = str(item['label']).lower()
    
    # 'spoofed' ise 0 (Sahte), 'bonafide' veya başka bir şeyse 1 (Gerçek) yap
    if raw_label == 'spoofed':
        label = 0
    else:
        label = 1
        
    data = process_audio_array(audio_array, orig_sr)
    
    if data is not None:
        features.append(data)
        labels.append(label)
        processed_count += 1
        
        # Süreci takip edebilmek için her 100 dosyada bir bilgi yazdır
        if processed_count % 100 == 0:
            print(f"İşlenen dosya sayısı: {processed_count}")

# NumPy dizilerine çevir
X_train_mel = np.array(features, dtype=np.float32)
y_train_mel = np.array(labels, dtype=np.int8)

print("\nVeriler diske kaydediliyor...")
np.save(X_SAVE_PATH, X_train_mel)
np.save(Y_SAVE_PATH, y_train_mel)

print(f"İşlem tamam! Toplam işlenen dosya: {len(X_train_mel)}")
print(f"X_train_dfadd_mel boyutu: {X_train_mel.shape}")