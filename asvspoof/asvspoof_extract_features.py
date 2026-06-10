"""
ASVspoof 2019 LA — Özellik Çıkarma (MFCC + Mel-Spektrogram + Standart Spektrogram)

Giriş:
  Dataset/ASVspoof2019_LA/
    ├── ASVspoof2019_LA_train/flac/
    ├── ASVspoof2019_LA_dev/flac/
    ├── ASVspoof2019_LA_eval/flac/
    └── ASVspoof2019_LA_cm_protocols/
          ├── ASVspoof2019.LA.cm.train.trn.txt
          ├── ASVspoof2019.LA.cm.dev.trl.txt
          └── ASVspoof2019.LA.cm.eval.trl.txt

Çıktılar (her modalite kendi konuşmacı-grup dizisiyle, GroupKFold için hizalı):
  ML   (194-boyut MFCC):  ASV_ml_x_{split}.npy,   ASV_ml_y_{split}.npy,   ASV_ml_groups_{split}.npy
  MEL  (128×216):         ASV_mel_x_{split}.npy,  ASV_mel_y_{split}.npy,  ASV_mel_groups_{split}.npy
  SPEC (257×216):         ASV_spec_x_{split}.npy, ASV_spec_y_{split}.npy, ASV_spec_groups_{split}.npy

NOTLAR:
  * groups = KONUŞMACI ID'si (protokol dosyasından). GroupKFold konuşmacı sızıntısını
    böylece önler. Etiket: 1=bonafide (gerçek), 0=spoof (sahte).
  * Ses dosyası dosya başına SADECE BİR KEZ decode edilir; istenen tüm modaliteler
    aynı sinyalden hesaplanır (121 bin dosya için kritik hız kazancı).
  * MEL/SPEC dizileri belleğe sığmayabilir (eval-spec ~16 GB). Bu yüzden diske
    doğrudan memory-mapped (.npy) olarak yazılır — RAM'i doldurmaz.

Kullanım:
  python asvspoof/asvspoof_extract_features.py                # ml + mel + spec
  python asvspoof/asvspoof_extract_features.py --ml-only      # sadece MFCC (hızlı, ~küçük)
  python asvspoof/asvspoof_extract_features.py --mel-only     # sadece Mel
  python asvspoof/asvspoof_extract_features.py --spec-only    # sadece Standart Spektrogram
  python asvspoof/asvspoof_extract_features.py --cnn-only     # mel + spec (ML yok)
"""

import os
import sys
import numpy as np
import librosa
from numpy.lib.format import open_memmap
from tqdm import tqdm

# ─────────────────────────────────────────────
# YAPILANDIRMA  (ses okuma ve öznitelik çıkarım parametreleri)
# ─────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASV_DIR = os.path.join(_BASE_DIR, "Dataset", "ASVspoof2019_LA")
FEATURES_DIR = os.path.join(_BASE_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

SPLITS = {
    "train": {
        "flac_dir": os.path.join(ASV_DIR, "ASVspoof2019_LA_train", "flac"),
        "protocol": os.path.join(ASV_DIR, "ASVspoof2019_LA_cm_protocols",
                                 "ASVspoof2019.LA.cm.train.trn.txt"),
    },
    "dev": {
        "flac_dir": os.path.join(ASV_DIR, "ASVspoof2019_LA_dev", "flac"),
        "protocol": os.path.join(ASV_DIR, "ASVspoof2019_LA_cm_protocols",
                                 "ASVspoof2019.LA.cm.dev.trl.txt"),
    },
    "eval": {
        "flac_dir": os.path.join(ASV_DIR, "ASVspoof2019_LA_eval", "flac"),
        "protocol": os.path.join(ASV_DIR, "ASVspoof2019_LA_cm_protocols",
                                 "ASVspoof2019.LA.cm.eval.trl.txt"),
    },
}

SR = 22050
DURATION = 5.0
HOP_LENGTH = 512
MAX_PAD_LEN = int((SR * DURATION) / HOP_LENGTH) + 1   # 216

N_MELS = 128                                          # mel  → (128, 216)
N_FFT = 512                                           # spec → (257, 216)
N_FREQ_BINS = (N_FFT // 2) + 1                        # 257

MEL_SHAPE = (N_MELS, MAX_PAD_LEN)
SPEC_SHAPE = (N_FREQ_BINS, MAX_PAD_LEN)


# ─────────────────────────────────────────────
# PROTOCOL OKUYUCU
# ─────────────────────────────────────────────
def parse_protocol(protocol_path):
    """
    Format: SPEAKER_ID  AUDIO_ID  -  ATTACK_ID  KEY
      LA_0079  LA_T_1138215  -  -    bonafide
      LA_0079  LA_T_6840632  -  A04  spoof
    Returns: [(audio_id, speaker_id, label_int, attack_id), ...]  label: 1=bonafide, 0=spoof
    """
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            speaker_id = parts[0]
            audio_id = parts[1]
            key = parts[-1]                              # son alan daima bonafide/spoof
            attack_id = parts[3] if len(parts) >= 5 else "-"
            label = 1 if key == "bonafide" else 0
            entries.append((audio_id, speaker_id, label, attack_id))
    return entries


# ─────────────────────────────────────────────
# ÖZNİTELİK HESAPLAYICILAR  (hepsi tek bir y sinyalinden)
# ─────────────────────────────────────────────
def _mfcc_194(y, sr):
    """194 boyutlu öznitelik vektörü (MFCC istatistikleri + spektral öznitelikler + chroma)."""
    f = []
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    f.extend(np.mean(mfccs, axis=1)); f.extend(np.std(mfccs, axis=1))
    f.extend(np.min(mfccs, axis=1));  f.extend(np.max(mfccs, axis=1))
    zcr = librosa.feature.zero_crossing_rate(y);           f.extend([np.mean(zcr), np.std(zcr)])
    cent = librosa.feature.spectral_centroid(y=y, sr=sr);  f.extend([np.mean(cent), np.std(cent)])
    band = librosa.feature.spectral_bandwidth(y=y, sr=sr); f.extend([np.mean(band), np.std(band)])
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr);   f.extend([np.mean(roll), np.std(roll)])
    rms = librosa.feature.rms(y=y);                        f.extend([np.mean(rms), np.std(rms)])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f.extend(np.mean(chroma, axis=1)); f.extend(np.std(chroma, axis=1))
    return np.array(f, dtype=np.float32)


def _pad(mat):
    """Zaman eksenini MAX_PAD_LEN'e sabitler (kırp ya da sıfır-doldur)."""
    if mat.shape[1] > MAX_PAD_LEN:
        return mat[:, :MAX_PAD_LEN]
    return np.pad(mat, ((0, 0), (0, MAX_PAD_LEN - mat.shape[1])), mode="constant")


def _mel_128(y, sr):
    """128×216 boyutlu mel-spektrogram (dB ölçeğinde)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    return _pad(librosa.power_to_db(mel, ref=np.max)).astype(np.float32)


def _spec_257(y, sr):
    """257×216 boyutlu STFT log-genlik spektrogramı."""
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return _pad(librosa.amplitude_to_db(np.abs(stft), ref=np.max)).astype(np.float32)


# ─────────────────────────────────────────────
# ANA İŞLEM DÖNGÜSÜ  (split başına, tek-decode, memmap yazımı)
# ─────────────────────────────────────────────
def process_split(split_name, cfg, do_ml, do_mel, do_spec):
    print(f"\n{'='*65}\n  {split_name.upper()} işleniyor...\n{'='*65}")

    entries = parse_protocol(cfg["protocol"])
    flac_dir = cfg["flac_dir"]

    # Diskte gerçekten var olan dosyalar (decode etmeden ön-tarama → memmap boyutu)
    present = [(aid, spk, lab)
               for (aid, spk, lab, _atk) in entries
               if os.path.exists(os.path.join(flac_dir, f"{aid}.flac"))]
    n = len(present)
    missing = len(entries) - n

    n_bona = sum(1 for _, _, l in present if l == 1)
    print(f"  Protokol: {len(entries)} kayıt | diskte: {n} | eksik: {missing}")
    print(f"  Dağılım (diskte): {n_bona} bonafide + {n - n_bona} spoof")
    if n == 0:
        print("  ⚠ İşlenecek dosya yok, atlanıyor.")
        return

    # CNN modaliteleri için diske doğrudan memory-mapped diziler (RAM dostu)
    mel_mm = open_memmap(os.path.join(FEATURES_DIR, f"ASV_mel_x_{split_name}.npy"),
                         mode="w+", dtype=np.float32, shape=(n, *MEL_SHAPE)) if do_mel else None
    spec_mm = open_memmap(os.path.join(FEATURES_DIR, f"ASV_spec_x_{split_name}.npy"),
                          mode="w+", dtype=np.float32, shape=(n, *SPEC_SHAPE)) if do_spec else None

    ml_x = []                       # MFCC küçük → listede tutulur
    y_list, g_list = [], []         # tüm modaliteler için ORTAK (tek-decode → tek hizalama)

    w = 0                           # başarıyla yazılan satır sayacı
    bad = 0
    for audio_id, speaker_id, label in tqdm(present, desc=f"  {split_name}"):
        path = os.path.join(flac_dir, f"{audio_id}.flac")
        try:
            y, sr = librosa.load(path, sr=SR, duration=DURATION)
            mfcc = _mfcc_194(y, sr) if do_ml else None
            mel = _mel_128(y, sr) if do_mel else None
            spec = _spec_257(y, sr) if do_spec else None
        except Exception:
            bad += 1
            continue

        if do_ml:
            ml_x.append(mfcc)
        if do_mel:
            mel_mm[w] = mel
        if do_spec:
            spec_mm[w] = spec
        y_list.append(label)
        g_list.append(speaker_id)
        w += 1

    y_arr = np.array(y_list, dtype=np.int8)
    g_arr = np.array(g_list)
    if bad:
        print(f"  ⚠ {bad} dosya decode edilemedi ve atlandı.")

    # ── ML kaydet ──
    if do_ml:
        np.save(os.path.join(FEATURES_DIR, f"ASV_ml_x_{split_name}.npy"),
                np.array(ml_x, dtype=np.float32))
        np.save(os.path.join(FEATURES_DIR, f"ASV_ml_y_{split_name}.npy"), y_arr)
        np.save(os.path.join(FEATURES_DIR, f"ASV_ml_groups_{split_name}.npy"), g_arr)
        print(f"  ✔ ML  : x=({w}, 194)  konuşmacı={len(np.unique(g_arr))}")

    # ── MEL kaydet (gerekiyorsa kırp) ──
    if do_mel:
        _finalize_memmap(mel_mm, os.path.join(FEATURES_DIR, f"ASV_mel_x_{split_name}.npy"),
                         w, n, MEL_SHAPE)
        np.save(os.path.join(FEATURES_DIR, f"ASV_mel_y_{split_name}.npy"), y_arr)
        np.save(os.path.join(FEATURES_DIR, f"ASV_mel_groups_{split_name}.npy"), g_arr)
        print(f"  ✔ MEL : x=({w}, {MEL_SHAPE[0]}, {MEL_SHAPE[1]})  "
              f"~{w * int(np.prod(MEL_SHAPE)) * 4 / 1024**3:.2f} GB")

    # ── SPEC kaydet (gerekiyorsa kırp) ──
    if do_spec:
        _finalize_memmap(spec_mm, os.path.join(FEATURES_DIR, f"ASV_spec_x_{split_name}.npy"),
                         w, n, SPEC_SHAPE)
        np.save(os.path.join(FEATURES_DIR, f"ASV_spec_y_{split_name}.npy"), y_arr)
        np.save(os.path.join(FEATURES_DIR, f"ASV_spec_groups_{split_name}.npy"), g_arr)
        print(f"  ✔ SPEC: x=({w}, {SPEC_SHAPE[0]}, {SPEC_SHAPE[1]})  "
              f"~{w * int(np.prod(SPEC_SHAPE)) * 4 / 1024**3:.2f} GB")


def _finalize_memmap(mm, path, w, n, shape):
    """w == n ise memmap zaten doğru; değilse ilk w satıra parça parça kırpar."""
    mm.flush()
    del mm
    if w == n:
        return
    # Nadir durum (decode hatası): doğru boyutlu yeni dosyaya parça parça kopyala
    src = np.load(path, mmap_mode="r")
    tmp = path + ".tmp"
    dst = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(w, *shape))
    CH = 1000
    for i in range(0, w, CH):
        dst[i:i + CH] = src[i:i + CH]
    dst.flush(); del dst, src
    os.replace(tmp, path)


if __name__ == "__main__":
    flags = set(sys.argv[1:])
    if "--ml-only" in flags:
        do_ml, do_mel, do_spec = True, False, False
    elif "--mel-only" in flags:
        do_ml, do_mel, do_spec = False, True, False
    elif "--spec-only" in flags:
        do_ml, do_mel, do_spec = False, False, True
    elif "--cnn-only" in flags:
        do_ml, do_mel, do_spec = False, True, True
    else:
        do_ml, do_mel, do_spec = True, True, True

    mods = [m for m, on in [("ML", do_ml), ("MEL", do_mel), ("SPEC", do_spec)] if on]
    print(f"ASVspoof 2019 LA — Özellik Çıkarma ({' + '.join(mods)})")
    print(f"Dataset: {ASV_DIR}")

    if not os.path.isdir(ASV_DIR):
        print(f"\n❌ HATA: Dataset dizini yok: {ASV_DIR}")
        print("  Beklenen yapı: Dataset/ASVspoof2019_LA/ASVspoof2019_LA_{train,dev,eval}/flac/ ...")
        sys.exit(1)

    for split_name, cfg in SPLITS.items():
        if not os.path.exists(cfg["protocol"]):
            print(f"\n❌ HATA: Protokol bulunamadı: {cfg['protocol']}")
            sys.exit(1)
        process_split(split_name, cfg, do_ml, do_mel, do_spec)

    print("\n" + "=" * 65)
    print("  ✔ Tüm istenen özellikler çıkarıldı.")
    print("=" * 65)
