"""
FORMAT CONFOUND İSPATI
======================

Hipotez: FoR-original veri setinde gerçek (.wav) ve sahte (.mp3) dosyaların
KAPSAYICI FORMATI etikete sızıyor. Model "deepfake artefaktı" yerine "mp3
sıkıştırma izi" öğreniyor. Resmî split'in test seti tamamen .wav olduğu için
bu kestirme orada çalışmaz (~%80); k-fold ile her şey havuzlanınca mp3-fake'ler
test fold'larına dağılır ve kestirme geri gelir (~%99).

Bu script bunu 3 bağımsız deneyle ispatlar:

  EXP-1  Confound'un büyüklüğü (ses çözmeden, sadece dosya listesinden):
         "wav→gerçek, mp3→sahte" diyen, sesi HİÇ dinlemeyen kuralın doğruluğu.

  EXP-2  Nedensel kanıt (mevcut .npy öznitelikleriyle, ses çözmeden):
         (a) Havuzlanmış GroupKFold  → ~%99  (sızıntılı kurulum)
         (b) Resmî split (train+val'da eğit, sadece tamamı-wav test setinde test)
             → büyük düşüş. Format ipucu kalkınca doğruluk çöküyor.

  EXP-3  Format sesin İÇİNE sızıyor mu? (dengeli bir örneklemi çözer):
         (a) SADECE sahte dosyalar arasında mp3-fake vs wav-fake'i ses
             özniteliklerinden tahmin et. İçerik aynı (ikisi de fake), tek fark
             kap formatı. Yüksek doğruluk → format, modelin kullandığı
             özniteliklerin içinde gömülü.
         (b) Gerçek/sahte modeli kur; sahteleri yakalama oranını mp3 ve wav
             alt kümelerinde ayrı ayrı ölç. mp3-fake'ler kolay, wav-fake'ler zor
             → model formata yaslanıyor.

Çalıştırma:
    cd <proje-dizini>
    .venv/bin/python experiments/prove_format_confound.py
"""

import os
import sys
import numpy as np
from multiprocessing import Pool

import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(_BASE_DIR, "Dataset", "for-original")
SPLITS = ["training", "validation", "testing"]

SR = 22050
DURATION = 5.0
RNG = np.random.default_rng(42)

# EXP-3 örneklem boyutu (sınıf başına dosya). Düşürürsen daha hızlı.
SAMPLE_PER_BUCKET = 1500
N_ESTIMATORS = 150


# ───────────────────────── Öznitelik çıkarımı ─────────────────────────
# (extract_ml_features.py ile BİREBİR aynı 194 boyutlu vektör)
def extract_advanced_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
        f = []
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        f.extend(np.mean(mfccs, axis=1)); f.extend(np.std(mfccs, axis=1))
        f.extend(np.min(mfccs, axis=1));  f.extend(np.max(mfccs, axis=1))
        zcr = librosa.feature.zero_crossing_rate(y)
        f.extend([np.mean(zcr), np.std(zcr)])
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        f.extend([np.mean(cent), np.std(cent)])
        band = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        f.extend([np.mean(band), np.std(band)])
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        f.extend([np.mean(roll), np.std(roll)])
        rms = librosa.feature.rms(y=y)
        f.extend([np.mean(rms), np.std(rms)])
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        f.extend(np.mean(chroma, axis=1)); f.extend(np.std(chroma, axis=1))
        return np.array(f, dtype=np.float32)
    except Exception:
        return None


def _worker(path):
    return path, extract_advanced_features(path)


# ───────────────────────── Dosya envanteri ─────────────────────────
def enumerate_files():
    """(path, split, label, fmt) listesi. label: gerçek=1 sahte=0. fmt: 'wav'/'mp3'. Ses ÇÖZMEZ."""
    rows = []
    for split in SPLITS:
        for sub, label in [("real", 1), ("fake", 0)]:
            d = os.path.join(BASE_DIR, split, sub)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                ext = os.path.splitext(fn)[1].lower().lstrip(".")
                if ext in ("wav", "mp3"):
                    rows.append((os.path.join(d, fn), split, label, ext))
    return rows


def banner(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ════════════════════════════ EXP-1 ════════════════════════════
def exp1_confound_magnitude(rows):
    banner("EXP-1 | Confound büyüklüğü — sesi DİNLEMEYEN kuralın doğruluğu")
    labels = np.array([r[2] for r in rows])
    fmts = np.array([r[3] for r in rows])

    print("\nSplit × Etiket × Format dağılımı:")
    print(f"  {'split':<12}{'sınıf':<8}{'wav':>8}{'mp3':>8}")
    for split in SPLITS:
        for sub, lab in [("real(1)", 1), ("fake(0)", 0)]:
            m = np.array([(r[1] == split and r[2] == lab) for r in rows])
            w = int(((fmts == "wav") & m).sum())
            mp = int(((fmts == "mp3") & m).sum())
            print(f"  {split:<12}{sub:<8}{w:>8}{mp:>8}")

    # Sesi hiç dinlemeyen, SADECE kaba bakan kural: wav→gerçek, mp3→sahte
    pred = np.where(fmts == "wav", 1, 0)
    acc = accuracy_score(labels, pred)
    print(f"\n  >>> 'wav→gerçek, mp3→sahte' kuralının doğruluğu: {acc:.4f}")
    print("      (Sesi hiç analiz etmeyen, yalnızca dosya uzantısına bakan kural.)")
    print(f"      Doğruluğun ~%{acc*100:.0f}'ı tek başına format korelasyonuyla açıklanabilir.")
    return acc


# ════════════════════════════ EXP-2 ════════════════════════════
def _load_emf():
    p = lambda n: os.path.join(_BASE_DIR, n)
    x = [np.load(p(f"EMF{i}-x_{s}_mfcc.npy"))
         for i, s in [(0, "training"), (1, "validation"), (2, "testing")]]
    y = [np.load(p(f"EMF{i}-y_{s}_labels.npy"))
         for i, s in [(0, "training"), (1, "validation"), (2, "testing")]]
    g = [np.load(p(f"EMF_groups_{s}.npy")) for s in SPLITS]
    return x, y, g


def exp2_causal(rows):
    banner("EXP-2 | Nedensel kanıt — format ipucunu kaldırınca doğruluk çöküyor")
    try:
        x, y, g = _load_emf()
    except FileNotFoundError as e:
        print(f"  (EMF .npy bulunamadı, EXP-2 atlanıyor: {e})")
        return None

    X = np.concatenate(x); Y = np.concatenate(y); G = np.concatenate(g)

    # (a) Havuzlanmış GroupKFold (train_ml_models.py ile aynı sızıntılı kurulum)
    gkf = GroupKFold(n_splits=5)
    accs = []
    for tr, te in gkf.split(X, Y, groups=G):
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
        clf.fit(sc.transform(X[tr]), Y[tr])
        accs.append(accuracy_score(Y[te], clf.predict(sc.transform(X[te]))))
    pooled = float(np.mean(accs))
    print(f"\n  (a) Havuzlanmış 5-Fold (sızıntılı)         : {pooled:.4f}")

    # (b) Resmî split: train+val'da eğit, SADECE testing'de test (testing fake'leri %100 wav)
    Xtr = np.concatenate([x[0], x[1]]); Ytr = np.concatenate([y[0], y[1]])
    Xte, Yte = x[2], y[2]
    sc = StandardScaler().fit(Xtr)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
    clf.fit(sc.transform(Xtr), Ytr)
    official = accuracy_score(Yte, clf.predict(sc.transform(Xte)))
    print(f"  (b) Resmî split (test seti tamamı .wav-fake): {official:.4f}")
    print(f"\n  >>> Tek değişen şey partition. Format ipucu olan test fold'larında "
          f"{pooled:.2f},\n      ipucunun olmadığı resmî test setinde {official:.2f}. "
          f"Fark = sızıntı.")
    return pooled, official


# ════════════════════════════ EXP-3 ════════════════════════════
def _sample(rows, label, fmt, n):
    pool = [r[0] for r in rows if r[2] == label and r[3] == fmt]
    if len(pool) <= n:
        return pool
    idx = RNG.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def _extract_many(paths, tag):
    print(f"    [{tag}] {len(paths)} dosya çözülüyor...")
    feats, keep = [], []
    with Pool() as pool:
        for path, vec in pool.imap(_worker, paths, chunksize=16):
            if vec is not None:
                feats.append(vec); keep.append(path)
    return np.array(feats, dtype=np.float32), keep


def exp3_format_in_audio(rows):
    banner("EXP-3 | Format sesin İÇİNE sızıyor mu? (öznitelikleri çözerek test)")

    real_wav = _sample(rows, 1, "wav", SAMPLE_PER_BUCKET)
    fake_mp3 = _sample(rows, 0, "mp3", SAMPLE_PER_BUCKET)
    fake_wav = _sample(rows, 0, "wav", SAMPLE_PER_BUCKET)
    print(f"  Örneklem: real-wav={len(real_wav)}, fake-mp3={len(fake_mp3)}, fake-wav={len(fake_wav)}")

    Xrw, _ = _extract_many(real_wav, "real-wav")
    Xfm, _ = _extract_many(fake_mp3, "fake-mp3")
    Xfw, _ = _extract_many(fake_wav, "fake-wav")

    # (a) SADECE sahteler: mp3-fake vs wav-fake format sınıflandırması.
    #     İçerik aynı (hepsi fake), tek fark kap formatı → saf format sinyali.
    Xf = np.concatenate([Xfm, Xfw])
    yf = np.concatenate([np.ones(len(Xfm)), np.zeros(len(Xfw))])  # 1=mp3, 0=wav
    sc = StandardScaler()
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
    pred = cross_val_predict(
        clf, sc.fit_transform(Xf), yf,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
    )
    fmt_acc = accuracy_score(yf, pred)
    print(f"\n  (a) mp3-fake vs wav-fake ayırt etme doğruluğu: {fmt_acc:.4f}")
    print("      (İkisi de 'sahte'; sadece format farklı. Yüksek doğruluk = format,")
    print("       modelin kullandığı 194 özniteliğin İÇİNE sızmış demektir.)")

    # (b) FORMAT BAĞIMLILIĞI İZOLASYON TESTİ.
    #     Eğitim: real-wav + fake-MP3  → burada format etiketi MÜKEMMEL belirliyor
    #             (resmî training split'in durumu: gerçek=wav, sahte=mp3).
    #     Test  : AYRI real-wav + fake-WAV → ikisi de wav, format ipucu İŞE YARAMAZ.
    #     Eğer model wav-fake'leri yakalayamıyorsa → formata yaslanmış demektir.
    #     Yakalayabiliyorsa → içerik artefaktları da genelleşiyor demektir.
    half = len(Xrw) // 2
    Xr_tr, Xr_te = Xrw[:half], Xrw[half:]

    X_tr = np.concatenate([Xr_tr, Xfm])
    y_tr = np.concatenate([np.ones(len(Xr_tr)), np.zeros(len(Xfm))])
    sc2 = StandardScaler().fit(X_tr)
    clf2 = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
    clf2.fit(sc2.transform(X_tr), y_tr)

    # Test setindeki gerçekleri doğru tanıma + sahteleri (wav) yakalama
    real_ok = (clf2.predict(sc2.transform(Xr_te)) == 1).mean()
    caught_wavfake = (clf2.predict(sc2.transform(Xfw)) == 0).mean()
    print(f"\n  (b) İZOLASYON — eğitim 'gerçek=wav / sahte=mp3', test ikisi de wav:")
    print(f"        gerçek-wav doğru tanındı     : {real_ok:.4f}")
    print(f"        wav-fake yakalandı (recall)  : {caught_wavfake:.4f}")
    if caught_wavfake < 0.85:
        print("      → wav-fake'ler kaçtı: model büyük ölçüde FORMATA yaslanıyor.")
    else:
        print("      → wav-fake'ler de yakalandı: içerik artefaktları da ayırt edici;")
        print("        format tek sebep değil, ama EXP-3a'ya göre ek bir kestirme olarak MEVCUT.")
    return fmt_acc, real_ok, caught_wavfake


# ════════════════════════════ MAIN ════════════════════════════
if __name__ == "__main__":
    if not os.path.isdir(BASE_DIR):
        print(f"HATA: {BASE_DIR} yok.", file=sys.stderr)
        raise SystemExit(1)

    rows = enumerate_files()
    print(f"Toplam dosya: {len(rows)}")

    a1 = exp1_confound_magnitude(rows)
    r2 = exp2_causal(rows)
    a3, rok, cwf = exp3_format_in_audio(rows)

    banner("ÖZET")
    print(f"  EXP-1  Sadece-uzantı kuralı doğruluğu              : {a1:.3f}")
    if r2:
        print(f"  EXP-2  Havuzlanmış k-fold vs resmî split           : {r2[0]:.3f} → {r2[1]:.3f}")
    print(f"  EXP-3a Format ses özniteliklerinden tahmin edilir  : {a3:.3f}")
    print(f"  EXP-3b İzolasyon — wav-fake recall (sadece mp3'te eğitilmiş): {cwf:.3f}")
    print("\n  SONUÇ: Etiket büyük ölçüde dosya formatıyla (.mp3 fake / .wav real)")
    print("  korele. Bu format sesin içine sızdığı için model bunu kestirme olarak")
    print("  kullanıyor. K-fold mp3-fake'leri test fold'larına dağıtınca kestirme")
    print("  geri geliyor → yapay %99. Resmî split (tamamı-wav test) bunu engellediği")
    print("  için dürüst ~%80 veriyor.")
