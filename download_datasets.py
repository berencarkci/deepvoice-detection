"""
Veri seti indirme yardımcısı.

Bu betik, projenin beklediği klasör yapısını oluşturacak şekilde veri setlerini
indirir ve açar:

  Dataset/ASVspoof2019_LA/ASVspoof2019_LA_{train,dev,eval}/flac/ ...
  Dataset/for-original/{training,validation,testing}/{real,fake}/ ...

Kullanım:
    python download_datasets.py asvspoof     # yalnız ASVspoof 2019 LA (~7 GB)
    python download_datasets.py for          # yalnız Fake-or-Real (Kaggle)
    python download_datasets.py all          # her ikisi (varsayılan)

Notlar:
  * ASVspoof 2019 LA, Edinburgh DataShare üzerinden doğrudan indirilir (kimlik
    doğrulama gerektirmez).
  * Fake-or-Real, Kaggle aynası üzerinden indirilir; `kaggle` komut satırı aracı
    ve API anahtarı (~/.kaggle/kaggle.json) gerektirir.
  * Hedef klasör zaten varsa ilgili adım atlanır.
"""

import os
import sys
import shutil
import zipfile
import subprocess

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(_BASE_DIR, "Dataset")

ASVSPOOF_URL = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
    "?sequence=3&isAllowed=y"
)
ASVSPOOF_DIR = os.path.join(DATASET_DIR, "ASVspoof2019_LA")
FOR_DIR = os.path.join(DATASET_DIR, "for-original")
FOR_KAGGLE_SLUG = "mohammedabdeldayem/the-fake-or-real-dataset"


def _download(url, dest):
    """url'i dest dosyasına indirir. wget/curl varsa devam ettirme (resume) destekli."""
    if shutil.which("wget"):
        cmd = ["wget", "-c", "-O", dest, url]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "-o", dest, url]
    else:
        import urllib.request
        print(f"  indiriliyor (urllib): {url}")
        urllib.request.urlretrieve(url, dest)
        return
    print(f"  indiriliyor: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _unzip(zip_path, dest_dir):
    print(f"  açılıyor: {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def download_asvspoof():
    print("\n=== ASVspoof 2019 LA ===")
    if os.path.isdir(ASVSPOOF_DIR):
        print(f"  zaten mevcut: {ASVSPOOF_DIR} (atlanıyor)")
        return
    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(DATASET_DIR, "LA.zip")

    if not os.path.exists(zip_path):
        _download(ASVSPOOF_URL, zip_path)
    else:
        print(f"  arşiv zaten indirilmiş: {zip_path}")

    _unzip(zip_path, DATASET_DIR)

    # Arşiv kökü "LA/" klasörüdür; beklenen isme taşı.
    extracted = os.path.join(DATASET_DIR, "LA")
    if os.path.isdir(extracted) and not os.path.isdir(ASVSPOOF_DIR):
        shutil.move(extracted, ASVSPOOF_DIR)

    if os.path.isdir(ASVSPOOF_DIR):
        print(f"  ✔ hazır: {ASVSPOOF_DIR}")
        try:
            os.remove(zip_path)
        except OSError:
            pass
    else:
        print("  ⚠ Beklenen klasör oluşmadı; arşiv yapısını kontrol edin.")


def download_for():
    print("\n=== Fake-or-Real (for-original) ===")
    if os.path.isdir(FOR_DIR):
        print(f"  zaten mevcut: {FOR_DIR} (atlanıyor)")
        return
    if not shutil.which("kaggle"):
        print("  ⚠ 'kaggle' komut satırı aracı bulunamadı.")
        print("    Kurulum:  pip install kaggle")
        print("    Anahtar:  Kaggle hesabı → Settings → API → 'Create New Token'")
        print("              indirilen kaggle.json dosyasını ~/.kaggle/ altına koyun.")
        print(f"    Sonra:    kaggle datasets download -d {FOR_KAGGLE_SLUG} -p Dataset/ --unzip")
        print("    Alternatif kaynak: https://bil.eecs.yorku.ca/datasets/")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"  indiriliyor (kaggle): {FOR_KAGGLE_SLUG}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", FOR_KAGGLE_SLUG, "-p", DATASET_DIR, "--unzip"],
        check=True,
    )

    # Açılan içerikte 'for-original' klasörünü bul ve beklenen konuma taşı.
    if not os.path.isdir(FOR_DIR):
        for root, dirs, _ in os.walk(DATASET_DIR):
            if "for-original" in dirs:
                src = os.path.join(root, "for-original")
                if os.path.abspath(src) != os.path.abspath(FOR_DIR):
                    shutil.move(src, FOR_DIR)
                break

    if os.path.isdir(FOR_DIR):
        print(f"  ✔ hazır: {FOR_DIR}")
    else:
        print("  ⚠ 'for-original' klasörü bulunamadı; arşiv yapısını kontrol edin.")


def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if target not in ("asvspoof", "for", "all"):
        print("Kullanım: python download_datasets.py [asvspoof|for|all]")
        sys.exit(1)

    if target in ("asvspoof", "all"):
        download_asvspoof()
    if target in ("for", "all"):
        download_for()

    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
