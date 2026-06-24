"""
ASVspoof sonuçları için ortak kayıt modülü.

Eğitim/değerlendirme scriptleri sonuçlarını record_result(...) ile buraya yazar:
  * UPSERT: aynı (method, protocol) tekrar yazılırsa satır güncellenir (kopya olmaz).
  * asvspoof_results.csv kalıcı tutulur; asvspoof_results.md tablosu CSV'den üretilir.

Değerlendirme protokolü: "attack_kfold" — saldırı-gruplu 5 katlı çapraz doğrulama
(her katta görülmemiş saldırı türleriyle test edilir).
"""

import os
import csv

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(_RESULTS_DIR, "asvspoof_results.csv")
MD_PATH = os.path.join(_RESULTS_DIR, "asvspoof_results.md")

COLUMNS = ["method", "protocol", "EER", "EER_std", "ACC", "BalACC",
           "Precision", "Recall", "F1", "AUC"]
_METHOD_ORDER = ["RF", "SVM", "Mel-CNN", "Spec-CNN"]
_PROTO_LABEL = {"attack_kfold": "Saldırı-gruplu 5 katlı çapraz doğrulama (görülmemiş saldırı)"}


def _fmt(v):
    return "" if v is None or v == "" else f"{float(v):.4f}"


def _load():
    rows = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[(r["method"], r["protocol"])] = r
    return rows


def _sort_key(key):
    m, _ = key
    mi = _METHOD_ORDER.index(m) if m in _METHOD_ORDER else len(_METHOD_ORDER)
    return (mi, m)


def record_result(method, protocol, eer, balacc, recall, auc, eer_std=None,
                  acc=None, precision=None, f1=None):
    """Bir (method, protocol) sonucunu CSV'ye upsert eder ve md'yi yeniden üretir."""
    rows = _load()
    rows[(method, protocol)] = {
        "method": method, "protocol": protocol,
        "EER": _fmt(eer), "EER_std": _fmt(eer_std),
        "ACC": _fmt(acc), "BalACC": _fmt(balacc),
        "Precision": _fmt(precision), "Recall": _fmt(recall),
        "F1": _fmt(f1), "AUC": _fmt(auc),
    }
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for key in sorted(rows, key=_sort_key):
            w.writerow(rows[key])
    _write_md(rows)
    print(f"  ✔ Sonuç kaydedildi: {method} / {protocol} → asvspoof_results.csv (+md)")


def _write_md(rows):
    def get(m, col):
        r = rows.get((m, "attack_kfold"))
        return r[col] if r and r.get(col) else ""

    def cell(m, col):
        return get(m, col) or "—"

    def eer_pm(m):
        e = get(m, "EER"); s = get(m, "EER_std")
        if not e:
            return "—"
        return f"{e} ± {s}" if s else e

    methods = [m for m in _METHOD_ORDER if (m, "attack_kfold") in rows]

    L = []
    L.append("# ASVspoof 2019 LA — Sonuç Özeti")
    L.append("")
    L.append("> `asvspoof_results.py` tarafından `asvspoof_results.csv`'den üretilir.")
    L.append("")
    L.append("Değerlendirme: **saldırı-gruplu 5 katlı çapraz doğrulama** — her katta "
             "modeller görülmemiş saldırı türleriyle test edilir. Ana metrik **EER** "
             "(düşük = daha iyi); sınıf dengesizliği (~1:9) nedeniyle dengeli doğruluk "
             "(BalACC) ve AUC de raporlanır.")
    L.append("")
    L.append("| Yöntem | EER | Accuracy | BalACC | Precision | Recall | F1 | AUC |")
    L.append("|--------|-----|----------|--------|-----------|--------|----|-----|")
    for m in methods:
        L.append(f"| {m} | {eer_pm(m)} | {cell(m, 'ACC')} | {cell(m, 'BalACC')} "
                 f"| {cell(m, 'Precision')} | {cell(m, 'Recall')} | {cell(m, 'F1')} "
                 f"| {cell(m, 'AUC')} |")
    L.append("")
    L.append("EER değerleri 5 katın ortalaması ± standart sapma olarak verilir.")
    L.append("")
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
