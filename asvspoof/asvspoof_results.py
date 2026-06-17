"""
ASVspoof sonuçları için ortak kayıt modülü.

Her eğitim scripti (asvspoof_train_ml / _cnn / _spectrogram_cnn) sonuçlarını buraya
record_result(...) ile yazar:
  * UPSERT: aynı (method, protocol) tekrar yazılırsa satır güncellenir (kopya olmaz).
  * asvspoof_results.csv kalıcı tutulur; asvspoof_results.md tablosu CSV'den üretilir.

protocol: "kfold_pooled" (havuzlanmış 5-fold) | "official_eval" (standart protokol, eval)
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
_PROTO_ORDER = ["kfold_pooled", "attack_kfold", "official_dev", "official_eval"]
_PROTO_LABEL = {"kfold_pooled": "Havuzlanmış 5-fold çapraz doğrulama (konuşmacıya göre)",
                "attack_kfold": "Saldırı-gruplu 5-fold çapraz doğrulama (görülmemiş saldırı)",
                "official_dev": "Standart protokol (dev / doğrulama)",
                "official_eval": "Standart protokol (eval)"}


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
    m, p = key
    mi = _METHOD_ORDER.index(m) if m in _METHOD_ORDER else len(_METHOD_ORDER)
    pi = _PROTO_ORDER.index(p) if p in _PROTO_ORDER else len(_PROTO_ORDER)
    return (mi, m, pi, p)


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
    methods = []
    for key in sorted(rows, key=_sort_key):
        if key[0] not in methods:
            methods.append(key[0])

    def get(m, p, col):
        r = rows.get((m, p))
        return r[col] if r and r.get(col) else ""

    def cell(m, p, col):
        return get(m, p, col) or "—"

    def eer_pm(m, p):
        e = get(m, p, "EER"); s = get(m, p, "EER_std")
        if not e:
            return "—"
        return f"{e} ± {s}" if s else e

    def inflation(m):
        try:
            k = float(get(m, "kfold_pooled", "EER"))
            o = float(get(m, "official_eval", "EER"))
            return f"{o / k:.1f}×" if k > 0 else "—"
        except (ValueError, ZeroDivisionError):
            return "—"

    L = []
    L.append("# ASVspoof 2019 LA — Sonuç Özeti")
    L.append("")
    L.append("> `asvspoof_results.py` tarafından `asvspoof_results.csv`'den üretilir.")
    L.append("")
    L.append("Metrik = **EER** (düşük = daha iyi). Sınıf dengesizliği (~1:9) nedeniyle EER, "
             "dengeli doğruluk (BalACC) ve AUC esas alınır.")
    L.append("")
    L.append("- **Havuzlanmış k-fold:** train+dev+eval birleştirilir, konuşmacıya göre GroupKFold(5).")
    L.append("- **Standart protokol:** train ile eğit, dev ile doğrula, eval ile test.")
    L.append("")
    L.append("## Ana tablo — EER")
    L.append("")
    L.append("| Yöntem | Havuzlanmış k-fold | Standart protokol (eval) | Oran |")
    L.append("|--------|--------------------|--------------------------|------|")
    for m in methods:
        L.append(f"| {m} | {eer_pm(m, 'kfold_pooled')} | {cell(m, 'official_eval', 'EER')} "
                 f"| {inflation(m)} |")
    L.append("")
    for proto in _PROTO_ORDER:
        proto_methods = [m for m in methods if (m, proto) in rows]
        if not proto_methods:
            continue
        L.append(f"### {_PROTO_LABEL[proto]}")
        L.append("")
        L.append("| Yöntem | EER | Accuracy | BalACC | Precision | Recall | F1 | AUC |")
        L.append("|--------|-----|----------|--------|-----------|--------|----|-----|")
        for m in proto_methods:
            L.append(f"| {m} | {cell(m, proto, 'EER')} | {cell(m, proto, 'ACC')} "
                     f"| {cell(m, proto, 'BalACC')} | {cell(m, proto, 'Precision')} "
                     f"| {cell(m, proto, 'Recall')} | {cell(m, proto, 'F1')} "
                     f"| {cell(m, proto, 'AUC')} |")
        L.append("")
    L.append('"Oran" sütunu = standart protokol EER / havuzlanmış k-fold EER.')
    L.append("")
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
