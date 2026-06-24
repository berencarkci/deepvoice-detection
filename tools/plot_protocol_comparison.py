"""
=============================================================================
  ASVspoof 2019 LA — Protokol Karşılaştırma Grafikleri
=============================================================================
  Amaç:
    results/asvspoof_results.csv dosyasındaki dört modelin (RF, SVM, CNN-Mel,
    CNN-Spec) üç değerlendirme protokolündeki sonuçlarını yan yana görselleştirir:

      1. Havuzlanmış 5-fold      (konuşmacıya göre GroupKFold; iyimser)
      2. Saldırı-gruplu 5-fold   (görülmemiş saldırı; dürüst)
      3. Standart protokol (eval)(train→dev→eval; görülmemiş konuşmacı + saldırı)

  Üretilen Çıktılar:
    1. figures/asvspoof_protocol_eer.png        — EER (düşük = daha iyi)
    2. figures/asvspoof_protocol_balacc.png     — Dengeli doğruluk (yüksek = daha iyi)
    3. figures/asvspoof_protocol_inflation.png  — Havuzlanmış k-fold / standart EER oranı

  Not: Sınıf dengesizliği (~1:9) nedeniyle düz doğruluk yanıltıcıdır; bu yüzden
  birincil metrikler EER ve dengeli doğruluktur.
=============================================================================
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_BASE_DIR, "results", "asvspoof_results.csv")
FIGURES_DIR = os.path.join(_BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

METHODS = ["RF", "SVM", "Mel-CNN", "Spec-CNN"]
PROTOCOLS = ["kfold_pooled", "attack_kfold", "official_eval"]
PROTO_LABEL = {
    "kfold_pooled": "Havuzlanmış 5-fold\n(konuşmacıya göre)",
    "attack_kfold": "Saldırı-gruplu 5-fold\n(görülmemiş saldırı)",
    "official_eval": "Standart protokol\n(eval)",
}
PROTO_COLOR = {
    "kfold_pooled": "#d65f5f",   # iyimser — kırmızı
    "attack_kfold": "#4c78a8",   # dürüst — mavi
    "official_eval": "#59a14f",  # altın standart — yeşil
}


def load_results():
    """CSV'yi (method, protocol) -> {metrik: float} sözlüğüne okur."""
    data = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rec = {}
            for col in ("EER", "EER_std", "ACC", "BalACC", "AUC"):
                v = r.get(col, "")
                rec[col] = float(v) if v not in (None, "") else None
            data[(r["method"], r["protocol"])] = rec
    return data


def _grouped_bars(data, metric, title, ylabel, out_name, higher_better):
    x = np.arange(len(METHODS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, proto in enumerate(PROTOCOLS):
        vals, errs = [], []
        for m in METHODS:
            rec = data.get((m, proto))
            vals.append(rec[metric] if rec and rec[metric] is not None else np.nan)
            errs.append(rec["EER_std"] if (metric == "EER" and rec and rec["EER_std"]) else 0.0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                      label=PROTO_LABEL[proto], color=PROTO_COLOR[proto],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    arrow = "↑ yüksek = daha iyi" if higher_better else "↓ düşük = daha iyi"
    ax.text(0.5, 0.97, arrow, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, style="italic", color="#555")

    p = os.path.join(FIGURES_DIR, out_name)
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {out_name}")


def _inflation_plot(data, out_name):
    """Havuzlanmış k-fold EER'in standart protokol EER'ine göre kaç kat
    iyimser olduğunu (şişme faktörü) gösterir."""
    factors = []
    for m in METHODS:
        k = data.get((m, "kfold_pooled"))
        o = data.get((m, "official_eval"))
        if k and o and k["EER"]:
            factors.append(o["EER"] / k["EER"])
        else:
            factors.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METHODS))
    colors = plt.cm.OrRd(np.linspace(0.4, 0.85, len(METHODS)))
    bars = ax.bar(x, factors, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="#333", linestyle="--", linewidth=1)
    ax.text(len(METHODS) - 0.5, 1.05, "şişme yok (1×)", ha="right",
            va="bottom", fontsize=9, color="#333")
    for b, v in zip(bars, factors):
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}×", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("EER şişme faktörü\n(standart / havuzlanmış k-fold)", fontsize=11)
    ax.set_title("Havuzlanmış k-fold EER'i ne kadar iyimser gösteriyor?",
                 fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    p = os.path.join(FIGURES_DIR, out_name)
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {out_name}")


def main():
    data = load_results()
    print("Protokol karşılaştırma grafikleri üretiliyor:")
    _grouped_bars(data, "EER",
                  "ASVspoof 2019 LA — Protokole göre EER",
                  "EER", "asvspoof_protocol_eer.png", higher_better=False)
    _grouped_bars(data, "BalACC",
                  "ASVspoof 2019 LA — Protokole göre dengeli doğruluk",
                  "Dengeli doğruluk (BalACC)", "asvspoof_protocol_balacc.png",
                  higher_better=True)
    _inflation_plot(data, "asvspoof_protocol_inflation.png")
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
