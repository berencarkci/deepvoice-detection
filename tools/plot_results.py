"""
=============================================================================
  ASVspoof 2019 LA — Sonuç Grafikleri
=============================================================================
  Amaç:
    results/asvspoof_results.csv dosyasındaki dört modelin (RF, SVM, CNN-Mel,
    CNN-Spec) saldırı-gruplu 5 katlı çapraz doğrulama (görülmemiş saldırı)
    sonuçlarını görselleştirir.

  Üretilen Çıktılar:
    1. figures/asvspoof_eer.png      — EER (5 kat ortalaması ± std; düşük = daha iyi)
    2. figures/asvspoof_metrics.png  — Dengeli doğruluk / F1 / AUC (yüksek = daha iyi)

  Not: Sınıf dengesizliği (~1:9) nedeniyle düz doğruluk yanıltıcıdır; bu yüzden
  birincil metrikler EER, dengeli doğruluk ve AUC'dur.
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
PROTOCOL = "attack_kfold"
BAR_COLOR = "#4c78a8"


def load_results():
    """CSV'yi method -> {metrik: float} sözlüğüne okur."""
    data = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["protocol"] != PROTOCOL:
                continue
            rec = {}
            for col in ("EER", "EER_std", "ACC", "BalACC", "Precision",
                        "Recall", "F1", "AUC"):
                v = r.get(col, "")
                rec[col] = float(v) if v not in (None, "") else None
            data[r["method"]] = rec
    return data


def plot_eer(data):
    vals = [data[m]["EER"] for m in METHODS]
    errs = [data[m]["EER_std"] or 0.0 for m in METHODS]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METHODS))
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=BAR_COLOR,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("EER", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_title("ASVspoof 2019 LA — Modellere göre EER",
                 fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.97, "↓ düşük = daha iyi", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, style="italic", color="#555")

    p = os.path.join(FIGURES_DIR, "asvspoof_eer.png")
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ asvspoof_eer.png")


def plot_metrics(data):
    metrics = [("BalACC", "Dengeli doğruluk"), ("F1", "F1"), ("AUC", "AUC")]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    x = np.arange(len(METHODS))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label) in enumerate(metrics):
        vals = [data[m][key] for m in METHODS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[i],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("ASVspoof 2019 LA — Dengeli doğruluk / F1 / AUC",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.text(0.01, 0.97, "↑ yüksek = daha iyi", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, style="italic", color="#555")

    p = os.path.join(FIGURES_DIR, "asvspoof_metrics.png")
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ asvspoof_metrics.png")


def main():
    data = load_results()
    print("Sonuç grafikleri üretiliyor:")
    plot_eer(data)
    plot_metrics(data)
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
