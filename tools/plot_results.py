"""
=============================================================================
  ASVspoof 2019 LA — Sonuç Grafikleri
=============================================================================
  Amaç:
    results/asvspoof_results.csv dosyasındaki dört modelin (RF, SVM, CNN-Mel,
    CNN-Spec) saldırı-gruplu 5 katlı çapraz doğrulama (görülmemiş saldırı)
    sonuçlarını birden çok açıdan görselleştirir.

  Üretilen Çıktılar (figures/):
    1. asvspoof_eer.png              — EER (5 kat ort. ± std; düşük = daha iyi)
    2. asvspoof_metrics.png          — Dengeli doğruluk / F1 / AUC (yüksek = daha iyi)
    3. asvspoof_all_metrics.png      — Tüm metriklerin model × metrik ısı haritası
    4. asvspoof_radar.png            — Modellerin metrik radar (örümcek) karşılaştırması
    5. asvspoof_precision_recall.png — Precision–Recall dengesi (model başına)

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
MODEL_COLORS = {"RF": "#e15759", "SVM": "#f28e2b", "Mel-CNN": "#4e79a7", "Spec-CNN": "#59a14f"}


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


def _save(fig, name):
    p = os.path.join(FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {name}")


# ─────────────────────────────────────────────
# 1) EER (± std)
# ─────────────────────────────────────────────
def plot_eer(data):
    vals = [data[m]["EER"] for m in METHODS]
    errs = [data[m]["EER_std"] or 0.0 for m in METHODS]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METHODS))
    bars = ax.bar(x, vals, yerr=errs, capsize=4,
                  color=[MODEL_COLORS[m] for m in METHODS],
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("EER", fontsize=11); ax.set_ylim(bottom=0)
    ax.set_title("ASVspoof 2019 LA — Modellere göre EER", fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    ax.text(0.5, 0.97, "↓ düşük = daha iyi", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, style="italic", color="#555")
    _save(fig, "asvspoof_eer.png")


# ─────────────────────────────────────────────
# 2) BalACC / F1 / AUC
# ─────────────────────────────────────────────
def plot_metrics(data):
    metrics = [("BalACC", "Dengeli doğruluk"), ("F1", "F1"), ("AUC", "AUC")]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    x = np.arange(len(METHODS)); width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label) in enumerate(metrics):
        vals = [data[m][key] for m in METHODS]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("Skor", fontsize=11); ax.set_ylim(0, 1.05)
    ax.set_title("ASVspoof 2019 LA — Dengeli doğruluk / F1 / AUC", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    ax.text(0.01, 0.97, "↑ yüksek = daha iyi", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, style="italic", color="#555")
    _save(fig, "asvspoof_metrics.png")


# ─────────────────────────────────────────────
# 3) Tüm metrikler — ısı haritası
# ─────────────────────────────────────────────
def plot_all_metrics(data):
    rows = [("ACC", "Accuracy"), ("BalACC", "Dengeli doğr."), ("Precision", "Precision"),
            ("Recall", "Recall"), ("F1", "F1"), ("AUC", "AUC")]
    M = np.array([[data[m][k] for m in METHODS] for k, _ in rows])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lbl for _, lbl in rows], fontsize=11)
    for i in range(len(rows)):
        for j in range(len(METHODS)):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.6 else "black", fontweight="bold")
    ax.set_title("ASVspoof 2019 LA — Tüm metrikler (yüksek = daha iyi)",
                 fontsize=12.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Skor", fontsize=10)
    _save(fig, "asvspoof_all_metrics.png")


# ─────────────────────────────────────────────
# 4) Radar (örümcek) karşılaştırma
# ─────────────────────────────────────────────
def plot_radar(data):
    axes_keys = [("BalACC", "Dengeli\ndoğruluk"), ("Precision", "Precision"),
                 ("Recall", "Recall"), ("F1", "F1"), ("AUC", "AUC")]
    labels = [lbl for _, lbl in axes_keys]
    N = len(axes_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for m in METHODS:
        vals = [data[m][k] for k, _ in axes_keys]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, label=m, color=MODEL_COLORS[m])
        ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[m])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#777")
    ax.set_title("ASVspoof 2019 LA — Model karşılaştırması (radar)\n(dışa doğru = daha iyi)",
                 fontsize=12.5, fontweight="bold", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), fontsize=9)
    _save(fig, "asvspoof_radar.png")


# ─────────────────────────────────────────────
# 5) Precision–Recall dengesi
# ─────────────────────────────────────────────
def plot_precision_recall(data):
    x = np.arange(len(METHODS)); width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    prec = [data[m]["Precision"] for m in METHODS]
    rec = [data[m]["Recall"] for m in METHODS]
    b1 = ax.bar(x - width / 2, prec, width, label="Precision", color="#76b7b2",
                edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + width / 2, rec, width, label="Recall", color="#edc948",
                edgecolor="black", linewidth=0.4)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("Skor", fontsize=11); ax.set_ylim(0, 1.05)
    ax.set_title("ASVspoof 2019 LA — Precision–Recall dengesi (bonafide sınıfı)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    _save(fig, "asvspoof_precision_recall.png")


def main():
    data = load_results()
    print("Sonuç grafikleri üretiliyor:")
    plot_eer(data)
    plot_metrics(data)
    plot_all_metrics(data)
    plot_radar(data)
    plot_precision_recall(data)
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
