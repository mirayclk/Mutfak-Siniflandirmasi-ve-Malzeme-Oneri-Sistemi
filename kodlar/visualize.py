"""
visualize.py — Sunum için model görselleştirmeleri.

Çalıştır:
    python visualize.py

Üretilen görseller (./plots/ klasörüne kaydedilir):
    01_cuisine_distribution.png   → Mutfak bazlı tarif sayısı
    02_per_class_accuracy.png     → Mutfak bazlı doğruluk oranları
    03_confusion_matrix.png       → Karışıklık matrisi (heatmap)
    04_top_ingredients.png        → Her mutfağın en ayırt edici malzemeleri
    05_idf_distribution.png       → IDF dağılımı & en nadir/yaygın malzemeler
    06_ingredient_overlap.png     → Mutfaklar arası malzeme örtüşme haritası
    07_pipeline_diagram.png       → Proje pipeline şeması
    08_recommendation_example.png → Örnek malzeme önerisi görselleştirmesi
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from data import load_data, build_vocabulary, build_sparse_matrix, vectorize, clean_recipe
from model import CuisineClassifier
from recommender import recommend_ingredients
from config import DEFAULT_TOP_K_INGREDIENT

# ── Stil Ayarları ──────────────────────────────────────────────────────────────
PALETTE = [
    "#2D6A9F", "#E84855", "#F9A825", "#2ECC71", "#9B59B6",
    "#1ABC9C", "#E67E22", "#3498DB", "#C0392B", "#27AE60",
    "#8E44AD", "#F39C12", "#16A085", "#D35400", "#2980B9",
    "#7D3C98", "#148F77", "#B7950B", "#922B21", "#1F618D",
]
BACKGROUND = "#FAFAFA"
ACCENT     = "#2D6A9F"
TEXT_COLOR = "#2C3E50"

plt.rcParams.update({
    "figure.facecolor":  BACKGROUND,
    "axes.facecolor":    BACKGROUND,
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BACKGROUND,
})

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CUISINE_LABELS = {
    "brazilian": "Brezilya", "british": "İngiliz", "cajun_creole": "Cajun/Creole",
    "chinese": "Çin", "filipino": "Filipin", "french": "Fransız",
    "greek": "Yunan", "indian": "Hint", "irish": "İrlanda",
    "italian": "İtalyan", "jamaican": "Jamaika", "japanese": "Japon",
    "korean": "Kore", "mexican": "Meksika", "moroccan": "Fas",
    "russian": "Rus", "southern_us": "Güney ABD", "spanish": "İspanyol",
    "thai": "Tay", "vietnamese": "Vietnam",
}

def _save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1 — MUTFAKLARDAKİ TARİF SAYISI
# ══════════════════════════════════════════════════════════════════════════════

def plot_cuisine_distribution(train_data):
    counts = Counter(ex["cuisine"] for ex in train_data)
    cuisines_sorted = sorted(counts, key=lambda c: counts[c], reverse=True)
    values = [counts[c] for c in cuisines_sorted]
    labels = [CUISINE_LABELS.get(c, c) for c in cuisines_sorted]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(cuisines_sorted))]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(labels, values, color=colors, width=0.65, zorder=3)
    ax.set_title("Mutfak Bazlı Tarif Sayısı (Train Seti)", pad=16)
    ax.set_ylabel("Tarif Sayısı")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 60,
            f"{val:,}",
            ha="center", va="bottom", fontsize=9, color=TEXT_COLOR,
        )

    total = sum(values)
    ax.text(0.99, 0.97, f"Toplam: {total:,} tarif",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#888888")
    fig.tight_layout()
    _save(fig, "01_cuisine_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — MUTFAK BAZLI DOĞRULUK ORANLARI
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_class_accuracy(model, X_train, y_train, cuisines):
    tr_idx, val_idx = train_test_split(
        np.arange(X_train.shape[0]), test_size=0.15,
        random_state=42, stratify=y_train
    )
    X_val = np.asarray(X_train[val_idx].todense())
    y_val = y_train[val_idx]
    preds = model.predict(X_val)

    accs, labels, colors = [], [], []
    for i, c in enumerate(cuisines):
        mask = y_val == i
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == y_val[mask]).mean()
        accs.append(acc)
        labels.append(CUISINE_LABELS.get(c, c))
        colors.append(PALETTE[i % len(PALETTE)])

    order = np.argsort(accs)[::-1]
    accs   = [accs[i] for i in order]
    labels = [labels[i] for i in order]
    colors = [colors[i] for i in order]

    overall = (preds == y_val).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(labels, [a * 100 for a in accs], color=colors, width=0.65, zorder=3)
    ax.axhline(overall * 100, color="#E84855", linewidth=1.8,
               linestyle="--", zorder=4, label=f"Genel Doğruluk: %{overall*100:.1f}")
    ax.set_title("Mutfak Bazlı Validasyon Doğruluğu", pad=16)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=40)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=10)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"%{acc*100:.0f}",
            ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR,
        )

    fig.tight_layout()
    _save(fig, "02_per_class_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3 — KARIŞIKLIK MATRİSİ
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(model, X_train, y_train, cuisines):
    tr_idx, val_idx = train_test_split(
        np.arange(X_train.shape[0]), test_size=0.15,
        random_state=42, stratify=y_train
    )
    X_val = np.asarray(X_train[val_idx].todense())
    y_val = y_train[val_idx]
    preds = model.predict(X_val)

    cm = confusion_matrix(y_val, preds)
    # Satır normalize et (recall bazlı)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    short = [CUISINE_LABELS.get(c, c)[:8] for c in cuisines]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Recall (normalize)")

    ax.set_xticks(range(len(cuisines)))
    ax.set_yticks(range(len(cuisines)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Tahmin Edilen Mutfak")
    ax.set_ylabel("Gerçek Mutfak")
    ax.set_title("Karışıklık Matrisi (Normalize)", pad=14)

    for i in range(len(cuisines)):
        for j in range(len(cuisines)):
            val = cm_norm[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if val > 0.5 else TEXT_COLOR)

    fig.tight_layout()
    _save(fig, "03_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4 — MUTFAK BAZLI EN AYIRTEDİCİ MALZEMELER
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_ingredients(model, vocab, cuisines, selected=None):
    if selected is None:
        selected = ["italian", "mexican", "indian", "japanese",
                    "french", "thai", "greek", "chinese"]

    idx_to_ing = {i: ing for ing, i in vocab.items()}
    c2i = {c: i for i, c in enumerate(cuisines)}
    idf = model.idf_

    n = len(selected)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.8))
    axes = axes.flatten()
    fig.suptitle("Mutfak Başına En Ayırt Edici 7 Malzeme\n(IDF Ağırlıklı Frekans)", 
                 fontsize=15, fontweight="bold", y=1.01)

    for ax_idx, cuisine in enumerate(selected):
        ax = axes[ax_idx]
        c_idx = c2i[cuisine]
        counts = model.cuisine_counts_[c_idx]
        score = counts * idf
        top_k = np.argsort(score)[::-1][:7]

        ings   = [idx_to_ing[i].replace(" ", "\n") for i in top_k]
        scores = [score[i] for i in top_k]
        color  = PALETTE[c_idx % len(PALETTE)]

        ax.barh(ings[::-1], scores[::-1], color=color, alpha=0.85, zorder=3)
        ax.set_title(CUISINE_LABELS.get(cuisine, cuisine), color=color, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=7)

    for ax in axes[len(selected):]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, "04_top_ingredients.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5 — IDF DAĞILIMI
# ══════════════════════════════════════════════════════════════════════════════

def plot_idf_distribution(model, vocab):
    idf = model.idf_
    idx_to_ing = {i: ing for ing, i in vocab.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("IDF Dağılımı: Malzemelerin Ayırt Edicilik Analizi",
                 fontsize=14, fontweight="bold")

    # Sol: Histogram
    ax1.hist(idf, bins=50, color=ACCENT, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax1.axvline(idf.mean(), color="#E84855", linewidth=1.8,
                linestyle="--", label=f"Ortalama: {idf.mean():.2f}")
    ax1.set_xlabel("IDF Değeri")
    ax1.set_ylabel("Malzeme Sayısı")
    ax1.set_title("IDF Değer Dağılımı")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend()

    # Sağ: En yaygın (düşük IDF) ve en nadir (yüksek IDF) malzemeler
    order = np.argsort(idf)
    n_show = 10
    common_idx = order[:n_show]
    rare_idx   = order[-n_show:][::-1]

    common_ings  = [idx_to_ing[i] for i in common_idx]
    common_idfs  = [idf[i] for i in common_idx]
    rare_ings    = [idx_to_ing[i] for i in rare_idx]
    rare_idfs    = [idf[i] for i in rare_idx]

    y_pos = np.arange(n_show)
    ax2.barh(y_pos - 0.2, common_idfs, height=0.35, color="#E84855",
             alpha=0.85, label="En Yaygın (Düşük IDF)", zorder=3)
    ax2.barh(y_pos + 0.2, rare_idfs,   height=0.35, color="#2ECC71",
             alpha=0.85, label="En Nadir (Yüksek IDF)", zorder=3)

    ax2.set_yticks(y_pos)
    combined_labels = [f"{c}  /  {r}" for c, r in zip(common_ings, rare_ings)]
    ax2.set_yticklabels(combined_labels, fontsize=8)
    ax2.set_xlabel("IDF Değeri")
    ax2.set_title("Yaygın vs. Nadir Malzeme Karşılaştırması")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "05_idf_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6 — MUTFAKLAR ARASI MALZEME ÖRTÜŞME HARİTASI
# ══════════════════════════════════════════════════════════════════════════════

def plot_ingredient_overlap(model, cuisines):
    n = len(cuisines)
    counts = model.cuisine_counts_          # (n_classes, n_vocab)
    binary = (counts > 0).astype(float)     # binary varlık matrisi

    # Jaccard benzerliği
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = (binary[i] * binary[j]).sum()
            union = np.maximum(binary[i], binary[j]).sum()
            overlap[i, j] = inter / (union + 1e-9)

    labels = [CUISINE_LABELS.get(c, c) for c in cuisines]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(overlap, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Jaccard Benzerliği")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_title("Mutfaklar Arası Malzeme Örtüşmesi (Jaccard)", pad=14)

    for i in range(n):
        for j in range(n):
            val = overlap[i, j]
            if val > 0.15 and i != j:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.6 else TEXT_COLOR)

    fig.tight_layout()
    _save(fig, "06_ingredient_overlap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7 — PROJENİN PIPELINE ŞEMASI
# ══════════════════════════════════════════════════════════════════════════════

def plot_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle("Proje Pipeline'ı", fontsize=15, fontweight="bold", y=0.98)

    steps = [
        ("Kullanıcı\nMalzemeleri",   "#3498DB", 1.0),
        ("Temizleme &\nStemming",    "#2ECC71", 3.5),
        ("Binary\nVektör",           "#9B59B6", 6.0),
        ("IDF\nAğırlıklandırma",     "#E67E22", 8.5),
        ("Naive Bayes\nSınıflandırıcı", "#E84855", 11.0),
        ("Mutfak\nTahmini",          "#1ABC9C", 13.5),
    ]
    box_w, box_h = 2.0, 1.4
    cy = 2.5

    for label, color, cx in steps:
        fancy = mpatches.FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.92,
            zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=4)

    for i in range(len(steps) - 1):
        x1 = steps[i][2] + box_w / 2 + 0.05
        x2 = steps[i + 1][2] - box_w / 2 - 0.05
        ax.annotate("",
            xy=(x2, cy), xytext=(x1, cy),
            arrowprops=dict(arrowstyle="-|>", color="#555555",
                            lw=1.8, mutation_scale=18),
            zorder=2,
        )

    # Alt notlar
    notes = [
        (1.0,  "ham metin"),
        (3.5,  "stopword kaldır\nPorter stem"),
        (6.0,  "vocab boyutu\n~5943"),
        (8.5,  "nadir=yüksek ağırlık\nyaygın=düşük"),
        (11.0, "log P(cuisine|ing)\n%72 val. acc."),
        (13.5, "top-K mutfak\n+ öneri"),
    ]
    for cx, note in notes:
        ax.text(cx, cy - box_h / 2 - 0.35, note,
                ha="center", va="top", fontsize=7.5, color="#666666",
                style="italic")

    # İkinci satır: öneri kolu
    rec_steps = [
        ("Cosine\nBenzerliği",       "#F39C12", 8.5),
        ("Cuisine Frekans\n× IDF",   "#8E44AD", 11.0),
        ("Malzeme\nÖnerisi",         "#27AE60", 13.5),
    ]
    cy2 = 0.9
    for label, color, cx in rec_steps:
        fancy = mpatches.FancyBboxPatch(
            (cx - box_w / 2, cy2 - box_h / 2 + 0.1), box_w, box_h - 0.2,
            boxstyle="round,pad=0.10",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85,
            zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(cx, cy2 + 0.1, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=4)

    for i in range(len(rec_steps) - 1):
        x1 = rec_steps[i][2] + box_w / 2 + 0.05
        x2 = rec_steps[i + 1][2] - box_w / 2 - 0.05
        ax.annotate("",
            xy=(x2, cy2 + 0.1), xytext=(x1, cy2 + 0.1),
            arrowprops=dict(arrowstyle="-|>", color="#555555",
                            lw=1.5, mutation_scale=16),
            zorder=2,
        )

    # Dikey bağlantı: Sınıflandırıcı → Öneri kolu
    ax.annotate("",
        xy=(11.0, cy2 + box_h / 2 - 0.1), xytext=(11.0, cy - box_h / 2),
        arrowprops=dict(arrowstyle="-|>", color="#888888",
                        lw=1.2, linestyle="dashed", mutation_scale=14),
        zorder=2,
    )

    ax.text(0.5, 4.7, "Mutfak Tahmini Kolu", fontsize=9,
            color="#555555", fontweight="bold")
    ax.text(7.8, 1.6, "Malzeme Öneri Kolu", fontsize=9,
            color="#555555", fontweight="bold")

    _save(fig, "07_pipeline_diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8 — ÖRNEK MALZEMELERİN TAHMİN + ÖNERİ GÖRSELİ
# ══════════════════════════════════════════════════════════════════════════════

def plot_recommendation_example(model, vocab, idx_to_cuisine, X_train, y_train):
    examples = [
        {
            "label": "Örnek 1 — Japon/Asya",
            "ingredients": ["mirin", "dashi", "sake", "soy sauce"],
            "color": "#3498DB",
        },
        {
            "label": "Örnek 2 — Meksika",
            "ingredients": ["tortilla", "cumin", "chili powder", "lime juice"],
            "color": "#E84855",
        },
        {
            "label": "Örnek 3 — İtalyan",
            "ingredients": ["pasta", "tomato", "olive oil", "basil"],
            "color": "#2ECC71",
        },
    ]

    cuisines_list = [idx_to_cuisine[i] for i in range(len(idx_to_cuisine))]

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Örnek Kullanıcı Girdileri: Mutfak Tahmini & Malzeme Önerisi",
                 fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(len(examples), 2, figure=fig,
                           width_ratios=[1, 1.4], hspace=0.55, wspace=0.35)

    for row, ex in enumerate(examples):
        user_vec = vectorize(ex["ingredients"], vocab)
        cuisine_results = model.predict_cuisine_from_vec(user_vec, idx_to_cuisine, top_k=5)
        rec_results = recommend_ingredients(
            ex["ingredients"], model, vocab, idx_to_cuisine,
            X_train, y_train, top_k=6
        )

        color = ex["color"]

        # ── Sol panel: Mutfak olasılıkları ──
        ax_l = fig.add_subplot(gs[row, 0])
        c_labels = [CUISINE_LABELS.get(r["cuisine"], r["cuisine"]) for r in cuisine_results]
        c_probs  = [r["probability"] * 100 for r in cuisine_results]
        bar_colors = [color if i == 0 else "#CCCCCC" for i in range(len(c_labels))]

        bars = ax_l.barh(c_labels[::-1], c_probs[::-1],
                         color=bar_colors[::-1], zorder=3, height=0.55)
        ax_l.set_xlim(0, max(c_probs) * 1.25)
        ax_l.set_xlabel("Olasılık (%)")
        ax_l.set_title(ex["label"], color=color, fontweight="bold", fontsize=11)
        ax_l.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_l.set_axisbelow(True)
        for bar, val in zip(bars, c_probs[::-1]):
            ax_l.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                      f"%{val:.1f}", va="center", fontsize=9)

        # Girdi malzemeleri
        ax_l.text(0.01, -0.22,
                  "Girdi: " + ", ".join(ex["ingredients"]),
                  transform=ax_l.transAxes, fontsize=7.5,
                  color="#888888", style="italic")

        # ── Sağ panel: Malzeme önerileri ──
        ax_r = fig.add_subplot(gs[row, 1])
        if rec_results:
            r_labels = [r["ingredient"] for r in rec_results]
            r_scores = [r["score"] for r in rec_results]
            max_s = max(r_scores) + 1e-9
            norm_scores = [s / max_s * 100 for s in r_scores]

            r_colors = [
                color if i == 0
                else f"#{hex(int(180 - i * 18))[2:].zfill(2)}"
                     f"{hex(int(180 - i * 18))[2:].zfill(2)}"
                     f"{hex(int(180 - i * 18))[2:].zfill(2)}"
                for i in range(len(r_labels))
            ]
            bars_r = ax_r.barh(
                r_labels[::-1], norm_scores[::-1],
                color=r_colors[::-1], zorder=3, height=0.55,
            )
            ax_r.set_xlim(0, 125)
            ax_r.set_xlabel("Normalize Skor (%)")
            ax_r.set_title(
                f"Önerilen Malzemeler  →  {CUISINE_LABELS.get(rec_results[0]['cuisine'], rec_results[0]['cuisine'])} Mutfağı",
                fontsize=10, fontweight="bold", color="#444444",
            )
            ax_r.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
            ax_r.set_axisbelow(True)
            for bar, val in zip(bars_r, norm_scores[::-1]):
                ax_r.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                          f"{val:.0f}", va="center", fontsize=9)

    fig.tight_layout()
    _save(fig, "08_recommendation_example.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  GÖRSELLEŞTİRME — Başlatılıyor")
    print("=" * 60)

    # Veri & model hazırlık
    print("\nVeri yükleniyor ve model eğitiliyor...")
    train_data, _ = load_data()
    vocab = build_vocabulary(train_data)

    cuisines = sorted({ex["cuisine"] for ex in train_data})
    c2i = {c: i for i, c in enumerate(cuisines)}
    i2c = {i: c for c, i in c2i.items()}

    X_train, y_train, _ = build_sparse_matrix(train_data, vocab, c2i, has_labels=True)
    model = CuisineClassifier()
    model.fit(X_train, y_train)
    print("  Model hazır.\n")

    print("Görseller oluşturuluyor  →  ./plots/")
    print("-" * 40)

    plot_cuisine_distribution(train_data)
    plot_per_class_accuracy(model, X_train, y_train, cuisines)
    plot_confusion_matrix(model, X_train, y_train, cuisines)
    plot_top_ingredients(model, vocab, cuisines)
    plot_idf_distribution(model, vocab)
    plot_ingredient_overlap(model, cuisines)
    plot_pipeline_diagram()
    plot_recommendation_example(model, vocab, i2c, X_train, y_train)

    print("-" * 40)
    print(f"\n✓ Tüm görseller  →  {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    main()
