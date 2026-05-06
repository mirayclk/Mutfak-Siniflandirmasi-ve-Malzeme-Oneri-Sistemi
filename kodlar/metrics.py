"""
metrics.py — Genişletilmiş değerlendirme metrikleri.

Mevcut projeye eklenecek üç yeni metrik:
  1. Precision / Recall / F1  (per-class ve macro/weighted ortalama)
  2. Confusion Matrix          (görsel + sayısal)
  3. Log-Loss (Cross-Entropy)  (kalibrasyon kalitesi)

Kullanım (main.py içinde train_model() sonrasına ekle):
    from metrics import evaluate_all

    X_val_dense = np.asarray(X_train[val_idx].todense())
    evaluate_all(
        model       = model,
        X           = X_val_dense,
        y_true      = y_train[val_idx],
        idx_to_cuisine = idx_to_cuisine,
    )
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1. PRECISION / RECALL / F1
# ══════════════════════════════════════════════════════════════════════════════

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    """
    Her sınıf için Precision, Recall ve F1 hesaplar.
    Ayrıca Macro ve Weighted ortalama döndürür.

    Formüller:
        Precision_c = TP_c / (TP_c + FP_c)
        Recall_c    = TP_c / (TP_c + FN_c)
        F1_c        = 2 * P_c * R_c / (P_c + R_c)

    Args:
        y_true    : Gerçek etiketler (n_samples,)
        y_pred    : Tahmin edilen etiketler (n_samples,)
        n_classes : Toplam sınıf sayısı

    Returns:
        per_class : dict — her sınıf için {"precision", "recall", "f1", "support"}
        macro     : dict — sınıf ağırlıksız ortalama
        weighted  : dict — destek (support) ağırlıklı ortalama
    """
    per_class = {}

    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[c] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }

    # Macro ortalama: her sınıfa eşit ağırlık
    macro = {
        "precision": round(np.mean([v["precision"] for v in per_class.values()]), 4),
        "recall":    round(np.mean([v["recall"]    for v in per_class.values()]), 4),
        "f1":        round(np.mean([v["f1"]        for v in per_class.values()]), 4),
    }

    # Weighted ortalama: her sınıfa support oranında ağırlık
    total = sum(v["support"] for v in per_class.values())
    weighted = {
        "precision": round(sum(v["precision"] * v["support"] for v in per_class.values()) / total, 4),
        "recall":    round(sum(v["recall"]    * v["support"] for v in per_class.values()) / total, 4),
        "f1":        round(sum(v["f1"]        * v["support"] for v in per_class.values()) / total, 4),
    }

    return per_class, macro, weighted


def print_classification_report(
    per_class: dict,
    macro: dict,
    weighted: dict,
    idx_to_cuisine: dict,
) -> None:
    """Precision/Recall/F1 sonuçlarını tablo formatında yazdırır."""
    print("\n" + "─" * 68)
    print("  SINIFLANDIRMA RAPORU")
    print("─" * 68)
    header = f"  {'Mutfak':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Destek':>8}"
    print(header)
    print("─" * 68)

    for idx, vals in per_class.items():
        cuisine = idx_to_cuisine.get(idx, str(idx))
        print(
            f"  {cuisine:<20} "
            f"{vals['precision']:>10.4f} "
            f"{vals['recall']:>10.4f} "
            f"{vals['f1']:>10.4f} "
            f"{vals['support']:>8}"
        )

    print("─" * 68)
    print(
        f"  {'Macro Ort.':<20} "
        f"{macro['precision']:>10.4f} "
        f"{macro['recall']:>10.4f} "
        f"{macro['f1']:>10.4f}"
    )
    print(
        f"  {'Weighted Ort.':<20} "
        f"{weighted['precision']:>10.4f} "
        f"{weighted['recall']:>10.4f} "
        f"{weighted['f1']:>10.4f}"
    )
    print("─" * 68)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """
    (n_classes × n_classes) Confusion Matrix oluşturur.

    cm[i][j] = Gerçekte i sınıfı olan, j sınıfı olarak tahmin edilen örnek sayısı.
    Köşegen (i==j) doğru tahminleri gösterir.

    Args:
        y_true    : Gerçek etiketler
        y_pred    : Tahmin edilen etiketler
        n_classes : Toplam sınıf sayısı

    Returns:
        cm : (n_classes, n_classes) int array
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm


def print_confusion_matrix(
    cm: np.ndarray,
    idx_to_cuisine: dict,
    top_n: int = 8,
) -> None:
    """
    Confusion matrix'in en çok karıştırılan sınıf çiftlerini yazdırır.
    Tam matris çok büyük olduğu için (20×20) sadece en kötü off-diagonal hücreler gösterilir.

    Args:
        cm           : confusion_matrix() çıktısı
        idx_to_cuisine : index → mutfak adı eşlemesi
        top_n        : Kaç hata çifti gösterilsin
    """
    n = cm.shape[0]

    # Doğruluk (köşegen)
    correct = cm.diagonal().sum()
    total   = cm.sum()
    print(f"\n  Genel doğruluk (CM'den): {correct}/{total} = {correct/total*100:.2f}%")

    # Per-class recall (satır bazlı doğruluk)
    print("\n  Sınıf bazlı doğruluk (Recall):")
    recalls = []
    for i in range(n):
        row_sum = cm[i].sum()
        recall  = cm[i][i] / row_sum if row_sum > 0 else 0.0
        recalls.append((recall, idx_to_cuisine.get(i, str(i)), row_sum))
    recalls.sort()  # en düşük recall öne

    for recall, cuisine, support in recalls:
        bar = "█" * max(1, int(recall * 30))
        print(f"    {cuisine:<20} {bar:<32} {recall*100:.1f}%  (n={support})")

    # En çok karıştırılan çiftler (off-diagonal maksimumlar)
    print(f"\n  En çok karıştırılan {top_n} çift:")
    print(f"  {'Gerçek':<20} {'Tahmin':<20} {'Sayı':>6}")
    print("  " + "─" * 48)

    off_diag = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                off_diag.append((cm[i][j], i, j))
    off_diag.sort(reverse=True)

    for count, i, j in off_diag[:top_n]:
        true_c = idx_to_cuisine.get(i, str(i))
        pred_c = idx_to_cuisine.get(j, str(j))
        print(f"  {true_c:<20} {pred_c:<20} {count:>6}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOG-LOSS (CROSS-ENTROPY)
# ══════════════════════════════════════════════════════════════════════════════

def log_loss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-12) -> float:
    """
    Çok sınıflı Log-Loss (Cross-Entropy) hesaplar.

    Formül:
        L = -1/N * Σ_i log(p(y_i))

    Burada p(y_i), modelin doğru sınıfa atadığı olasılık.
    Düşük değer daha iyi kalibrasyon demektir.

    Referans değerler (20 sınıflı problem için):
        < 0.5  → Mükemmel kalibrasyon
        0.5–1  → İyi
        1–2    → Orta
        > 2    → Zayıf kalibrasyon (model aşırı güveniyor)

    Args:
        y_true  : Gerçek etiketler (n_samples,)
        y_proba : Tahmin olasılıkları (n_samples, n_classes)
        eps     : log(0) hatasını önlemek için küçük sabit

    Returns:
        loss : float — ortalama log-loss
    """
    n = len(y_true)
    # Her örnek için doğru sınıfın olasılığını al
    correct_probs = y_proba[np.arange(n), y_true]
    correct_probs = np.clip(correct_probs, eps, 1.0)
    return float(-np.mean(np.log(correct_probs)))


def print_log_loss(loss: float, y_proba: np.ndarray, y_true: np.ndarray) -> None:
    """Log-loss sonucunu ve kalibrasyon yorumunu yazdırır."""
    print("\n" + "─" * 68)
    print("  LOG-LOSS (CROSS-ENTROPY) — KALİBRASYON ANALİZİ")
    print("─" * 68)
    print(f"  Log-Loss: {loss:.4f}")

    if loss < 0.5:
        yorum = "Mükemmel — model olasılıklarına güvenilebilir."
    elif loss < 1.0:
        yorum = "İyi — genel kalibrasyon yeterli."
    elif loss < 2.0:
        yorum = "Orta — model bazı tahminlerde aşırı güvenli."
    else:
        yorum = "Zayıf — model olasılıkları güvenilir değil, kalibrasyon gerekli."
    print(f"  Yorum   : {yorum}")

    # Ortalama maksimum olasılık (modelin ne kadar emin olduğu)
    max_probs = y_proba.max(axis=1)
    avg_confidence = max_probs.mean()
    print(f"  Ortalama model güveni (max prob): {avg_confidence*100:.1f}%")

    # Aşırı güvenli tahminler (%99+ olasılık)
    overconfident = (max_probs >= 0.99).sum()
    print(f"  %99+ güvenle yapılan tahmin sayısı: {overconfident} / {len(y_true)}")
    print("─" * 68)


# ══════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON — Tüm metrikleri birden çalıştır
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    idx_to_cuisine: dict,
    label: str = "Validasyon",
) -> None:
    """
    Tüm metrikleri hesaplayıp yazdırır.

    Args:
        model          : Eğitilmiş CuisineClassifier
        X              : Dense feature matrisi (n_samples, n_vocab)
        y_true         : Gerçek etiketler
        idx_to_cuisine : index → mutfak adı eşlemesi
        label          : Çıktı başlığı için etiket
    """
    print(f"\n{'═'*68}")
    print(f"  METRİK RAPORU — {label.upper()}")
    print(f"{'═'*68}")

    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)
    n_classes = model.n_classes

    # 1. Precision / Recall / F1
    per_class, macro, weighted = precision_recall_f1(y_true, y_pred, n_classes)
    print_classification_report(per_class, macro, weighted, idx_to_cuisine)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, n_classes)
    print_confusion_matrix(cm, idx_to_cuisine)

    # 3. Log-Loss
    loss = log_loss(y_true, y_proba)
    print_log_loss(loss, y_proba, y_true)
