"""
main.py — Ana giriş noktası.

Kullanım:
    python main.py                  → Modeli eğit, doğruluğu ölç, interaktif moda geç
    python main.py --predict-only   → Sadece interaktif mod (modeli yeniden eğitme)
    python main.py --save-csv       → Test tahminlerini submission.csv olarak kaydet

Gereksinimler:
    pip install nltk scipy numpy scikit-learn
    (wordnet gerekmez — PorterStemmer built-in'dir)
"""

import argparse
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from metrics import evaluate_all

from config import (
    TRAIN_PATH, TEST_PATH, OUTPUT_PATH,
    DEFAULT_TOP_K_CUISINE, DEFAULT_TOP_K_INGREDIENT,
)
from data import (
    load_data, build_vocabulary,
    build_sparse_matrix, vectorize,
)
from model import CuisineClassifier
from recommender import recommend_ingredients


# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI ÇIKTI FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value: float, scale: float = 40) -> str:
    return "█" * max(1, int(value * scale))


def print_cuisine_results(results: list) -> None:
    print("\n  Tahmin edilen mutfaklar:")
    for r in results:
        print(f"    {r['cuisine']:<18} {_bar(r['probability']):<40} {r['probability']*100:.1f}%")


def print_ingredient_results(results: list) -> None:
    if not results:
        print("  Öneri üretilemedi.")
        return
    cuisine = results[0]["cuisine"]
    print(f"\n  Tahmin edilen mutfak: {cuisine}")
    print("  Önerilen malzemeler:")
    max_score = max(r["score"] for r in results) + 1e-9
    for r in results:
        normalized = r["score"] / max_score
        print(f"    {r['ingredient']:<28} {_bar(normalized):<35} (skor: {r['score']:.5f})")


def save_predictions(
    ids: list,
    preds: np.ndarray,
    idx_to_cuisine: dict,
    path: str = OUTPUT_PATH
) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("id,cuisine\n")
            for recipe_id, pred in zip(ids, preds):
                f.write(f"{recipe_id},{idx_to_cuisine[pred]}\n")
        print(f"\n  ✓ Tahminler kaydedildi: {path}")
    except OSError as e:
        print(f"\n  [HATA] Dosya yazılamadı: {e}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL EĞİTİMİ
# ══════════════════════════════════════════════════════════════════════════════

def train_model(train_data: list, test_data: list):
    """
    Vocabulary oluştur, matris hazırla, modeli eğit, doğruluğu ölç.

    Returns:
        model, vocab, idx_to_cuisine, cuisine_to_idx,
        X_train (sparse), y_train, X_test (sparse), ids_test
    """
    # Vocabulary (sadece train)
    vocab = build_vocabulary(train_data)

    # Mutfak etiket eşlemeleri
    cuisines = sorted({ex["cuisine"] for ex in train_data})
    cuisine_to_idx = {c: i for i, c in enumerate(cuisines)}
    idx_to_cuisine = {i: c for c, i in cuisine_to_idx.items()}
    print(f"  Mutfak sayısı: {len(cuisines)}")

    # Matrisler
    print("\nMatrisler hazırlanıyor...")
    X_train, y_train, _ = build_sparse_matrix(
        train_data, vocab, cuisine_to_idx, has_labels=True
    )
    X_test, _, ids_test = build_sparse_matrix(
        test_data, vocab, cuisine_to_idx, has_labels=False
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Validation split
    tr_idx, val_idx = train_test_split(
        np.arange(X_train.shape[0]), test_size=0.15, random_state=42, stratify=y_train
    )
    X_tr_dense = np.asarray(X_train[tr_idx].todense())
    X_val_dense = np.asarray(X_train[val_idx].todense())
  

    # Model
    print("\nModel eğitiliyor...")
    from scipy.sparse import csr_matrix as _csr
    model = CuisineClassifier()
    model.fit(X_train[tr_idx], y_train[tr_idx])

    # Doğruluk
    train_acc = model.score(X_tr_dense, y_train[tr_idx])
    val_acc   = model.score(X_val_dense, y_train[val_idx])
    print(f"  Eğitim doğruluğu    : {train_acc*100:.2f}%")
    print(f"  Validasyon doğruluğu: {val_acc*100:.2f}%")

            # Validation metrikleri
    X_val_dense = np.asarray(X_train[val_idx].todense())   # ← EKLE
    evaluate_all(                                           # ← EKLE
        model          = model,                            # ← EKLE
        X              = X_val_dense,                      # ← EKLE
        y_true         = y_train[val_idx],                 # ← EKLE
        idx_to_cuisine = idx_to_cuisine,                   # ← EKLE
        label          = "Validasyon",                     # ← EKLE
    ) 

    # Tam train üzerinde yeniden eğit (tahmin için)
    model.fit(X_train, y_train)

    return model, vocab, idx_to_cuisine, cuisine_to_idx, X_train, y_train, X_test, ids_test


# ══════════════════════════════════════════════════════════════════════════════
# İNTERAKTİF MOD
# ══════════════════════════════════════════════════════════════════════════════

def interactive_mode(model, vocab, idx_to_cuisine, X_train, y_train) -> None:
    """
    Kullanıcıdan malzeme listesi alarak mutfak tahmini ve öneri döngüsü.
    Çıkmak için: 'q' veya boş satır.
    """
    print("\n" + "═" * 60)
    print("  İNTERAKTİF MUTFAK TAHMİN & MALZEME ÖNERİ SİSTEMİ")
    print("═" * 60)
    print("  Malzemeleri virgülle ayırarak girin.")
    print("  Çıkmak için: q veya Enter\n")

    while True:
        try:
            raw = input("Malzemeler > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Çıkılıyor...")
            break

        if not raw or raw.lower() in {"q", "quit", "exit", "çıkış"}:
            print("  Çıkılıyor...")
            break

        user_ingredients = [ing.strip() for ing in raw.split(",") if ing.strip()]
        if not user_ingredients:
            print("  En az bir malzeme girin.\n")
            continue

        # Mutfak tahmini
        user_vec = vectorize(user_ingredients, vocab)
        cuisine_results = model.predict_cuisine_from_vec(
            user_vec, idx_to_cuisine, top_k=DEFAULT_TOP_K_CUISINE
        )
        print_cuisine_results(cuisine_results)

        # Malzeme önerisi
        ing_results = recommend_ingredients(
            user_ingredients, model, vocab, idx_to_cuisine,
            X_train, y_train, top_k=DEFAULT_TOP_K_INGREDIENT
        )
        print_ingredient_results(ing_results)
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mutfak Sınıflandırıcı")
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Test tahminlerini submission.csv olarak kaydet"
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="İnteraktif modu atla (sadece eğitim + CSV)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MUTFAK SINIFLANDIRICI — Başlatılıyor")
    print("=" * 60)

    # 1. Veri
    print("\nVeri yükleniyor...")
    train_data, test_data = load_data(TRAIN_PATH, TEST_PATH)

    # 2. Model
    model, vocab, idx_to_cuisine, cuisine_to_idx, X_train, y_train, X_test, ids_test = \
        train_model(train_data, test_data)

    # 3. CSV kaydet
    if args.save_csv:
        X_test_dense = np.asarray(X_test.todense())
        y_pred = model.predict(X_test_dense)
        save_predictions(ids_test, y_pred, idx_to_cuisine)

    # 4. İnteraktif
    if not args.no_interactive:
        interactive_mode(model, vocab, idx_to_cuisine, X_train, y_train)


if __name__ == "__main__":
    main()
