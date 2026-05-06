"""
recommender.py — Eksik malzeme önerisi modülü.

Yaklaşım (orijinal iki versiyonun birleştirilmiş ve düzeltilmiş hali):
  1. Modelden en iyi mutfağı bul.
  2. O mutfağa ait ham frekansları IDF ile ağırlıklandır (cuisine_freq × idf).
  3. Cosine benzerliği ile en benzer tarifleri bul; yerel frekansı hesapla.
  4. final_score = cuisine_idf_score × (0.7 + 0.3 × local_score)
     - Bu formül hem global mutfak profilini hem de kullanıcıya özgü
       benzer tarifleri dengeli şekilde birleştirir.
  5. Kullanıcının zaten girdiği malzemeleri filtrele ve en yüksek skorluları sun.

Neden bu yaklaşım daha doğru (orijinal iki versiyona kıyasla):
  - Basit versiyon (recommend_ingredients): Sadece Dirichlet posterior kullanır,
    IDF yoktur → "salt", "water", "oil" her zaman 1. sıraya çıkar.
  - Smart versiyon (recommend_ingredients_smart): IDF + local var ama
    user_vec posterior'a toplanıyor (kavramsal hata). Burada düzeltildi:
    user_vec posterior hesabına dahil edilmiyor; sadece cosine benzerliği
    için kullanılıyor.
"""

from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix

from data import vectorize, clean_recipe
from model import CuisineClassifier
from config import DEFAULT_TOP_K_INGREDIENT


def _find_similar_recipes(
    user_vec: np.ndarray,
    X_train: csr_matrix,
    top_n: int = 100
) -> np.ndarray:
    """
    Kullanıcı vektörüne en benzer tariflerin indekslerini cosine similarity ile bul.
    Sıfır vektör kontrolü dahil.
    """
    user_norm = np.linalg.norm(user_vec)
    if user_norm < 1e-6:
        # Vocabulary'de hiç malzeme bulunamadı — tüm tarifleri eşit ağırlıkla döndür
        return np.arange(min(top_n, X_train.shape[0]))

    X_dense = np.asarray(X_train.todense())
    row_norms = np.linalg.norm(X_dense, axis=1) + 1e-8
    sims = (X_dense @ user_vec) / (row_norms * user_norm)
    return np.argsort(sims)[::-1][:top_n]


def recommend_ingredients(
    user_ingredients: List[str],
    model: CuisineClassifier,
    vocab: Dict[str, int],
    idx_to_cuisine: Dict[int, str],
    X_train: csr_matrix,
    y_train: np.ndarray,
    top_k: int = DEFAULT_TOP_K_INGREDIENT,
) -> List[dict]:
    """
    Kullanıcının girdiği malzemelere göre eksik malzeme öner.

    Adımlar:
      1. Mutfak tahmini (modelden)
      2. O mutfağın IDF-ağırlıklı malzeme skoru
      3. Benzer tarif yerel skoru (cosine)
      4. Bileşik skor = cuisine_idf × (0.7 + 0.3 × local)
      5. Kullanıcı malzemelerini filtrele, en yüksek top_k'yı döndür

    Args:
        user_ingredients : Kullanıcının girdiği ham malzeme listesi
        model            : Eğitilmiş CuisineClassifier
        vocab            : malzeme → indeks sözlüğü
        idx_to_cuisine   : indeks → mutfak adı sözlüğü
        X_train          : Eğitim sparse matrisi
        y_train          : Eğitim etiketleri
        top_k            : Kaç malzeme önerilecek

    Returns:
        [{"ingredient": str, "score": float, "cuisine": str}, ...]
    """
    n_vocab = len(vocab)
    idx_to_ing = {i: ing for ing, i in vocab.items()}

    # 1. Kullanıcı vektörü
    user_vec = vectorize(user_ingredients, vocab)
    if user_vec.sum() == 0:
        print("  [UYARI] Girilen malzemelerin hiçbiri vocabulary'de bulunamadı.")
        return []

    # 2. Mutfak tahmini
    probs = model.predict_proba(user_vec.reshape(1, -1))[0]
    best_c = int(np.argmax(probs))
    best_cuisine = idx_to_cuisine[best_c]

    # 3. Global mutfak skoru (IDF ağırlıklı frekans)
    cuisine_raw = model.cuisine_counts_[best_c]           # (n_vocab,)
    cuisine_total = cuisine_raw.sum() + 1e-8
    cuisine_freq = cuisine_raw / cuisine_total             # normalize
    cuisine_idf_score = cuisine_freq * model.idf_         # IDF baskılama

    # 4. Yerel (benzer tarifler) skoru
    similar_idx = _find_similar_recipes(user_vec, X_train, top_n=100)
    local_counts = np.asarray(X_train[similar_idx].sum(axis=0)).flatten()
    local_score = local_counts / (local_counts.sum() + 1e-8)

    # 5. Bileşik skor
    final_score = cuisine_idf_score * (0.7 + 0.3 * local_score)

    # 6. Kullanıcının girdiklerini çıkar
    user_cleaned = set(clean_recipe(user_ingredients))
    for ing in user_cleaned:
        if ing in vocab:
            final_score[vocab[ing]] = 0.0

    # 7. En yüksek top_k
    top_idx = np.argsort(final_score)[::-1][:top_k]
    return [
        {
            "ingredient": idx_to_ing[i],
            "score": round(float(final_score[i]), 6),
            "cuisine": best_cuisine,
        }
        for i in top_idx
        if final_score[i] > 0
    ]
