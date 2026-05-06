"""
data.py — Veri yükleme, malzeme temizleme ve vektörizasyon işlemleri.
"""

import json
import re
from typing import Dict, List, Tuple

import numpy as np
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, lil_matrix

from config import STOPWORDS, TRAIN_PATH, TEST_PATH

# ── Stemmer (tek instance, modül seviyesinde) ──────────────────────────────────
_stemmer = PorterStemmer()


# ── TEMİZLEME ─────────────────────────────────────────────────────────────────

def clean_ingredient(ingredient: str) -> str:
    """
    Tek bir malzemeyi normalize et:
      - küçük harfe çevir
      - özel karakter temizle
      - stopword ve tek-harf token'ları çıkar
      - Porter stemming uygula
    """
    ing = ingredient.lower()
    ing = re.sub(r"[^a-z\s]", " ", ing)
    ing = re.sub(r"\s+", " ", ing).strip()

    tokens = [
        _stemmer.stem(w)
        for w in ing.split()
        if w not in STOPWORDS and len(w) > 1
    ]
    return " ".join(tokens)


def clean_recipe(ingredients: List[str]) -> List[str]:
    """
    Bir tarifte tüm malzemeleri temizle.
    Boş string ve saf rakam sonuçları filtrele.
    """
    cleaned = [clean_ingredient(ing) for ing in ingredients]
    return [ing for ing in cleaned if ing and not ing.isdigit()]


# ── VERİ YÜKLEME ──────────────────────────────────────────────────────────────

def load_data(
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH
) -> Tuple[list, list]:
    """Train ve test JSON dosyalarını yükle."""
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test = json.load(f)
    print(f"  Train: {len(train)} tarif  |  Test: {len(test)} tarif")
    return train, test


# ── VOCABULARY ────────────────────────────────────────────────────────────────

def build_vocabulary(train_data: list) -> Dict[str, int]:
    """
    Vocabulary'i YALNIZCA train verisinden oluştur.
    Test verisini dahil etmek data leakage'a yol açar.
    """
    vocab: set = set()
    for ex in train_data:
        for ing in clean_recipe(ex["ingredients"]):
            vocab.add(ing)
    vocab_dict = {ing: i for i, ing in enumerate(sorted(vocab))}
    print(f"  Vocabulary boyutu: {len(vocab_dict)} unique malzeme")
    return vocab_dict


# ── VEKTÖRİZASYON ─────────────────────────────────────────────────────────────

def vectorize(ingredients: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """
    Malzeme listesini binary vektöre çevir (vocab'da olmayan malzemeler atlanır).
    """
    vec = np.zeros(len(vocab), dtype=np.float32)
    for ing in clean_recipe(ingredients):
        if ing in vocab:
            vec[vocab[ing]] = 1.0
    return vec


def build_sparse_matrix(
    data: list,
    vocab: Dict[str, int],
    cuisine_to_idx: Dict[str, int],
    has_labels: bool = True,
) -> Tuple:
    """
    Veri listesinden sparse (CSR) matris oluştur.

    Returns:
        X       : csr_matrix (n_samples, n_vocab)
        y       : np.ndarray int32 — sadece has_labels=True ise dolu
        ids     : list[int]
    """
    n = len(data)
    n_vocab = len(vocab)
    X = lil_matrix((n, n_vocab), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    ids = []

    for j, ex in enumerate(data):
        ids.append(ex["id"])
        if has_labels:
            y[j] = cuisine_to_idx[ex["cuisine"]]
        for ing in clean_recipe(ex["ingredients"]):
            if ing in vocab:
                X[j, vocab[ing]] = 1.0

    return csr_matrix(X), y, ids


# ── IDF ───────────────────────────────────────────────────────────────────────

def compute_idf(X: csr_matrix, smooth: float = 1.0) -> np.ndarray:
    """
    Smooth IDF hesapla:  log((N+1) / (df+1))  +  smooth

    Yaygın malzemeleri (salt, water, oil) baskılar,
    ayırt edici malzemelerin ağırlığını artırır.
    """
    N = X.shape[0]
    df = np.asarray((X > 0).sum(axis=0)).flatten()
    return np.log((N + 1) / (df + 1)) + smooth
