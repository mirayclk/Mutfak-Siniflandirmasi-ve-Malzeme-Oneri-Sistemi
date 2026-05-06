"""
model.py — IDF-ağırlıklı Multinomial Naive Bayes sınıflandırıcı.

Teorik temel:
  - Her mutfak için her malzemenin posterior olasılığı:
        θ_jc = (x_jc · idf_j + α) / (Σ_j x_jc · idf_j + α · V)
    Bu formül Laplace (α=1) veya Lidstone düzleştirmesinin IDF-ağırlıklı versiyonudur.
  - Orijinal koddan düzeltilen hata: pay/payda'da  -k  terimi yoktu;
    doğru Multinomial NB normalizer yalnızca toplam count + prior toplamıdır.
"""

from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix

from config import ALPHA_PRIOR, IDF_SMOOTH
from data import compute_idf, vectorize, clean_recipe


class CuisineClassifier:
    """
    IDF-ağırlıklı Multinomial Naive Bayes sınıflandırıcı.

    Attributes:
        n_classes        : Mutfak sayısı
        log_prior_       : (n_classes,) — log P(cuisine)
        log_likelihood_  : (n_classes, n_vocab) — log P(ingredient | cuisine)
        idf_             : (n_vocab,) — IDF ağırlıkları
        cuisine_counts_  : (n_classes, n_vocab) — ham frekans sayıları (öneri için)
    """

    def __init__(self, alpha: float = ALPHA_PRIOR, idf_smooth: float = IDF_SMOOTH):
        self.alpha = alpha
        self.idf_smooth = idf_smooth

    # ── EĞİTİM ────────────────────────────────────────────────────────────────

    def fit(self, X: csr_matrix, y: np.ndarray) -> "CuisineClassifier":
        """
        Modeli eğit.

        Args:
            X : (n_train, n_vocab) sparse binary matris
            y : (n_train,) int sınıf etiketleri
        """
        n_samples, n_vocab = X.shape
        self.n_classes = int(y.max()) + 1
        self.n_vocab = n_vocab

        # IDF (sadece eğitim verisi üzerinden)
        self.idf_ = compute_idf(X, smooth=self.idf_smooth)

        # Log prior: P(cuisine)
        counts = np.bincount(y, minlength=self.n_classes)
        self.log_prior_ = np.log(counts / n_samples + 1e-12)

        # Log likelihood: P(ingredient | cuisine)
        self.log_likelihood_ = np.zeros((self.n_classes, n_vocab))
        self.cuisine_counts_ = np.zeros((self.n_classes, n_vocab))

        for c in range(self.n_classes):
            mask = y == c
            raw_counts = np.asarray(X[mask].sum(axis=0)).flatten()
            self.cuisine_counts_[c] = raw_counts

            # IDF ağırlıklandırma
            weighted = raw_counts * self.idf_
            theta = (weighted + self.alpha) / (weighted.sum() + self.alpha * n_vocab)
            self.log_likelihood_[c] = np.log(theta + 1e-12)

        return self

    # ── TAHMİN ────────────────────────────────────────────────────────────────

    def _log_proba(self, X: np.ndarray) -> np.ndarray:
        """Ham log-posterior skorları döndür. X: (n_samples, n_vocab) dense."""
        return (X * self.idf_) @ self.log_likelihood_.T + self.log_prior_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalize edilmiş olasılıklar. (n_samples, n_classes)"""
        log_p = self._log_proba(X)
        log_p -= log_p.max(axis=1, keepdims=True)   # numerical stability
        probs = np.exp(log_p)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """En yüksek olasılıklı sınıf etiketleri. (n_samples,)"""
        return np.argmax(self._log_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Doğruluk oranı (accuracy)."""
        return float((self.predict(X) == y).mean())

    # ── TEK ÖRNEK TAHMİNİ ─────────────────────────────────────────────────────

    def predict_cuisine_from_vec(
        self,
        user_vec: np.ndarray,
        idx_to_cuisine: Dict[int, str],
        top_k: int = 3
    ) -> List[dict]:
        """
        Tek bir binary vektörden en olası K mutfağı döndür.

        Returns:
            [{"cuisine": str, "probability": float}, ...]
        """
        probs = self.predict_proba(user_vec.reshape(1, -1))[0]
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [
            {"cuisine": idx_to_cuisine[i], "probability": round(float(probs[i]), 4)}
            for i in top_idx
        ]
