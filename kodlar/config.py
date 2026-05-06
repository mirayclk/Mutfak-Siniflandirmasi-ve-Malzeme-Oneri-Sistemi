"""
pip install -r requirements.txt

# Normal çalıştır (eğit + interaktif mod)
python main.py

# Sadece eğitim + doğruluk ölçümü
python main.py --no-interactive

# Test tahminlerini submission.csv olarak kaydet
python main.py --save-csv


config.py — Proje genelinde kullanılan sabit değerler ve ayarlar.
"""

# ── Dosya Yolları ──────────────────────────────────────────────────────────────
TRAIN_PATH = "train.json"
TEST_PATH  = "test.json"
OUTPUT_PATH = "submission.csv"

# ── Model Hiperparametreleri ───────────────────────────────────────────────────
ALPHA_PRIOR      = 1.1    # Dirichlet düzleştirme sabiti
IDF_SMOOTH       = 1.0    # IDF formülüne eklenen sabit (log((N+1)/(df+1)) + IDF_SMOOTH)

# ── Temizleme ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    "fresh", "dried", "ground", "chopped", "sliced", "minced",
    "large", "small", "medium", "whole", "half", "low", "reduced",
    "fat", "free", "plain", "pure", "organic", "extra", "virgin",
    "boneless", "skinless", "frozen", "cooked", "raw", "shredded",
    "grated", "crushed", "packed", "firmly", "finely", "thinly",
    "lightly", "divided", "softened", "drained", "rinsed", "peeled",
    "seeded", "trimmed", "quartered", "diced", "cut", "into",
    "pieces", "taste", "needed", "optional",
}

# ── Öneri Parametreleri ────────────────────────────────────────────────────────
DEFAULT_TOP_K_CUISINE    = 3   # Kaç mutfak gösterilsin
DEFAULT_TOP_K_INGREDIENT = 5   # Kaç malzeme önerilsin
