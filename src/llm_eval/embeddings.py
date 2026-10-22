from __future__ import annotations

import math
import re
import zlib
from collections.abc import Callable, Sequence

Embedder = Callable[[Sequence[str]], list[list[float]]]
DIM = 512


def hashed_ngram_embedder(texts: Sequence[str]) -> list[list[float]]:
    """Dependency-free embedder: hashed word unigrams and character trigrams.

    It captures lexical/morphological overlap, not deep semantics. Use sentence_transformer_embedder
    for real semantic similarity.
    """
    vectors = []
    for text in texts:
        vec = [0.0] * DIM
        words = re.findall(r"[a-z0-9]+", text.lower())
        features = list(words)
        for word in words:
            padded = f"#{word}#"
            features.extend(padded[i:i + 3] for i in range(len(padded) - 2))
        for feature in features:
            vec[zlib.crc32(feature.encode()) % DIM] += 1.0
        vectors.append(vec)
    return vectors


def sentence_transformer_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Real sentence embeddings. Needs `pip install sentence-transformers` and a one-time model
    download. Not exercised by this repo's tests."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    return embed


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
