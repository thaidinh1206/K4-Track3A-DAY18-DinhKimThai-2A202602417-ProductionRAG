from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    """Reuse model weights across reranker instances."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0 or not query.strip():
            return []
        import numpy as np

        pairs = [(query, doc["text"]) for doc in documents]
        scores = np.asarray(self._load_model().predict(
            pairs, batch_size=8, show_progress_bar=False,
        )).reshape(-1)
        if len(scores) != len(documents):
            raise ValueError("Cross-encoder must return one score per document")
        scored = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        return [RerankResult(
            text=doc["text"], original_score=float(doc.get("score", 0.0)),
            rerank_score=float(score), metadata=dict(doc.get("metadata", {})), rank=i,
        ) for i, (score, doc) in enumerate(scored[:top_k])]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0 or not query.strip():
            return []
        from flashrank import Ranker, RerankRequest

        if self._model is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     ".cache", "flashrank")
            self._model = Ranker(cache_dir=cache_dir)
        passages = [{"id": i, "text": doc["text"]}
                    for i, doc in enumerate(documents)]
        ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
        ranked = sorted(ranked, key=lambda passage: passage["score"], reverse=True)
        results = []
        for rank, passage in enumerate(ranked[:top_k]):
            doc = documents[passage["id"]]
            results.append(RerankResult(
                text=doc["text"], original_score=float(doc.get("score", 0.0)),
                rerank_score=float(passage["score"]),
                metadata=dict(doc.get("metadata", {})), rank=rank,
            ))
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
