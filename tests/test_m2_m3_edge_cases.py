"""Search/index integration and reranking edge cases without model downloads."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src import m2_search as m2
from src import m3_rerank as m3


class DeterministicEncoder:
    def encode(self, texts, **kwargs):
        def vector(text):
            result = np.zeros(m2.EMBEDDING_DIM, dtype=np.float32)
            result[0 if "nghỉ" in text.lower() else 1] = 1.0
            return result

        if isinstance(texts, str):
            return vector(texts)
        return np.array([vector(text) for text in texts])


@pytest.fixture
def dense():
    # Exercise actual Qdrant APIs locally, without connecting to Docker.
    search = m2.DenseSearch.__new__(m2.DenseSearch)
    search.client = QdrantClient(":memory:")
    search._encoder = DeterministicEncoder()
    yield search
    search.client.close()


def test_dense_round_trip_preserves_metadata(dense):
    metadata = {"source": "leave.md", "parent_id": "parent_leave", "chunk_index": 2}
    chunks = [{"text": "Được nghỉ phép 15 ngày.", "metadata": metadata},
              {"text": "VPN cần MFA.", "metadata": {"source": "it.md"}}]
    dense.index(chunks)
    results = dense.search("nghỉ phép", top_k=1)
    assert len(results) == 1
    assert results[0].text == chunks[0]["text"]
    assert results[0].metadata == metadata
    assert results[0].method == "dense"
    assert results[0].score == pytest.approx(1.0)
    assert "text" not in metadata


def test_dense_reindex_removes_stale_points_and_keeps_other_collection(dense):
    chunks = [{"text": "nghỉ phép"}, {"text": "VPN"}]
    dense.index(chunks, collection="production")
    dense.index(chunks, collection="baseline")
    dense.index([chunks[0]], collection="production")
    assert dense.client.count("production").count == 1
    assert dense.client.count("baseline").count == 2


def test_dense_empty_index_and_invalid_queries_do_not_load_model(dense, monkeypatch):
    def unexpected_encoder():
        raise AssertionError("No encoder is needed")

    monkeypatch.setattr(dense, "_get_encoder", unexpected_encoder)
    assert dense.search("nghỉ phép", collection="missing") == []
    dense.index([{"text": "  "}])
    assert dense.search("nghỉ phép") == []
    assert dense.search(" ") == []
    assert dense.search("nghỉ phép", top_k=0) == []


def test_dense_dimension_error_keeps_previous_index(dense):
    dense.index([{"text": "nghỉ phép"}])
    dense._encoder = SimpleNamespace(encode=lambda texts, **kwargs: np.zeros((len(texts), 3)))
    with pytest.raises(ValueError, match="Expected embeddings"):
        dense.index([{"text": "replacement"}])
    assert dense.client.count(m2.COLLECTION_NAME).count == 1
    assert dense.client.scroll(m2.COLLECTION_NAME)[0][0].payload["text"] == "nghỉ phép"


def test_dense_uploads_multiple_batches_without_losing_points(dense):
    dense.index([{"text": f"nghỉ phép {i}"} for i in range(65)])
    assert dense.client.count(m2.COLLECTION_NAME).count == 65


def test_hybrid_search_combines_actual_bm25_and_local_dense(dense, monkeypatch):
    monkeypatch.setattr(m2, "DenseSearch", lambda: dense)
    search = m2.HybridSearch()
    search.index([
        {"text": "Nhân viên nghỉ phép năm 15 ngày.", "metadata": {"source": "leave.md"}},
        {"text": "Mật khẩu cần thay đổi định kỳ.", "metadata": {"source": "password.md"}},
        {"text": "VPN phải có MFA.", "metadata": {"source": "vpn.md"}},
    ])
    results = search.search("nghỉ phép", top_k=2)
    assert len(results) == 2
    assert results[0].metadata["source"] == "leave.md"
    assert all(result.method == "hybrid" for result in results)
    assert search.search("nghỉ phép", top_k=0) == []


def test_bm25_case_normalization_empty_and_unknown_queries():
    search = m2.BM25Search()
    assert search.search("nghỉ phép") == []
    search.index([{"text": "NHÂN VIÊN NGHỈ PHÉP"},
                  {"text": "VPN cần MFA"}, {"text": "Thưởng cuối năm"}])
    results = search.search("nghỉ phép")
    assert results and results[0].text == "NHÂN VIÊN NGHỈ PHÉP"
    assert search.search("qwertyzzzz") == []
    assert search.search("nghỉ phép", top_k=0) == []
    search.index([])
    assert search.search("nghỉ phép") == []
    search.index([{"text": "   "}])
    assert search.search("nghỉ phép") == []


def test_rrf_exact_scores_deduplicates_and_keeps_input_scores():
    first = m2.SearchResult("leave", 100.0, {"source": "leave.md"}, "bm25")
    second = m2.SearchResult("vpn", 999.0, {}, "bm25")
    dense = [m2.SearchResult("vpn", 0.9, {}, "dense"),
             m2.SearchResult("leave", 0.8, first.metadata, "dense")]
    results = m2.reciprocal_rank_fusion([[first, second, first], dense])
    assert len(results) == 2
    assert all(r.score == pytest.approx(1 / 61 + 1 / 62) for r in results)
    assert first.score == 100.0 and first.method == "bm25"
    assert m2.reciprocal_rank_fusion([], top_k=3) == []
    assert m2.reciprocal_rank_fusion([[first]], top_k=0) == []
    with pytest.raises(ValueError):
        m2.reciprocal_rank_fusion([[first]], k=-1)


@pytest.mark.parametrize("prediction", [0.7, np.array([0.7]), np.array([[0.7]])])
def test_reranker_handles_single_score_and_preserves_metadata(prediction):
    reranker = m3.CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs, **kwargs: prediction)
    doc = {"text": "nghỉ phép", "score": 0.2, "metadata": {"parent_id": "p1"}}
    results = reranker.rerank("nghỉ phép?", [doc])
    assert len(results) == 1
    assert results[0].rerank_score == pytest.approx(0.7)
    assert results[0].original_score == 0.2
    assert results[0].metadata == doc["metadata"]
    assert results[0].rank == 0


def test_reranker_uses_pairs_and_changes_retrieval_order():
    reranker = m3.CrossEncoderReranker()

    def predict(pairs, **kwargs):
        assert pairs == [("leave?", "VPN"), ("leave?", "leave"), ("leave?", "salary")]
        return np.array([0.1, 0.9, 0.3])

    reranker._model = SimpleNamespace(predict=predict)
    docs = [{"text": "VPN", "score": 0.9}, {"text": "leave", "score": 0.4},
            {"text": "salary", "score": 0.6}]
    results = reranker.rerank("leave?", docs, top_k=2)
    assert [r.text for r in results] == ["leave", "salary"]
    assert [r.rank for r in results] == [0, 1]
    assert results[0].original_score == 0.4


def test_reranker_empty_inputs_do_not_load_model(monkeypatch):
    def unexpected_load():
        raise AssertionError("No model is needed")

    reranker = m3.CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_load_model", unexpected_load)
    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", [{"text": "text"}], top_k=0) == []
    assert reranker.rerank(" ", [{"text": "text"}]) == []
    with pytest.raises(ValueError):
        m3.benchmark_reranker(reranker, "query", [], n_runs=0)


def test_reranker_rejects_missing_scores():
    reranker = m3.CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs, **kwargs: [0.5])
    with pytest.raises(ValueError, match="one score per document"):
        reranker.rerank("query", [{"text": "first"}, {"text": "second"}])


def test_flashrank_preserves_original_scores_and_reuses_model(monkeypatch):
    loads = []

    class Ranker:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def rerank(self, request):
            assert request.query == "leave?"
            assert [p["id"] for p in request.passages] == [0, 1]
            return [{"id": 0, "score": 0.1}, {"id": 1, "score": 0.9}]

    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(
        Ranker=Ranker, RerankRequest=SimpleNamespace,
    ))
    reranker = m3.FlashrankReranker()
    docs = [{"text": "VPN", "score": 0.8},
            {"text": "leave", "score": 0.4, "metadata": {"source": "leave.md"}}]
    for _ in range(2):
        results = reranker.rerank("leave?", docs, top_k=1)
        assert results[0].text == "leave"
        assert results[0].original_score == 0.4
        assert results[0].rerank_score == 0.9
        assert results[0].metadata == {"source": "leave.md"}
    assert len(loads) == 1
