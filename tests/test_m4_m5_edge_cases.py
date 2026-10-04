"""Evaluation failures, provider adapters, enrichment schema and evidence integrity."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import m4_eval as m4
from src import m5_enrichment as m5


def test_eval_no_key_and_empty_dataset_are_explicitly_skipped():
    result = m4.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["status"] == "skipped" and result["per_question"] == []
    assert result["num_requested"] == 1
    assert m4.evaluate_ragas([], [], [], [])["status"] == "skipped"


def test_eval_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        m4.evaluate_ragas(["q"], [], [["c"]], ["gt"])
    with pytest.raises(ValueError, match="lists of strings"):
        m4.evaluate_ragas(["q"], ["a"], ["not a context list"], ["gt"])


def fake_evaluation(monkeypatch, rows):
    import ragas

    closed = []
    evaluator = SimpleNamespace(close=lambda: closed.append(True))
    embeddings = object()
    monkeypatch.setattr(m4, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(m4, "_build_evaluators", lambda: (evaluator, embeddings, "run-config"))

    def evaluate(dataset, **kwargs):
        assert kwargs["llm"] is evaluator and kwargs["embeddings"] is embeddings
        assert {metric.name for metric in kwargs["metrics"]} == set(m4.METRIC_NAMES)
        assert len(dataset) == len(rows)
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))

    monkeypatch.setattr(ragas, "evaluate", evaluate)
    return closed


def test_eval_converts_scores_and_preserves_question_evidence(monkeypatch, tmp_path):
    closed = fake_evaluation(monkeypatch, [dict.fromkeys(m4.METRIC_NAMES, 0.8)])
    result = m4.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["status"] == "ok" and result["faithfulness"] == 0.8
    assert result["per_question"][0].contexts == ["c"]
    assert closed == [True]
    report_path = tmp_path / "report.json"
    m4.save_report(result, m4.failure_analysis(result["per_question"]), str(report_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["aggregate"] == dict.fromkeys(m4.METRIC_NAMES, 0.8)
    assert report["per_question"][0]["ground_truth"] == "gt"
    assert report["evaluation"]["status"] == "ok"


def test_eval_nan_is_flagged_and_excluded_from_aggregate(monkeypatch):
    first = dict.fromkeys(m4.METRIC_NAMES, 0.8)
    first["faithfulness"] = np.nan
    fake_evaluation(monkeypatch, [first, dict.fromkeys(m4.METRIC_NAMES, 0.6)])
    result = m4.evaluate_ragas(["q1", "q2"], ["a", "a"], [["c"], ["c"]], ["gt", "gt"])
    assert result["status"] == "partial"
    assert result["faithfulness"] == pytest.approx(0.6)
    assert result["context_recall"] == pytest.approx(0.7)
    assert result["per_question"][0].evaluation_errors == ["faithfulness"]
    assert len(m4.failure_analysis(result["per_question"])) == 1


def test_eval_exception_returns_failed_status(monkeypatch):
    monkeypatch.setattr(m4, "LLM_API_KEY", "fake-key")

    def fail():
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(m4, "_build_evaluators", fail)
    result = m4.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["status"] == "failed" and "provider unavailable" in result["error"]
    assert all(result[name] == 0 for name in m4.METRIC_NAMES)


def test_failure_analysis_selects_bottom_five_and_diagnoses_recall():
    rows = [m4.EvalResult(f"q{i}", "answer", ["context"], "truth", 0.9, 0.9, 0.8, i / 10)
            for i in range(7)]
    failures = m4.failure_analysis(rows, bottom_n=5)
    assert [f["question"] for f in failures] == [f"q{i}" for i in range(5)]
    assert failures[0]["worst_metric"] == "context_recall"
    assert failures[0]["suggested_fix"] and failures[0]["error_tree"]
    assert m4.failure_analysis(rows, bottom_n=0) == []


def test_bottom_ranked_high_scores_are_not_declared_failures():
    row = m4.EvalResult("q", "a", ["c"], "gt", 1.0, 0.99, 1.0, 1.0)
    failure = m4.failure_analysis([row], bottom_n=1)[0]
    assert "no low-score failure" in failure["diagnosis"]


def fake_client(monkeypatch, content, reason="stop"):
    calls = []

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=reason,
            )])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(m5, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(m5, "create_llm_client", Client)
    return calls


def test_combined_uses_one_call_and_indexes_generated_cues(monkeypatch):
    payload = {"summary": "Phép năm 15 ngày.", "context": "Chính sách nghỉ phép.",
               "questions": ["Bao nhiêu ngày phép?"],
               "metadata": {"topic": "leave", "source": "invented.md", "parent_id": "wrong"}}
    calls = fake_client(monkeypatch, json.dumps(payload))
    metadata = {"source": "real.md", "parent_id": "p1", "chunk_index": 3}
    text = "Nhân viên chính thức được nghỉ 15 ngày phép."
    result = m5.enrich_chunks([{"text": text, "metadata": metadata}])[0]
    assert len(calls) == 1 and calls[0]["response_format"] == {"type": "json_object"}
    assert result.original_text == text and text in result.enriched_text
    assert payload["summary"] in result.enriched_text
    assert payload["questions"][0] in result.enriched_text
    assert result.auto_metadata["source"] == "real.md"
    assert result.auto_metadata["parent_id"] == "p1"
    assert result.auto_metadata["enrichment_status"] == "llm"
    assert metadata == {"source": "real.md", "parent_id": "p1", "chunk_index": 3}


@pytest.mark.parametrize("content", ["bad JSON", "[]", '{"summary":7}', '{"summary":"s","context":"c","questions":null}'])
def test_invalid_combined_output_falls_back_without_extra_calls(monkeypatch, content):
    calls = fake_client(monkeypatch, content)
    text = "Không được nghỉ phép trong thử việc."
    result = m5.enrich_chunks([{"text": text, "metadata": {"source": "trial.md"}}])[0]
    assert len(calls) == 1
    assert result.auto_metadata["enrichment_status"] == "fallback"
    assert text in result.enriched_text and result.hypothesis_questions


def test_truncated_response_and_missing_key_use_local_fallback(monkeypatch):
    calls = fake_client(monkeypatch, "unfinished", reason="length")
    result = m5._enrich_single_call("Nghỉ phép 15 ngày.", "leave.md")
    assert result["status"] == "fallback" and len(calls) == 1
    monkeypatch.setattr(m5, "LLM_API_KEY", "")
    m5._enrich_single_call("Nghỉ phép 15 ngày.", "leave.md")
    assert len(calls) == 1


def test_local_metadata_keeps_version_date_and_original_text():
    text = "# Chính sách nghỉ phép\n> Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024\n\n15 ngày."
    result = m5.enrich_chunks([{"text": text, "metadata": {"source": "leave.md"}}])[0]
    assert result.auto_metadata["version"] == "2.0"
    assert result.auto_metadata["effective_date"] == "01/01/2024"
    assert text in result.enriched_text
    assert result.enriched_text != text


def test_individual_methods_parse_json_questions_and_keep_raw_text(monkeypatch):
    fake_client(monkeypatch, '```json\n{"topic":"VPN", "entities":["MFA"], "source":"fake"}\n```')
    metadata = m5.extract_metadata("VPN cần MFA.")
    assert metadata["topic"] == "VPN" and "source" not in metadata
    fake_client(monkeypatch, "1. Ai cần MFA?\n2. VPN dùng khi nào?")
    assert m5.generate_hypothesis_questions("VPN cần MFA.", 1) == ["Ai cần MFA?"]
    assert m5.generate_hypothesis_questions("text", 0) == []
    fake_client(monkeypatch, "Trích từ chính sách VPN.")
    assert "VPN cần MFA." in m5.contextual_prepend("VPN cần MFA.", "IT")
    assert m5.enrich_chunks([], ["contextual"]) == []
    with pytest.raises(ValueError):
        m5.enrich_chunks([], ["unknown"])


def test_ragas_adapter_makes_three_single_completions():
    from ragas.run_config import RunConfig
    from src.ragas_adapters import ProviderRagasLLM

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        assert "n" not in kwargs
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=f"answer{len(calls)}"), finish_reason="stop",
        )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = ProviderRagasLLM(client, "openai/gpt-4o-mini", RunConfig())
    prompt = SimpleNamespace(to_string=lambda: "metric prompt")
    result = asyncio.run(llm.agenerate_text(prompt, n=3))
    assert len(calls) == 3
    assert [g.text for g in result.generations[0]] == ["answer1", "answer2", "answer3"]


def test_local_embedding_adapter_reuses_encoder(monkeypatch):
    from src import ragas_adapters

    encoder = SimpleNamespace(encode=lambda texts, **kwargs: np.array([[1.0, 0.0]] * len(texts)))
    monkeypatch.setattr(ragas_adapters, "_load_dense_encoder", lambda: encoder)
    embeddings = ragas_adapters.LocalBGEEmbeddings()
    assert embeddings.embed_documents([]) == []
    assert embeddings.embed_query("query") == [1.0, 0.0]
    assert asyncio.run(embeddings.aembed_documents(["a", "b"])) == [[1.0, 0.0], [1.0, 0.0]]
