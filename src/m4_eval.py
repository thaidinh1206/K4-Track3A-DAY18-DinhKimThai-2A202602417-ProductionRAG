from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
import math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field, asdict, is_dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SET_PATH, LLM_API_KEY, LLM_PROVIDER, LLM_MODEL,
                    create_llm_client, describe_llm_error)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    evaluation_errors: list[str] = field(default_factory=list)


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_evaluators():
    from ragas.run_config import RunConfig
    from src.ragas_adapters import ProviderRagasLLM, LocalBGEEmbeddings

    run_config = RunConfig(timeout=60, max_retries=2, max_wait=5, max_workers=2)
    return (ProviderRagasLLM(create_llm_client(), LLM_MODEL, run_config),
            LocalBGEEmbeddings(), run_config)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Evaluate all four metrics; explicitly label skipped or failed evaluations."""
    if not len(questions) == len(answers) == len(contexts) == len(ground_truths):
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
    if any(not isinstance(c, list) or any(not isinstance(t, str) for t in c) for c in contexts):
        raise ValueError("contexts must be a list of lists of strings")
    fallback = {name: 0.0 for name in METRIC_NAMES}
    fallback.update(per_question=[], status="skipped", num_requested=len(questions),
                    provider=LLM_PROVIDER, model=LLM_MODEL)
    if not questions or not LLM_API_KEY:
        fallback["error"] = "Empty dataset" if not questions else "No API key configured"
        return fallback

    evaluator = None
    try:
        from ragas import evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from datasets import Dataset

        evaluator, embeddings, run_config = _build_evaluators()
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        # Fresh instances avoid retaining evaluator state between repeated runs.
        result = evaluate(dataset, metrics=[Faithfulness(), AnswerRelevancy(),
                                           ContextPrecision(), ContextRecall()],
                          llm=evaluator, embeddings=embeddings, run_config=run_config,
                          raise_exceptions=False)
        records = result.to_pandas().to_dict(orient="records")
        if len(records) != len(questions):
            raise ValueError("RAGAS returned an unexpected number of rows")
        valid_scores = {name: [] for name in METRIC_NAMES}
        per_question, invalid_metrics = [], []
        for i, row in enumerate(records):
            scores, errors = {}, []
            for name in METRIC_NAMES:
                try:
                    score = float(row[name])
                    if not math.isfinite(score):
                        raise ValueError("non-finite score")
                except (KeyError, TypeError, ValueError):
                    scores[name] = 0.0
                    errors.append(name)
                    invalid_metrics.append({"question_index": i, "metric": name})
                else:
                    scores[name] = score
                    valid_scores[name].append(score)
            per_question.append(EvalResult(questions[i], answers[i], list(contexts[i]),
                                           ground_truths[i], **scores, evaluation_errors=errors))
        aggregate = {name: sum(values) / len(values) if values else 0.0
                     for name, values in valid_scores.items()}
        valid_count = sum(len(values) for values in valid_scores.values())
        status = "ok" if not invalid_metrics else ("partial" if valid_count else "failed")
        return {**aggregate, "per_question": per_question, "status": status,
                "invalid_metrics": invalid_metrics, "num_requested": len(questions),
                "provider": LLM_PROVIDER, "model": LLM_MODEL}
    except Exception as error:
        message = describe_llm_error(error)
        print(f"  ⚠️ RAGAS evaluation failed: {message}", flush=True)
        return {**fallback, "status": "failed", "error": message}
    finally:
        if evaluator is not None:
            evaluator.close()


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": ("Answer contains claims unsupported by context",
                         "Require source-grounded answers; lower temperature and abstain when evidence is missing",
                         "Output sai → Context đủ? → Kiểm tra grounding ở bước sinh câu trả lời"),
        "context_recall": ("Retrieved context is missing relevant evidence",
                           "Improve chunking, retrieve more candidates or return parent context",
                           "Output sai → Context thiếu? → Kiểm tra chunking và hybrid retrieval"),
        "context_precision": ("Retrieved context contains irrelevant evidence",
                              "Improve reranking and apply source/version metadata filters",
                              "Output sai → Context nhiễu? → Kiểm tra reranking và metadata"),
        "answer_relevancy": ("Answer does not address the question directly",
                             "Clarify the question and require a direct answer in the prompt",
                             "Output sai → Context phù hợp? → Query rõ? → Kiểm tra prompt trả lời"),
    }
    failures = []
    for result in eval_results:
        row = asdict(result) if is_dataclass(result) else dict(result)
        scores = {name: float(row[name]) for name in METRIC_NAMES}
        if row.get("evaluation_errors") or not all(math.isfinite(s) for s in scores.values()):
            # API/parser failures are evaluation errors, not evidence of a bad answer.
            continue
        worst = min(scores, key=scores.get)
        diagnosis, fix, tree = diagnostic_tree[worst]
        if scores[worst] >= 0.70:
            diagnosis = f"Weakest relative metric: {worst}; all metrics meet 0.70, so no low-score failure detected"
            fix = "Review this case only if manual inspection or repeated evaluations reveal an issue"
            tree = "Các metric đều ≥ 0.70 → Kiểm tra thủ công trước khi kết luận có lỗi"
        else:
            diagnosis = f"Possible cause (verify manually): {diagnosis}"
        failures.append({"question": row["question"], "answer": row["answer"],
                         "ground_truth": row["ground_truth"], "contexts": row["contexts"],
                         "worst_metric": worst, "worst_score": scores[worst],
                         "score": sum(scores.values()) / len(scores),
                         "diagnosis": diagnosis, "suggested_fix": fix, "error_tree": tree})
    return sorted(failures, key=lambda failure: failure["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save aggregate and per-question evidence for the bottom-five analysis."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {name: results.get(name, 0.0) for name in METRIC_NAMES},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(row) if is_dataclass(row) else dict(row)
                         for row in results.get("per_question", [])],
        "evaluation": {key: value for key, value in results.items()
                       if key not in METRIC_NAMES and key != "per_question"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
