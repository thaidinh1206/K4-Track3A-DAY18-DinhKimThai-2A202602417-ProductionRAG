"""Verify one real enrichment call and one RAGAS question with the configured key."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import LLM_API_KEY, LLM_PROVIDER, LLM_MODEL
from src.m4_eval import evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks


def main():
    if not LLM_API_KEY:
        raise SystemExit("No API key configured in .env")
    print(f"Provider: {LLM_PROVIDER}; model: {LLM_MODEL}", flush=True)
    text = "Nhân viên chính thức được nghỉ phép năm 15 ngày làm việc mỗi năm."
    enriched = enrich_chunks([{"text": text, "metadata": {"source": "nghi_phep_nam_v2024.md"}}])
    status = enriched[0].auto_metadata.get("enrichment_status")
    print(f"M5 status: {status}", flush=True)
    if status != "llm":
        raise SystemExit("M5 used fallback; inspect the provider error above")
    results = evaluate_ragas(
        ["Nhân viên chính thức có bao nhiêu ngày phép năm?"],
        ["Nhân viên chính thức có 15 ngày phép năm."], [[text]], ["15 ngày làm việc mỗi năm."],
    )
    report = Path(__file__).resolve().parents[1] / "reports" / "m4_m5_smoke_report.json"
    save_report(results, failure_analysis(results["per_question"], bottom_n=1), str(report))
    print(f"M4 status: {results['status']}", flush=True)
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        print(f"{metric}: {results[metric]:.4f}", flush=True)
    if results["status"] != "ok":
        raise SystemExit("M4 did not complete all metrics; inspect the smoke report")


if __name__ == "__main__":
    main()
