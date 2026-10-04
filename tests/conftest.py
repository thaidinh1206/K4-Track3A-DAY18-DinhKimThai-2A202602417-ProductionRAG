"""Unit tests stay offline even when a developer has configured a real API key."""

import pytest


@pytest.fixture(autouse=True)
def offline_llm_modules(monkeypatch):
    from src import m4_eval, m5_enrichment

    monkeypatch.setattr(m4_eval, "LLM_API_KEY", "")
    monkeypatch.setattr(m5_enrichment, "LLM_API_KEY", "")
