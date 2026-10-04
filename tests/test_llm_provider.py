"""Provider configuration and generation routing without external API calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import dotenv
import httpx
import pytest
from openai import OpenAI

from src.m2_search import SearchResult


def load_config(monkeypatch, **values):
    # Do not read a developer's real credentials into isolated configuration tests.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    path = Path(__file__).resolve().parents[1] / "config.py"
    spec = importlib.util.spec_from_file_location("isolated_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("provider,key_name,base_url,model", [
    ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
])
def test_provider_uses_its_own_key_and_endpoint(monkeypatch, provider, key_name, base_url, model):
    cfg = load_config(monkeypatch, LLM_PROVIDER=provider,
                      OPENROUTER_API_KEY="router-test-key", OPENAI_API_KEY="openai-test-key")
    assert cfg.LLM_API_KEY == getattr(cfg, key_name)
    assert cfg.LLM_BASE_URL == base_url
    assert cfg.LLM_MODEL == model
    client = cfg.create_llm_client()
    try:
        assert str(client.base_url).rstrip("/") == base_url
        assert client.api_key == cfg.LLM_API_KEY
    finally:
        client.close()


def test_router_key_selects_openrouter_automatically(monkeypatch):
    cfg = load_config(monkeypatch, OPENROUTER_API_KEY="router-test-key")
    assert cfg.LLM_PROVIDER == "openrouter"
    assert cfg.LLM_MODEL == "openai/gpt-4o-mini"


def test_custom_model_is_preserved(monkeypatch):
    cfg = load_config(monkeypatch, LLM_PROVIDER="openrouter", LLM_MODEL="vendor/custom-model")
    assert cfg.LLM_MODEL == "vendor/custom-model"


@pytest.mark.parametrize("key", ["", "sk-or-v1-...", "YOUR_KEY_HERE"])
def test_missing_router_key_never_uses_openai_key(monkeypatch, key):
    cfg = load_config(monkeypatch, LLM_PROVIDER="openrouter", OPENROUTER_API_KEY=key,
                      OPENAI_API_KEY="other-provider-test-key")
    assert cfg.LLM_API_KEY == ""
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        cfg.create_llm_client()


def test_invalid_provider_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_config(monkeypatch, LLM_PROVIDER="invalid")


@pytest.fixture
def router_client():
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.host == "openrouter.ai"
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer router-test-key"
        assert json.loads(request.content)["model"] == "openai/gpt-4o-mini"
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": "openai/gpt-4o-mini", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Được nghỉ 15 ngày."},
            }],
        })

    client = OpenAI(api_key="router-test-key", base_url="https://openrouter.ai/api/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    yield client, requests
    client.close()


def configure_generation(monkeypatch, module, client):
    monkeypatch.setattr(module, "LLM_API_KEY", "router-test-key")
    monkeypatch.setattr(module, "LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setattr(module, "create_llm_client", lambda: client)


def test_production_generation_routes_to_openrouter(monkeypatch, router_client):
    from src import pipeline

    client, requests = router_client
    configure_generation(monkeypatch, pipeline, client)
    result = SearchResult("Phép năm là 15 ngày.", 1.0, {}, "hybrid")
    search = SimpleNamespace(search=lambda query: [result])
    reranker = SimpleNamespace(rerank=lambda *args, **kwargs: [])
    answer, contexts = pipeline.run_query("Bao nhiêu ngày phép?", search, reranker)
    assert answer == "Được nghỉ 15 ngày."
    assert contexts == [result.text]
    assert len(requests) == 1


def test_baseline_generation_uses_same_provider(monkeypatch, router_client):
    import naive_baseline

    client, requests = router_client
    configure_generation(monkeypatch, naive_baseline, client)
    result = SearchResult("Phép năm là 15 ngày.", 1.0, {}, "dense")
    search = SimpleNamespace(index=lambda *args, **kwargs: None,
                             search=lambda *args, **kwargs: [result])
    monkeypatch.setattr(naive_baseline, "DenseSearch", lambda: search)
    monkeypatch.setattr(naive_baseline, "load_documents", lambda: [])
    monkeypatch.setattr(naive_baseline, "load_test_set", lambda: [
        {"question": "Bao nhiêu ngày phép?", "ground_truth": "15 ngày"},
    ])

    def evaluate(questions, answers, contexts, ground_truths):
        assert answers == ["Được nghỉ 15 ngày."]
        return {"per_question": []}

    monkeypatch.setattr(naive_baseline, "evaluate_ragas", evaluate)
    monkeypatch.setattr(naive_baseline, "save_report", lambda *args, **kwargs: None)
    naive_baseline.main()
    assert len(requests) == 1
