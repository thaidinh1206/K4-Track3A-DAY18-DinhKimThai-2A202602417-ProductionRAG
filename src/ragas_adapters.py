"""RAGAS 0.1 adapters for provider chat APIs and shared local BGE embeddings."""

import asyncio
from threading import RLock

from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation, LLMResult
from ragas.llms import BaseRagasLLM

from src.m2_search import _load_dense_encoder


class ProviderRagasLLM(BaseRagasLLM):
    """Make separate requests when a metric needs multiple completions."""

    def __init__(self, client, model: str, run_config):
        self.client = client
        self.model = model
        self.set_run_config(run_config)

    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        if n <= 0:
            raise ValueError("n must be positive")
        kwargs = {"model": self.model,
                  "messages": [{"role": "user", "content": prompt.to_string()}],
                  "temperature": 0 if temperature is None else temperature,
                  "max_tokens": 2048}
        if stop:
            kwargs["stop"] = stop
        generations = []
        for _ in range(n):
            # OpenRouter models need not support multiple choices in one request.
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content
            if not content or choice.finish_reason == "length":
                raise ValueError("Evaluator returned empty or truncated output")
            generations.append(Generation(text=content.strip()))
        return LLMResult(generations=[generations], llm_output={"model_name": self.model})

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        # Keep blocking SDK calls off RAGAS's event loop. The metric worker limit
        # bounds concurrent API requests; each metric's n requests run sequentially.
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

    def close(self):
        self.client.close()


class LocalBGEEmbeddings(Embeddings):
    """Reuse M2's cached encoder instead of loading a second model or using an API."""

    def __init__(self):
        self._lock = RLock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            vectors = _load_dense_encoder().encode(
                texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False,
            )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
