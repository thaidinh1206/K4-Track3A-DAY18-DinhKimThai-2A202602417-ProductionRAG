"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER", "openrouter" if OPENROUTER_API_KEY.strip() else "openai"
).strip().lower()
if LLM_PROVIDER == "openrouter":
    LLM_API_KEY = OPENROUTER_API_KEY.strip()
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
elif LLM_PROVIDER == "openai":
    LLM_API_KEY = OPENAI_API_KEY.strip()
    LLM_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_LLM_MODEL = "gpt-4o-mini"
else:
    raise ValueError("LLM_PROVIDER must be 'openrouter' or 'openai'")
if LLM_API_KEY in {"sk-...", "sk-or-v1-...", "YOUR_KEY_HERE"}:
    LLM_API_KEY = ""
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL


def create_llm_client():
    """Create a client with explicit credentials and the selected provider URL."""
    if not LLM_API_KEY:
        key_name = "OPENROUTER_API_KEY" if LLM_PROVIDER == "openrouter" else "OPENAI_API_KEY"
        raise ValueError(f"Set {key_name} in .env before calling the LLM")
    from openai import OpenAI

    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                  timeout=60.0, max_retries=2)


def describe_llm_error(error: Exception) -> str:
    """Describe failures without exposing configured API credentials."""
    message = str(error)
    for key in (OPENAI_API_KEY, OPENROUTER_API_KEY, LLM_API_KEY):
        if key:
            message = message.replace(key, "[REDACTED]")
    return f"{type(error).__name__}: {message[:400]}"

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
