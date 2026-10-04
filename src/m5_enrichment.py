from __future__ import annotations

"""M5: summarization, HyQA, contextual prepend and metadata enrichment."""

import json
import os
import re
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_API_KEY, LLM_MODEL, create_llm_client, describe_llm_error


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _request(system: str, text: str, max_tokens: int, json_mode: bool = False) -> str | None:
    """Use the configured provider; callers supply deterministic fallbacks."""
    if not LLM_API_KEY:
        return None
    try:
        kwargs = {"model": LLM_MODEL, "temperature": 0, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": text}]}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with create_llm_client() as client:
            response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content
        if not content or choice.finish_reason == "length":
            raise ValueError("Enrichment returned empty or truncated output")
        return content.strip()
    except Exception as error:
        print(f"  ⚠️ Enrichment fallback: {describe_llm_error(error)}", flush=True)
        return None


def _parse_object(content: str) -> dict:
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def _fallback_summary(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    summary = " ".join(sentences[:2])
    if len(summary) > 300:
        summary = summary[:300].rsplit(" ", 1)[0]
    return summary


def _fallback_questions(text: str, count: int) -> list[str]:
    sentences = [s.strip().lstrip("#- ") for s in
                 re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 10]
    return [f"Quy định về ‘{s.rstrip('.!?')[:160]}’ là gì?" for s in sentences[:count]]


def _fallback_metadata(text: str) -> dict:
    heading = re.search(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
    topic = heading.group(1).strip() if heading else (_fallback_summary(text)[:100] or "general")
    lower = text.casefold()
    category = "policy"
    for name, keywords in (("it", ("vpn", "mật khẩu", "bảo mật", "mfa")),
                           ("finance", ("lương", "chi phí", "tạm ứng", "thanh toán")),
                           ("hr", ("nghỉ phép", "thử việc", "nhân viên"))):
        if any(keyword in lower for keyword in keywords):
            category = name
            break
    result = {"topic": topic, "entities": [], "category": category, "language": "vi"}
    for key, pattern in (("version", r"Phiên bản:\s*([^|\n]+)"),
                         ("effective_date", r"Ngày hiệu lực:\s*(\d{2}/\d{2}/\d{4})")):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()
    return result


def _normalize_metadata(value: dict, text: str) -> dict:
    result = _fallback_metadata(text)
    # Never let generated metadata overwrite source, parent_id or chunk_index.
    for key in ("topic", "language", "version", "effective_date"):
        if isinstance(value.get(key), str) and value[key].strip():
            result[key] = value[key].strip()
    if isinstance(value.get("entities"), list):
        result["entities"] = [e.strip() for e in value["entities"] if isinstance(e, str) and e.strip()]
    if value.get("category") in {"policy", "hr", "it", "finance", "training", "admin", "safety", "compliance"}:
        result["category"] = value["category"]
    return result


def summarize_chunk(text: str) -> str:
    if not text.strip():
        return ""
    content = _request(
        "Tóm tắt trong tối đa 2 câu ngắn bằng tiếng Việt, không dài hơn đoạn gốc. "
        "Giữ nguyên số liệu, điều kiện và phủ định. Chỉ dựa trên đoạn văn; không thêm thông tin.",
        text, 150,
    )
    return content if content and len(content) <= len(text) * 2 else _fallback_summary(text)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    if n_questions <= 0 or not text.strip():
        return []
    content = _request(
        f"Tạo tối đa {n_questions} câu hỏi tiếng Việt mà đoạn văn trả lời được. "
        "Mỗi câu một dòng, kết thúc bằng dấu ?, không trả lời, không thêm thông tin.", text, 250,
    )
    if content:
        questions = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                     for line in content.splitlines() if line.strip()]
        questions = [q if q.endswith("?") else q + "?" for q in questions]
        if questions:
            return questions[:n_questions]
    return _fallback_questions(text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    if not text.strip():
        return text
    context = _request(
        "Viết đúng một câu tiếng Việt mô tả nguồn tài liệu và chủ đề đoạn trích. "
        "Không suy đoán vị trí, không thêm dữ kiện hoặc sửa số liệu.",
        f"Tài liệu: {document_title}\n\nĐoạn trích:\n{text}", 100,
    )
    context = context or f"Trích từ tài liệu {document_title or 'quy định nội bộ'}."
    return f"{context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    if not text.strip():
        return _fallback_metadata(text)
    content = _request(
        'Trả về JSON metadata: {"topic":"...","entities":[],"category":"policy|hr|it|finance",'
        '"language":"vi|en"}. Có thể thêm version và effective_date nếu ghi rõ trong đoạn văn. '
        'Không suy đoán ngày hoặc phiên bản. Không trả về source hoặc parent_id.', text, 200, True,
    )
    if content:
        try:
            return _normalize_metadata(_parse_object(content), text)
        except (ValueError, TypeError) as error:
            print(f"  ⚠️ Metadata fallback: {describe_llm_error(error)}", flush=True)
    return _fallback_metadata(text)


def _enrich_single_call(text: str, source: str) -> dict:
    """Produce all four enrichment fields in one API call, or fall back locally."""
    fallback = {"summary": _fallback_summary(text), "questions": _fallback_questions(text, 3),
                "context": f"Trích từ tài liệu {source or 'quy định nội bộ'}.",
                "metadata": _fallback_metadata(text), "status": "fallback"}
    if not text.strip():
        return {**fallback, "context": ""}
    content = _request(
        'Phân tích đoạn trích, trả về JSON với 4 trường: '
        '{"summary":"tóm tắt tối đa 2 câu", "questions":["3 câu hỏi kết thúc bằng ?"], '
        '"context":"1 câu mô tả nguồn và chủ đề", '
        '"metadata":{"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi"}}. '
        'Chỉ dùng thông tin có trong đoạn trích. Giữ số liệu, điều kiện và phủ định. '
        'Không suy đoán phiên bản, không tạo source/parent_id/chunk_index.',
        f"Tài liệu: {source}\n\nĐoạn trích:\n{text}", 600, True,
    )
    if content:
        try:
            result = _parse_object(content)
            if any(not isinstance(result.get(key), str) or not result[key].strip()
                   for key in ("summary", "context")):
                raise ValueError("summary and context must be non-empty strings")
            if (not isinstance(result.get("metadata"), dict)
                    or not isinstance(result.get("questions"), list)
                    or not result["questions"]
                    or any(not isinstance(q, str) or not q.strip() for q in result["questions"])):
                raise ValueError("Invalid enrichment metadata or questions")
            return {"summary": result["summary"].strip(),
                    "questions": [q.strip() if q.strip().endswith("?") else q.strip() + "?"
                                  for q in result["questions"][:3]],
                    "context": result["context"].strip(),
                    "metadata": _normalize_metadata(result["metadata"], text), "status": "llm"}
        except (ValueError, TypeError) as error:
            print(f"  ⚠️ Combined enrichment fallback: {describe_llm_error(error)}", flush=True)
    return fallback


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Default combined mode makes one API call per non-empty chunk."""
    methods = ["combined"] if methods is None else list(methods)
    if set(methods) - {"combined", "summary", "hyqa", "contextual", "metadata"}:
        raise ValueError("Unknown enrichment method")
    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        metadata = dict(chunk.get("metadata", {}))
        source = metadata.get("source", "")
        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary, questions = result["summary"], result["questions"]
            context, auto_meta = result["context"], result["metadata"]
            auto_meta = {**auto_meta, "enrichment_status": result["status"]}
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            # contextual_prepend always keeps the entire original string.
            contextual = contextual_prepend(text, source) if "contextual" in methods else text
            context = contextual[:-len(text)].rstrip() if text and contextual != text else ""
            auto_meta = extract_metadata(text) if "metadata" in methods else {}
        prefixes = [context] if context else []
        if summary:
            prefixes.append(f"Tóm tắt: {summary}")
        if questions:
            prefixes.append("Câu hỏi liên quan:\n" + "\n".join(questions))
        # Index the generated retrieval cues together with untouched original text.
        enriched_text = "\n\n".join([*prefixes, text]) if prefixes else text
        enriched.append(EnrichedChunk(text, enriched_text, summary, questions,
                                      {**auto_meta, **metadata}, "+".join(methods) or "none"))
        if (i + 1) % 10 == 0 or i + 1 == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
    return enriched


if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 15 ngày làm việc mỗi năm."
    result = enrich_chunks([{"text": sample, "metadata": {"source": "nghi_phep_nam_v2024.md"}}])
    print(result[0].enriched_text)
