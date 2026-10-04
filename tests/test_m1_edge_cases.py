"""Chunking boundaries and deterministic semantic checks without model downloads."""

import numpy as np
import pytest

from src import m1_chunking as m1


def test_empty_documents_do_not_load_encoder(monkeypatch):
    def unexpected_load():
        raise AssertionError("Empty text must not load a model")

    monkeypatch.setattr(m1, "_get_semantic_encoder", unexpected_load)
    assert m1.chunk_semantic(" \n\n ") == []
    assert m1.chunk_hierarchical(" \n\n ") == ([], [])
    assert m1.chunk_structure_aware(" \n\n ") == []


def test_semantic_splits_at_topic_change(monkeypatch):
    class Encoder:
        def encode(self, sentences):
            assert len(sentences) == 4
            return np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])

    monkeypatch.setattr(m1, "_get_semantic_encoder", lambda: Encoder())
    metadata = {"source": "policy.md"}
    chunks = m1.chunk_semantic("Leave policy. More leave. VPN policy. More VPN.",
                               threshold=0.5, metadata=metadata)
    assert [c.text for c in chunks] == ["Leave policy.\nMore leave.",
                                      "VPN policy.\nMore VPN."]
    assert all(c.metadata["source"] == "policy.md" for c in chunks)
    assert metadata == {"source": "policy.md"}


def test_hierarchy_bounds_and_preserves_long_paragraph():
    text = " ".join(f"word{i}" for i in range(100)) + "\n\n" + "x" * 150
    parents, children = m1.chunk_hierarchical(text, parent_size=100, child_size=30)
    assert all(0 < len(p.text) <= 100 for p in parents)
    assert all(0 < len(c.text) <= 30 for c in children)
    # Removing whitespace also accounts for a hard split in the long unbroken word.
    compact = lambda value: "".join(value.split())
    assert compact("".join(p.text for p in parents)) == compact(text)
    assert compact("".join(c.text for c in children)) == compact(text)
    for parent in parents:
        linked = [c for c in children if c.parent_id == parent.metadata["parent_id"]]
        assert linked
        assert compact("".join(c.text for c in linked)) == compact(parent.text)


def test_parent_ids_distinguish_sources_and_are_repeatable():
    text = "A paragraph about policy. " * 10
    first, _ = m1.chunk_hierarchical(text, metadata={"source": "a.md"})
    second, _ = m1.chunk_hierarchical(text, metadata={"source": "b.md"})
    repeat, _ = m1.chunk_hierarchical(text, metadata={"source": "a.md"})
    assert {p.metadata["parent_id"] for p in first}.isdisjoint(
        p.metadata["parent_id"] for p in second)
    assert [p.metadata["parent_id"] for p in first] == [
        p.metadata["parent_id"] for p in repeat]


@pytest.mark.parametrize("parent_size,child_size", [(0, 10), (10, 0), (10, 10), (10, 20)])
def test_invalid_hierarchy_sizes(parent_size, child_size):
    with pytest.raises(ValueError):
        m1.chunk_hierarchical("text", parent_size, child_size)


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_structure_preserves_tables_lists_and_fenced_headers(fence):
    text = (f"Preamble\n\n# Policy\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n"
            f"- item one\n- item two\n\n{fence}text\n# Inside code\n{fence}\n\n"
            "#### Detail\nDetails here.")
    metadata = {"source": "policy.md"}
    chunks = m1.chunk_structure_aware(text, metadata)
    assert len(chunks) == 3
    assert [c.metadata["section"] for c in chunks] == ["", "# Policy", "#### Detail"]
    assert "| A | B |\n|---|---|\n| 1 | 2 |" in chunks[1].text
    assert "- item one\n- item two" in chunks[1].text
    assert f"{fence}text\n# Inside code\n{fence}" in chunks[1].text
    assert metadata == {"source": "policy.md"}
