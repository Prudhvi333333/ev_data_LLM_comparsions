from __future__ import annotations

from src.answer_sanitizer import sanitize_answer


def test_answer_sanitizer_extracts_json_answer() -> None:
    result = sanitize_answer('{"answer": "Troup County has 2,435 employees."}', "Not found in provided context.")
    assert result.answer == "Troup County has 2,435 employees."
    assert result.changed is True
    assert any(note.startswith("extracted_json_field=") for note in result.notes)


def test_answer_sanitizer_removes_prompt_leakage_lines() -> None:
    raw = """
    QUESTION:
    Which county has the highest employment?
    PROVIDED CONTEXT:
    ignored
    Troup County has the highest total employment among Tier 1 suppliers with 2,435 employees.
    """.strip()
    result = sanitize_answer(raw, "Not found in provided context.")
    assert result.answer == "Troup County has the highest total employment among Tier 1 suppliers with 2,435 employees."
    assert result.changed is True
