from __future__ import annotations

from dataclasses import dataclass, field
import json
import re


_PROMPT_LEAK_PREFIXES = (
    "question:",
    "provided context:",
    "context:",
    "system prompt:",
    "user prompt:",
    "output rules:",
    "rules:",
    "answer exactly the question",
    "use only the provided context",
    "structured evidence:",
    "route:",
    "answer fields:",
    "summary:",
    "results:",
)


@dataclass
class SanitizedAnswer:
    answer: str
    changed: bool
    notes: list[str] = field(default_factory=list)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        return inner.rsplit("\n```", 1)[0].strip()
    return text


def _extract_json_answer(text: str) -> tuple[str | None, str | None]:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return None, None
    try:
        payload = json.loads(stripped)
    except Exception:
        return None, None
    if isinstance(payload, dict):
        for key in ("answer", "final_answer", "response", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), key
    return None, None


def sanitize_answer(answer: str, not_found_text: str) -> SanitizedAnswer:
    original = str(answer or "")
    working = _strip_code_fences(original).strip()
    notes: list[str] = []

    extracted_json_answer, extracted_key = _extract_json_answer(working)
    if extracted_json_answer is not None:
        working = extracted_json_answer
        notes.append(f"extracted_json_field={extracted_key}")

    cleaned_lines: list[str] = []
    skip_next_nonempty_lines = 0
    for raw_line in working.splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if skip_next_nonempty_lines > 0:
            skip_next_nonempty_lines -= 1
            notes.append(f"removed_section_content={stripped[:32]}")
            continue
        if lowered.startswith(_PROMPT_LEAK_PREFIXES):
            notes.append(f"removed_prompt_line={stripped[:32]}")
            if lowered in {"question:", "provided context:", "context:", "system prompt:", "user prompt:"}:
                skip_next_nonempty_lines = 1
            continue
        if lowered.startswith(("you are a", "do not ", "if the answer", "return the configured")):
            notes.append(f"removed_instruction_line={stripped[:32]}")
            continue
        cleaned_lines.append(stripped)

    working = "\n".join(cleaned_lines).strip()
    working = re.sub(r"\n{3,}", "\n\n", working)
    working = re.sub(r"[ \t]{2,}", " ", working)

    if working.startswith("{") and working.endswith("}"):
        notes.append("residual_json_removed")
        working = ""

    if not working:
        working = not_found_text
        notes.append("fallback_not_found")

    return SanitizedAnswer(answer=working, changed=working != original.strip(), notes=notes)
