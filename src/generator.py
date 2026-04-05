from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import httpx

from src.utils.async_helpers import RetryExhaustedError, retry_async
from src.utils.logger import get_logger
from src.utils.ollama_client import build_ollama_generate_url


logger = get_logger("generator")

try:
    import google.generativeai as genai
except Exception:
    genai = None


RAG_SYSTEM_PROMPT = """You are a precise data analyst for the Georgia EV Automotive Supply Chain knowledge base.
Use ONLY the provided context rows to answer the question.
Do not use outside knowledge and do not invent facts not supported by context.

Instructions:
- Treat each context row as structured evidence with fields such as Company, Tier, EV Role, OEMs, Employment, Products, and EV Relevant.
- Context rows are separated by `---`; inspect every row before finalizing the answer.
- Context may also include precomputed structured summaries (for example county totals or OEM mappings); treat those summaries as valid evidence.
- Use semantic matching across EV Role, Products, EV Relevant, Tier, and OEM fields; exact phrase match is not required.
- You may perform deterministic inference from row fields:
  - grouping/filtering across rows,
  - counting,
  - set overlap (for example OEM overlap),
  - and risk indication language requested by the question, as long as the underlying fields support it.
- If the question asks for a list, enumerate all matching companies from the context (not a sample subset) and include the requested fields.
- For list questions, include an explicit count of matched companies.
- If a question asks for Primary OEM links, include the Primary OEM value exactly as shown for each matched row; use "Not specified" if blank.
- Do not exclude matched rows only because Primary OEM is "Multiple OEMs" or blank.
- If a question asks about locations and facility types, include each matching record and provide a count when multiple records share the same location.
- If the question asks for count/aggregation, compute it explicitly from context and show the resulting entities.
- Keep answers concise and factual; prefer short bullets over long prose.
- Reply exactly "This information is not available in the knowledge base." only when no context rows can support even a partial, evidence-based answer.
- Do not mention that you are an AI model.
- Do not include markdown tables.
"""

NORAG_SYSTEM_PROMPT = """You are an automotive supply chain expert.
Answer the question concisely from your own knowledge.
If unsure, say so directly.
Do not include markdown tables.
"""


def _effective_max_tokens(config: SimpleNamespace, question: str, context: str | None) -> int:
    configured = int(config.generation.max_tokens)
    question_lower = question.lower()
    looks_like_list_or_count = question_lower.startswith(
        (
            "which",
            "identify",
            "list",
            "show",
            "name",
            "how many",
            "count",
        )
    )
    includes_exhaustive_list_intent = any(
        marker in question_lower
        for marker in (
            "show all",
            "list every",
            "map all",
            "full set",
            "all tier",
        )
    )
    if looks_like_list_or_count:
        configured = min(configured, 420)
    if includes_exhaustive_list_intent:
        configured = min(max(384, configured), 420)
    if context and len(context) > 2500 and not includes_exhaustive_list_intent:
        configured = min(configured, 380)
    return max(128, configured)


def _build_generation_payload(
    model_name: str,
    prompt: str,
    config: SimpleNamespace,
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
) -> dict:
    return {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(
                config.generation.temperature if temperature_override is None else temperature_override
            ),
            "num_predict": int(
                config.generation.max_tokens if max_tokens_override is None else max_tokens_override
            ),
            "num_ctx": int(getattr(config.generation, "num_ctx", 4096)),
            "top_p": float(getattr(config.generation, "top_p", 0.9)),
            "repeat_penalty": float(getattr(config.generation, "repeat_penalty", 1.05)),
        },
    }


class OllamaGenerator:
    _COMPANY_REGEX = re.compile(r"Company:\s*(.*?)\s*\|", re.IGNORECASE)

    def __init__(self, model_name: str, config: SimpleNamespace) -> None:
        self.model_name = model_name
        self.config = config
        self.endpoint = build_ollama_generate_url(
            getattr(config.generation, "ollama_endpoint", None)
        )

    @staticmethod
    def _format_context_rows(context: str) -> str:
        raw = context.replace("RELEVANT KNOWLEDGE BASE EXCERPTS:", "").strip()
        chunks = [chunk.strip() for chunk in raw.split("\n---\n") if chunk.strip()]
        if not chunks:
            return context
        numbered = []
        for idx, chunk in enumerate(chunks, start=1):
            numbered.append(f"[ROW {idx}]\n{chunk}")
        return f"TOTAL_ROWS: {len(chunks)}\n" + "\n\n".join(numbered)

    def _build_prompt(self, question: str, context: str | None) -> str:
        question_lower = question.lower()
        if context is not None:
            if context.startswith("STRUCTURED SUPPLIER ROLE-PRODUCT LIST"):
                return (
                    "You are a precise analyst. The context already contains a computed supplier list.\n"
                    "Use only that structured list and return all listed companies with Tier, EV Role, and Product/Service.\n"
                    "Include the explicit total company count.\n"
                    "Do not answer with unavailable when this structured list is present.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "ANSWER:"
                )
            if context.startswith("STRUCTURED INDIRECT EV HIGH EMPLOYMENT LIST"):
                return (
                    "You are a precise analyst. The context already contains a computed list of indirectly EV-relevant companies above an employment threshold.\n"
                    "Use this structured list directly.\n"
                    "Return all listed companies with employment and location, plus total count.\n"
                    "Do not answer with unavailable when this structured list is present.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "ANSWER:"
                )
            if context.startswith("STRUCTURED INNOVATION-STAGE SUPPLIER CANDIDATES"):
                return (
                    "You are a precise analyst. The context already contains innovation-stage supplier candidates derived from product descriptions.\n"
                    "Use that structured evidence directly and list the companies with the matching product phrase.\n"
                    "Include total count and do not answer with unavailable when structured candidates are present.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "ANSWER:"
                )
            if context.startswith("STRUCTURED COUNTY EMPLOYMENT TOTALS"):
                return (
                    "You are a precise analyst. The context already contains computed county totals.\n"
                    "Use those totals directly to answer the question.\n"
                    "Pick the highest county from the list and include the county name and total employment.\n"
                    "Do not answer with unavailable when totals are present.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "ANSWER:"
                )
            if "STRUCTURED VEHICLE ASSEMBLY OEM LIST" in context:
                return (
                    "You are a precise analyst. The context contains a structured OEM list and specific Tier 1 links.\n"
                    "Use that structured evidence directly.\n"
                    "List all OEMs shown and then list specific Tier 1 links.\n"
                    "Do not answer with unavailable when structured lists are present.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "ANSWER:"
                )
            formatted_context = self._format_context_rows(context)
            question_specific_instructions: list[str] = []
            if (
                "primary oem" in question_lower
                and any(marker in question_lower for marker in ("map all", "linked", "connection", "show"))
            ):
                question_specific_instructions.append(
                    "- Include all matched companies; for each, show Primary OEMs exactly as provided in context."
                )
                question_specific_instructions.append(
                    "- If Primary OEM is blank, write 'Not specified' instead of excluding the company."
                )
            if "location" in question_lower and "facility" in question_lower:
                question_specific_instructions.append(
                    "- Include location and facility for every matched record."
                )
                question_specific_instructions.append(
                    "- If multiple records share the same location/facility, report that count explicitly."
                )
            extra_instruction_block = (
                "QUESTION-SPECIFIC INSTRUCTIONS:\n"
                + "\n".join(question_specific_instructions)
                + "\n\n"
            ) if question_specific_instructions else ""
            return (
                f"{RAG_SYSTEM_PROMPT}\n"
                f"{extra_instruction_block}"
                f"CONTEXT ROWS:\n{formatted_context}\n\n"
                f"QUESTION:\n{question}\n\n"
                "ANSWER:"
            )
        return f"{NORAG_SYSTEM_PROMPT}\nQUESTION:\n{question}\n\nANSWER:"

    @staticmethod
    def _is_unavailable_answer(answer: str) -> bool:
        lowered = answer.strip().lower()
        unavailable_markers = (
            "not available in the knowledge base",
            "information is not available in the knowledge base",
            "context does not specify",
            "cannot be determined from the context",
            "cannot determine from the context",
            "insufficient information in the context",
        )
        return any(marker in lowered for marker in unavailable_markers)

    @staticmethod
    def _is_list_style_question(question: str) -> bool:
        lowered = question.strip().lower()
        return lowered.startswith(
            (
                "which",
                "identify",
                "list",
                "show",
                "name",
                "what locations",
                "what location",
            )
        )

    def _extract_context_companies(self, context: str | None) -> list[str]:
        if not context:
            return []
        deduped: list[str] = []
        for match in self._COMPANY_REGEX.findall(context):
            company = " ".join(match.split()).strip()
            if company and company not in deduped:
                deduped.append(company)
        return deduped

    @staticmethod
    def _count_companies_mentioned(answer: str, companies: list[str]) -> int:
        lowered = answer.lower()
        return sum(1 for company in companies if company.lower() in lowered)

    @staticmethod
    def _count_companies_in_primary_list(answer: str, companies: list[str]) -> int:
        list_like_lines = []
        for raw_line in (answer or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ")):
                list_like_lines.append(line[2:].strip().lower())
            elif re.match(r"^\d+\.\s+", line):
                list_like_lines.append(re.sub(r"^\d+\.\s+", "", line).strip().lower())
        if not list_like_lines:
            return 0
        joined = "\n".join(list_like_lines)
        return sum(1 for company in companies if company.lower() in joined)

    @staticmethod
    def _extract_top_county_total(context: str) -> tuple[str, str] | None:
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        for line in lines:
            if not line.startswith("- "):
                continue
            if ":" not in line:
                continue
            county_part, total_part = line[2:].split(":", 1)
            county = county_part.strip()
            total = total_part.strip()
            if county and total:
                return county, total
        return None

    @staticmethod
    def _extract_structured_lines(context: str, header: str, stop_headers: tuple[str, ...]) -> list[str]:
        lines = [line.rstrip() for line in context.splitlines()]
        capture = False
        captured: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped == header:
                capture = True
                continue
            if capture and any(stripped.startswith(stop) for stop in stop_headers):
                break
            if capture and stripped.startswith("- "):
                captured.append(stripped[2:].strip())
        return captured

    @staticmethod
    def _count_mentions_case_insensitive(answer: str, entries: list[str]) -> int:
        lowered = (answer or "").lower()
        return sum(1 for entry in entries if entry and entry.lower() in lowered)

    def _build_recovery_prompt(self, question: str, context: str, previous_answer: str) -> str:
        formatted_context = self._format_context_rows(context)
        return (
            f"{RAG_SYSTEM_PROMPT}\n"
            "REVIEW INSTRUCTIONS:\n"
            "- The prior draft incorrectly concluded unavailable.\n"
            "- Re-read the context and extract all candidate evidence that partially or fully satisfies the query constraints.\n"
            "- Structured summaries in context (if present) are valid evidence and should be used directly.\n"
            "- If this is a list-style question, return the complete matched set with count, not a partial subset.\n"
            "- Use evidence-first reasoning from row fields; if inference is needed, state it briefly as 'inferred from context fields'.\n"
            "- Only output unavailable if there are truly zero relevant rows.\n\n"
            f"CONTEXT ROWS:\n{formatted_context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"PRIOR DRAFT:\n{previous_answer}\n\n"
            "ANSWER:"
        )

    def _build_completeness_prompt(
        self,
        question: str,
        context: str,
        previous_answer: str,
        candidate_companies: list[str],
    ) -> str:
        candidates = "\n".join(f"- {company}" for company in candidate_companies)
        formatted_context = self._format_context_rows(context)
        return (
            f"{RAG_SYSTEM_PROMPT}\n"
            "COMPLETENESS REVIEW:\n"
            "- Candidate companies were extracted from context rows.\n"
            "- Evaluate each candidate against all question constraints and include every candidate that matches.\n"
            "- Do not return a partial subset if additional candidates also satisfy the constraints.\n"
            "- Do not put matched candidates under exclusion notes; include matched candidates in the primary result list.\n"
            "- If none match, output the unavailable sentence exactly.\n\n"
            f"CANDIDATE COMPANIES:\n{candidates}\n\n"
            f"CONTEXT ROWS:\n{formatted_context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"PRIOR DRAFT:\n{previous_answer}\n\n"
            "ANSWER:"
        )

    async def _invoke_ollama(self, payload: dict, timeout_seconds: int) -> str:
        timeout = httpx.Timeout(
            connect=min(30.0, float(timeout_seconds)),
            read=float(timeout_seconds),
            write=min(30.0, float(timeout_seconds)),
            pool=min(30.0, float(timeout_seconds)),
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data.get("response", "")).strip()

    async def generate(self, question: str, context: str | None) -> str:
        prompt = self._build_prompt(question, context)
        max_tokens = _effective_max_tokens(self.config, question, context)
        if context and context.startswith("STRUCTURED SUPPLIER ROLE-PRODUCT LIST"):
            max_tokens = min(max_tokens, 320)
        payload = _build_generation_payload(
            self.model_name,
            prompt,
            self.config,
            max_tokens_override=max_tokens,
            temperature_override=0.0 if context is not None else None,
        )
        request_timeout = int(self.config.generation.timeout_seconds)
        if context and context.startswith("STRUCTURED SUPPLIER ROLE-PRODUCT LIST"):
            request_timeout = max(request_timeout, 240)
        if context and (
            context.startswith("STRUCTURED INDIRECT EV HIGH EMPLOYMENT LIST")
            or context.startswith("STRUCTURED INNOVATION-STAGE SUPPLIER CANDIDATES")
        ):
            request_timeout = max(request_timeout, 180)

        try:
            output = await self._invoke_ollama(
                payload,
                timeout_seconds=request_timeout,
            )
            if (
                context
                and output
                and self._is_unavailable_answer(output)
            ):
                logger.info("Generation recovery triggered for unavailable draft")
                recovery_prompt = self._build_recovery_prompt(question, context, output)
                recovery_payload = _build_generation_payload(
                    self.model_name,
                    recovery_prompt,
                    self.config,
                    max_tokens_override=max_tokens,
                    temperature_override=0.0,
                )
                retry_output = await self._invoke_ollama(
                    recovery_payload,
                    timeout_seconds=request_timeout,
                )
                if retry_output:
                    output = retry_output

            candidate_companies = self._extract_context_companies(context)
            if (
                context
                and output
                and self._is_list_style_question(question)
                and candidate_companies
                and len(candidate_companies) <= 30
                and self._count_companies_in_primary_list(output, candidate_companies) < len(candidate_companies)
            ):
                logger.info(
                    "Generation completeness pass triggered | listed=%s/%s",
                    self._count_companies_in_primary_list(output, candidate_companies),
                    len(candidate_companies),
                )
                completeness_prompt = self._build_completeness_prompt(
                    question,
                    context,
                    output,
                    candidate_companies,
                )
                completeness_payload = _build_generation_payload(
                    self.model_name,
                    completeness_prompt,
                    self.config,
                    max_tokens_override=max_tokens,
                    temperature_override=0.0,
                )
                completeness_output = await self._invoke_ollama(
                    completeness_payload,
                    timeout_seconds=request_timeout,
                )
                if completeness_output:
                    output = completeness_output

            if context and context.startswith("STRUCTURED COUNTY EMPLOYMENT TOTALS"):
                lowered_output = (output or "").lower()
                needs_structured_guard = (
                    self._is_unavailable_answer(output or "")
                    or "please clarify" in lowered_output
                    or "if you meant" in lowered_output
                    or "however" in lowered_output
                )
                if needs_structured_guard:
                    top_county = self._extract_top_county_total(context)
                    if top_county is not None:
                        county, total = top_county
                        top_entries = self._extract_structured_lines(
                            context,
                            "STRUCTURED COUNTY EMPLOYMENT TOTALS (computed from retrieved rows):",
                            stop_headers=("STRUCTURED ", "RELEVANT KNOWLEDGE BASE EXCERPTS"),
                        )[:5]
                        output_lines = [
                            f"{county} has the highest total employment with {total} employees."
                        ]
                        if top_entries:
                            output_lines.append("Top counties by total employment:")
                            output_lines.extend(f"- {entry}" for entry in top_entries)
                        output = "\n".join(output_lines)

            if "STRUCTURED VEHICLE ASSEMBLY OEM LIST" in (context or ""):
                oem_entries = self._extract_structured_lines(
                    context or "",
                    "STRUCTURED VEHICLE ASSEMBLY OEM LIST (computed from retrieved rows):",
                    stop_headers=("STRUCTURED SPECIFIC TIER 1 LINKS", "STRUCTURED ", "RELEVANT KNOWLEDGE BASE EXCERPTS"),
                )
                link_entries = self._extract_structured_lines(
                    context or "",
                    "STRUCTURED SPECIFIC TIER 1 LINKS (Tier 1 only, excluding 'Multiple OEMs'):",
                    stop_headers=("STRUCTURED ", "RELEVANT KNOWLEDGE BASE EXCERPTS"),
                )
                missing_oems = (
                    bool(oem_entries)
                    and self._count_mentions_case_insensitive(output or "", oem_entries) < len(oem_entries)
                )
                if self._is_unavailable_answer(output or "") or missing_oems:
                    lines = ["Vehicle Assembly OEMs in Georgia:"]
                    lines.extend(f"- {entry}" for entry in oem_entries)
                    lines.append("")
                    lines.append("Specific Tier 1 links:")
                    if link_entries:
                        lines.extend(f"- {entry}" for entry in link_entries)
                    else:
                        lines.append("- No specific Tier 1 links found")
                    output = "\n".join(lines)

            if (context or "").startswith("STRUCTURED INDIRECT EV HIGH EMPLOYMENT LIST"):
                entries = self._extract_structured_lines(
                    context or "",
                    "STRUCTURED INDIRECT EV HIGH EMPLOYMENT LIST (computed from retrieved rows):",
                    stop_headers=("STRUCTURED ", "RELEVANT KNOWLEDGE BASE EXCERPTS"),
                )
                missing_entries = (
                    bool(entries)
                    and self._count_mentions_case_insensitive(output or "", entries) < max(1, len(entries) // 2)
                )
                if self._is_unavailable_answer(output or "") or missing_entries:
                    lines = [
                        "Indirectly EV-relevant companies above the employment threshold:",
                    ]
                    lines.extend(f"- {entry}" for entry in entries)
                    lines.append(f"Total companies: {len(entries)}")
                    output = "\n".join(lines)

            if (context or "").startswith("STRUCTURED INNOVATION-STAGE SUPPLIER CANDIDATES"):
                entries = self._extract_structured_lines(
                    context or "",
                    "STRUCTURED INNOVATION-STAGE SUPPLIER CANDIDATES (computed from retrieved rows):",
                    stop_headers=("STRUCTURED ", "RELEVANT KNOWLEDGE BASE EXCERPTS"),
                )
                missing_entries = (
                    bool(entries)
                    and self._count_mentions_case_insensitive(output or "", entries) < max(1, len(entries) // 2)
                )
                if self._is_unavailable_answer(output or "") or missing_entries:
                    lines = [
                        "Innovation-stage supplier candidates in Georgia:",
                    ]
                    lines.extend(f"- {entry}" for entry in entries)
                    lines.append(f"Total companies: {len(entries)}")
                    output = "\n".join(lines)

            return output or "GENERATION_ERROR: empty response from Ollama"
        except httpx.TimeoutException:
            logger.warning("Ollama generation timed out for %s; retrying with smaller output budget.", self.model_name)
            retry_payload = _build_generation_payload(
                self.model_name,
                prompt,
                self.config,
                max_tokens_override=min(160, max_tokens),
                temperature_override=0.0 if context is not None else None,
            )
            try:
                retry_output = await self._invoke_ollama(
                    retry_payload,
                    timeout_seconds=max(60, int(float(request_timeout) * 0.6)),
                )
                return retry_output or "GENERATION_TIMEOUT"
            except httpx.TimeoutException:
                logger.warning("Ollama generation retry also timed out for %s", self.model_name)
                return "GENERATION_TIMEOUT"
        except httpx.HTTPError as exc:
            logger.warning("Ollama generation failed for %s: %s", self.model_name, exc)
            return f"GENERATION_ERROR: {exc}"


class GeminiGenerator:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.model_name = config.models.gemini
        self.api_key = str(config.api_keys.gemini)
        self._local_only = genai is None or self.api_key.startswith("local-dev")

        if not self._local_only and genai is not None:
            genai.configure(api_key=self.api_key)

    def _build_prompt(self, question: str, context: str | None) -> str:
        if context is not None:
            return (
                f"{RAG_SYSTEM_PROMPT}\n"
                f"CONTEXT ROWS:\n{context}\n\n"
                f"QUESTION:\n{question}\n\n"
                "ANSWER:"
            )
        return f"{NORAG_SYSTEM_PROMPT}\nQUESTION:\n{question}\n\nANSWER:"

    @retry_async(max_retries=3, backoff_base=2.0)
    async def _invoke_gemini(self, prompt: str) -> str:
        if genai is None:
            raise httpx.HTTPError("google.generativeai is unavailable")

        model = genai.GenerativeModel(self.model_name)

        def _generate_sync():
            return model.generate_content(
                prompt,
                generation_config={
                    "temperature": float(self.config.generation.temperature),
                    "max_output_tokens": int(self.config.generation.max_tokens),
                    "top_p": float(getattr(self.config.generation, "top_p", 0.9)),
                },
            )

        try:
            response = await asyncio.to_thread(_generate_sync)
            return str(getattr(response, "text", "") or "").strip()
        except Exception as exc:
            error_text = str(exc)
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                await asyncio.sleep(60)
                raise httpx.HTTPError(error_text) from exc
            raise

    async def generate(self, question: str, context: str | None) -> str:
        if self._local_only:
            return "GENERATION_ERROR: Gemini is unavailable because no API key/package is configured."

        prompt = self._build_prompt(question, context)

        try:
            output = await asyncio.wait_for(
                self._invoke_gemini(prompt),
                timeout=float(self.config.generation.timeout_seconds),
            )
            return output or "GENERATION_ERROR: empty response from Gemini"
        except asyncio.TimeoutError:
            return "GENERATION_TIMEOUT"
        except (RetryExhaustedError, httpx.HTTPError, Exception) as exc:
            logger.warning("Gemini generation failed: %s", exc)
            return f"GENERATION_ERROR: {exc}"


def get_generator(model_key: str, config: SimpleNamespace) -> OllamaGenerator | GeminiGenerator:
    normalized = model_key.strip().lower()
    if normalized == "qwen":
        return OllamaGenerator(config.models.qwen, config)
    if normalized == "gemma":
        return OllamaGenerator(config.models.gemma, config)
    if normalized == "gemini":
        return GeminiGenerator(config)
    raise ValueError(f"Unsupported model key: {model_key}")
