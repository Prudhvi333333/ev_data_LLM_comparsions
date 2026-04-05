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
    if looks_like_list_or_count:
        configured = min(configured, 384)
    if context and len(context) > 2500:
        configured = min(configured, 320)
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
        if context is not None:
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
            return (
                f"{RAG_SYSTEM_PROMPT}\n"
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
        payload = _build_generation_payload(
            self.model_name,
            prompt,
            self.config,
            max_tokens_override=max_tokens,
            temperature_override=0.0 if context is not None else None,
        )

        try:
            output = await self._invoke_ollama(
                payload,
                timeout_seconds=int(self.config.generation.timeout_seconds),
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
                    timeout_seconds=int(self.config.generation.timeout_seconds),
                )
                if retry_output:
                    output = retry_output

            candidate_companies = self._extract_context_companies(context)
            if (
                context
                and output
                and self._is_list_style_question(question)
                and candidate_companies
                and len(candidate_companies) <= 10
                and self._count_companies_mentioned(output, candidate_companies) < len(candidate_companies)
            ):
                logger.info(
                    "Generation completeness pass triggered | mentioned=%s/%s",
                    self._count_companies_mentioned(output, candidate_companies),
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
                    timeout_seconds=int(self.config.generation.timeout_seconds),
                )
                if completeness_output:
                    output = completeness_output

            return output or "GENERATION_ERROR: empty response from Ollama"
        except httpx.TimeoutException:
            logger.warning("Ollama generation timed out for %s", self.model_name)
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
