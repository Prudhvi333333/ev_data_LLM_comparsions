from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any

import httpx

from src.utils.logger import get_logger
from src.utils.ollama_client import build_ollama_generate_url


logger = get_logger("evaluator")

METRIC_DEFINITIONS = {
    "faithfulness": "Checks whether each statement in the generated answer is grounded in the retrieved context only.",
    "answer_relevancy": "Checks whether the answer directly and completely addresses the user question.",
    "context_precision": "Checks whether retrieved context chunks are mostly relevant and useful for answering the question.",
    "context_recall": "Checks whether retrieved context covers the key facts present in the golden answer.",
    "answer_correctness": "Checks whether the generated answer is factually aligned with the human validated answer.",
}


def _clip_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _trim_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return f"{head}\n...[truncated]...\n{tail}"


class RAGASEvaluator:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.model = config.evaluation.judge_model
        self.provider = str(getattr(config.evaluation, "provider", "openrouter")).strip().lower()
        self.api_key = str(config.api_keys.openrouter)
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.ollama_url = build_ollama_generate_url(
            getattr(config.evaluation, "ollama_endpoint", None)
        )
        if self.provider == "openrouter" and (
            not self.api_key
            or self.api_key.startswith("local-dev")
            or "REPLACE_WITH" in self.api_key
        ):
            raise ValueError(
                "OpenRouter API key is not configured. Set api_keys.openrouter in config/config.yaml "
                "before running evaluation with Kimi."
            )
        if self.provider not in {"openrouter", "ollama"}:
            raise ValueError(
                f"Unsupported evaluation.provider={self.provider}. Use 'openrouter' or 'ollama'."
            )

    def _build_prompt(
        self,
        metric: str,
        question: str,
        golden: str,
        answer: str,
        context: str,
    ) -> str:
        question = _trim_text(question, 800)
        golden = _trim_text(golden, 2400)
        answer = _trim_text(answer, 2400)
        context = _trim_text(context, 4000)
        context_block = (
            f"RETRIEVED CONTEXT: {context}\n"
            if metric in {"faithfulness", "context_precision", "context_recall"}
            else ""
        )
        return (
            "You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.\n"
            f"Evaluate the following response on the metric: {metric}\n"
            f"METRIC DEFINITION: {METRIC_DEFINITIONS[metric]}\n"
            f"QUESTION: {question}\n"
            f"GOLDEN ANSWER: {golden}\n"
            f"GENERATED ANSWER: {answer}\n"
            f"{context_block}"
            "SCORING INSTRUCTIONS:\n"
            '- Return ONLY a valid JSON object: {"score": <float>, "reasoning": "<1-2 sentence explanation>"}\n'
            "- 0.0 = completely fails the metric\n"
            "- 1.0 = perfectly satisfies the metric\n"
            "- Use precise decimals (e.g., 0.85, 0.63)\n"
            "- No markdown, no extra text outside JSON"
        )

    @staticmethod
    def _parse_judge_response(raw_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                score_match = re.search(
                    r"(?is)score[^0-9\-]*([01](?:\.\d+)?)",
                    raw_text,
                )
                if score_match:
                    score = float(score_match.group(1))
                    reasoning = raw_text.strip() or "No reasoning provided"
                    if not 0.0 <= score <= 1.0:
                        raise ValueError(f"Judge score out of range: {score}")
                    return {"score": score, "reasoning": reasoning[:500]}
                raise ValueError("No JSON object found in judge response")
            payload = json.loads(raw_text[start : end + 1])

        score = float(payload["score"])
        reasoning = str(payload.get("reasoning", "")).strip() or "No reasoning provided"
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Judge score out of range: {score}")
        return {"score": score, "reasoning": reasoning}


    async def _call_openrouter_judge(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=float(self.config.evaluation.timeout_seconds)) as client:
            response = await client.post(self.openrouter_url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        return str(body["choices"][0]["message"]["content"])

    async def _call_ollama_judge(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 220,
            },
        }
        timeout = min(45.0, float(self.config.evaluation.timeout_seconds))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.ollama_url, json=payload)
            response.raise_for_status()
            body = response.json()
        response_text = str(body.get("response", "") or "").strip()
        if response_text:
            return response_text
        thinking_text = str(body.get("thinking", "") or "").strip()
        if thinking_text:
            logger.warning("Judge response empty; falling back to thinking text")
            return thinking_text
        logger.warning(
            "Judge returned empty response and thinking fields; done_reason=%s eval_count=%s",
            body.get("done_reason"),
            body.get("eval_count"),
        )
        return ""

    async def _call_judge_api(self, prompt: str) -> str:
        if self.provider == "ollama":
            return await self._call_ollama_judge(prompt)
        return await self._call_openrouter_judge(prompt)

    async def score_metric(
        self,
        metric: str,
        question: str,
        golden: str,
        answer: str,
        context: str,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(metric, question, golden, answer, context)
        last_error: Exception | None = None
        max_attempts = 2 if self.provider == "ollama" else 3

        for attempt in range(max_attempts):
            try:
                raw_text = await self._call_judge_api(prompt)
                return self._parse_judge_response(raw_text)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Judge response parse failed for %s (attempt %s): %s",
                    metric,
                    attempt + 1,
                    str(exc),
                )
                await asyncio.sleep(0.5 * (attempt + 1))
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                logger.warning("Judge API call failed for %s: %s", metric, exc)
                if isinstance(exc, httpx.HTTPStatusError):
                    status_code = exc.response.status_code
                    if status_code == 401:
                        raise RuntimeError(
                            "Kimi judge unauthorized (401). "
                            "If using Ollama cloud models, run `ollama signin` and verify model access."
                        ) from exc
                    if status_code == 429:
                        await asyncio.sleep(1.5 * (2 ** attempt))
                        continue
                await asyncio.sleep(0.75 * (attempt + 1))

        raise RuntimeError(f"Kimi judge failed for metric={metric}: {last_error}")

    async def evaluate_row(
        self,
        question: str,
        golden: str,
        answer: str,
        context: str,
    ) -> dict[str, Any]:
        metric_names = list(METRIC_DEFINITIONS)
        metric_results: dict[str, dict[str, Any]] = {}
        if self.provider == "ollama":
            for metric in metric_names:
                try:
                    metric_results[metric] = await self.score_metric(
                        metric, question, golden, answer, context
                    )
                except Exception as exc:
                    logger.error("Metric failed (%s): %s", metric, exc)
                    metric_results[metric] = {
                        "score": 0.0,
                        "reasoning": f"metric_error: {exc}",
                    }
        else:
            scored = await asyncio.gather(
                *[
                    self.score_metric(metric, question, golden, answer, context)
                    for metric in metric_names
                ],
                return_exceptions=True,
            )
            for metric, result in zip(metric_names, scored):
                if isinstance(result, Exception):
                    logger.error("Metric failed (%s): %s", metric, result)
                    metric_results[metric] = {
                        "score": 0.0,
                        "reasoning": f"metric_error: {result}",
                    }
                else:
                    metric_results[metric] = result

        weighted_score = 0.0
        for metric, result in metric_results.items():
            metric_weight = float(getattr(self.config.evaluation.weights, metric))
            weighted_score += metric_weight * float(result["score"])

        return {
            "faithfulness": float(metric_results["faithfulness"]["score"]),
            "answer_relevancy": float(metric_results["answer_relevancy"]["score"]),
            "context_precision": float(metric_results["context_precision"]["score"]),
            "context_recall": float(metric_results["context_recall"]["score"]),
            "answer_correctness": float(metric_results["answer_correctness"]["score"]),
            "faithfulness_reason": str(metric_results["faithfulness"]["reasoning"]),
            "answer_relevancy_reason": str(metric_results["answer_relevancy"]["reasoning"]),
            "context_precision_reason": str(metric_results["context_precision"]["reasoning"]),
            "context_recall_reason": str(metric_results["context_recall"]["reasoning"]),
            "answer_correctness_reason": str(metric_results["answer_correctness"]["reasoning"]),
            "final_score": _clip_score(weighted_score),
        }

    async def evaluate_all(
        self,
        rows: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> list[dict[str, Any]]:
        completed = 0

        async def evaluate_one(row: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed
            async with semaphore:
                metrics = await self.evaluate_row(
                    row["question"],
                    row["golden_answer"],
                    row["generated_answer"],
                    row.get("retrieved_context", ""),
                )
            completed += 1
            if completed % 10 == 0:
                logger.info("Evaluated %s/%s rows", completed, len(rows))
            return {**row, **metrics}

        return await asyncio.gather(*(evaluate_one(row) for row in rows))
