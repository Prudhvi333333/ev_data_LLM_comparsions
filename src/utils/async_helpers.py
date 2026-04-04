from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from functools import wraps
from time import perf_counter
from typing import Any, TypeVar

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger


T = TypeVar("T")

logger = get_logger("async_helpers")


class RetryExhaustedError(Exception):
    """Raised after retries are exhausted for a transient async operation."""


class AsyncRateLimiter:
    def __init__(self, max_concurrency: int) -> None:
        self.semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.semaphore.release()


def retry_async(max_retries: int = 3, backoff_base: float = 2.0) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapped(*args, **kwargs) -> T:
            last_error: Exception | None = None
            retryer = AsyncRetrying(
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=backoff_base, max=30),
                retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPError, ValueError)),
                reraise=False,
            )

            async for attempt in retryer:
                with attempt:
                    try:
                        return await func(*args, **kwargs)
                    except httpx.TimeoutException as exc:
                        last_error = exc
                        raise
                    except Exception as exc:
                        last_error = exc
                        raise

            raise RetryExhaustedError(
                f"{func.__name__} failed after {max_retries} attempts: {last_error}"
            ) from last_error

        return wrapped

    return decorator


class AdaptiveBatcher:
    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 10,
    ) -> None:
        self.batch_size = max(min_batch_size, int(initial_batch_size))
        self.min_batch_size = max(1, int(min_batch_size))
        self.max_batch_size = max(self.batch_size, int(max_batch_size))
        self._fast_batches = 0

    def record_success(self, elapsed_seconds: float) -> None:
        if elapsed_seconds < 5:
            self._fast_batches += 1
        else:
            self._fast_batches = 0

        if self._fast_batches >= 3 and self.batch_size < self.max_batch_size:
            self.batch_size += 1
            self._fast_batches = 0
            logger.info("Adaptive batch size increased to %s", self.batch_size)

    def record_timeout(self) -> None:
        self._fast_batches = 0
        if self.batch_size > self.min_batch_size:
            self.batch_size = max(self.min_batch_size, self.batch_size - 2)
            logger.warning("Adaptive batch size reduced to %s after timeout", self.batch_size)


async def create_timeout_guard(
    coro: Awaitable[T],
    timeout_sec: float,
    fallback_value: T,
) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        return fallback_value


async def async_batch(
    items: Iterable[Any],
    fn: Callable[[Any], Awaitable[T]],
    batch_size: int,
    semaphore: asyncio.Semaphore | None = None,
    adaptive_batcher: AdaptiveBatcher | None = None,
) -> list[T]:
    values = list(items)
    outputs: list[T] = []
    index = 0

    while index < len(values):
        current_batch_size = adaptive_batcher.batch_size if adaptive_batcher else max(1, int(batch_size))
        batch = values[index : index + current_batch_size]
        started = perf_counter()

        async def invoke(item: Any) -> T:
            if semaphore is None:
                return await fn(item)
            async with semaphore:
                return await fn(item)

        try:
            outputs.extend(await asyncio.gather(*(invoke(item) for item in batch)))
            elapsed = perf_counter() - started
            if adaptive_batcher is not None:
                adaptive_batcher.record_success(elapsed)
            index += len(batch)
        except httpx.TimeoutException as exc:
            if adaptive_batcher is not None:
                adaptive_batcher.record_timeout()
            raise RetryExhaustedError("Batch processing timed out.") from exc

    return outputs
