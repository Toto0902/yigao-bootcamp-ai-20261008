"""Retry with exponential backoff for transient errors.

This module provides retry logic for transient failures in LLM API calls
and database operations. It activates the previously-unused ``ResilienceConfig``
settings: ``retry_enabled``, ``max_retries``, ``retry_delay`` and
``backoff_factor``.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pg_mcp.config.settings import ResilienceConfig
from pg_mcp.models.errors import DatabaseConnectionError, LLMTimeoutError, LLMUnavailableError

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Exception types considered transient and therefore retryable.
#: Deterministic failures (SQL syntax errors, invalid requests, security
#: violations) are deliberately excluded to avoid pointless retries.
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    DatabaseConnectionError,
    LLMTimeoutError,
    LLMUnavailableError,
)


def is_retryable_error(exc: BaseException) -> bool:
    """Determine whether an exception is transient and worth retrying.

    Args:
        exc: The exception to inspect.

    Returns:
        bool: True if the error is transient, False otherwise.
    """
    return isinstance(exc, RETRYABLE_EXCEPTIONS)


async def retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    config: ResilienceConfig,
    *,
    retryable: Callable[[BaseException], bool] = is_retryable_error,
    operation_name: str = "operation",
    operation_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> T:
    """Execute an async operation with retry and exponential backoff.

    Retry behavior is fully driven by ``ResilienceConfig``:
    - ``retry_enabled=False`` executes once without any retry.
    - ``max_retries`` controls the number of retry attempts after the first call.
    - ``retry_delay`` is the initial sleep duration between attempts.
    - ``backoff_factor`` multiplies the delay after each failed attempt.

    Only exceptions accepted by ``retryable`` trigger a retry; any other
    exception propagates immediately to the caller.

    Args:
        operation: Zero-argument async callable to execute.
        config: Resilience configuration providing retry settings.
        retryable: Predicate deciding whether an exception is retryable.
        operation_name: Human-readable name used in log messages.
        operation_id: Optional correlation id (e.g. request_id) for logs.
        extra: Optional extra context merged into log records.

    Returns:
        T: The result of a successful ``operation`` invocation.

    Raises:
        Exception: The original exception if it is not retryable or retries
            are exhausted (the last exception is re-raised directly).

    Example:
        >>> sql = await retry_with_backoff(
        ...     lambda: generator.generate(question, schema),
        ...     resilience_config,
        ...     operation_name="generate_sql",
        ...     operation_id=request_id,
        ... )
    """
    max_attempts = config.max_retries + 1
    delay = config.retry_delay
    log_extra: dict[str, Any] = {**(extra or {}), "operation_id": operation_id}

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if not config.retry_enabled or not retryable(exc) or attempt >= max_attempts:
                logger.error(
                    "%s failed on attempt %d/%d, giving up: %s",
                    operation_name,
                    attempt,
                    max_attempts,
                    exc,
                    extra=log_extra,
                )
                raise

            logger.warning(
                "%s failed (attempt %d/%d), retrying in %.2fs: %s",
                operation_name,
                attempt,
                max_attempts,
                delay,
                exc,
                extra=log_extra,
            )
            await asyncio.sleep(delay)
            delay *= config.backoff_factor

    # Unreachable: every loop iteration either returns or raises.
    raise AssertionError("retry loop exited unexpectedly")
