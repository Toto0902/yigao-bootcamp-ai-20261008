"""Unit tests for retry with exponential backoff.

Tests cover success-without-retry, retry on transient errors with backoff,
disabled retry, immediate propagation of non-retryable errors, and exhaustion.
"""

import asyncio

import pytest

from pg_mcp.config.settings import ResilienceConfig
from pg_mcp.models.errors import (
    DatabaseConnectionError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from pg_mcp.models.errors import (
    ValidationError as PgValidationError,
)
from pg_mcp.resilience.retry import is_retryable_error, retry_with_backoff


class TestIsRetryableError:
    """Tests for the retryable error predicate."""

    def test_transient_errors_are_retryable(self) -> None:
        """Connection errors and LLM timeouts are considered transient."""
        assert is_retryable_error(DatabaseConnectionError("conn refused")) is True
        assert is_retryable_error(LLMTimeoutError("timeout")) is True
        assert is_retryable_error(LLMUnavailableError("503")) is True

    def test_deterministic_errors_are_not_retryable(self) -> None:
        """Validation and syntax errors must never be retried."""
        assert is_retryable_error(PgValidationError("bad sql")) is False
        assert is_retryable_error(ValueError("unexpected")) is False


class TestRetryWithBackoff:
    """Tests for retry_with_backoff."""

    @pytest.fixture
    def fast_config(self) -> ResilienceConfig:
        """Config with short delays so tests run quickly."""
        return ResilienceConfig(max_retries=3, retry_delay=0.1, backoff_factor=2.0)

    async def test_success_on_first_attempt(self, fast_config: ResilienceConfig) -> None:
        """Successful operation returns immediately without sleeping."""
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_with_backoff(operation, fast_config)
        assert result == "ok"
        assert calls == 1

    async def test_retries_transient_error_then_succeeds(
        self, fast_config: ResilienceConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Transient failure on first call retries and succeeds on second."""
        calls = 0
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise LLMTimeoutError("transient timeout")
            return "recovered"

        result = await retry_with_backoff(operation, fast_config)
        assert result == "recovered"
        assert calls == 2
        assert sleeps == [0.1]

    async def test_backoff_delays_grow_exponentially(
        self, fast_config: ResilienceConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each retry delay is multiplied by backoff_factor."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def operation() -> str:
            raise LLMTimeoutError("always timing out")

        with pytest.raises(LLMTimeoutError):
            await retry_with_backoff(operation, fast_config)

        assert sleeps == [0.1, 0.2, 0.4]

    async def test_retry_disabled_executes_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """retry_enabled=False runs the operation a single time."""
        config = ResilienceConfig(retry_enabled=False, max_retries=3)
        calls = 0

        async def fake_sleep(seconds: float) -> None:
            raise AssertionError("should not sleep when retry is disabled")

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise LLMTimeoutError("transient but retries disabled")

        with pytest.raises(LLMTimeoutError):
            await retry_with_backoff(operation, config)
        assert calls == 1

    async def test_non_retryable_error_propagates_immediately(
        self, fast_config: ResilienceConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-retryable exceptions raise at once without any retry."""
        calls = 0

        async def fake_sleep(seconds: float) -> None:
            raise AssertionError("should not sleep for non-retryable error")

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise PgValidationError("deterministic failure")

        with pytest.raises(PgValidationError):
            await retry_with_backoff(operation, fast_config)
        assert calls == 1

    async def test_exhaustion_raises_last_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After max_retries the last exception is re-raised directly."""
        config = ResilienceConfig(max_retries=2, retry_delay=0.1, backoff_factor=2.0)
        calls = 0

        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise LLMUnavailableError(f"attempt {calls} failed")

        with pytest.raises(LLMUnavailableError, match="attempt 3 failed"):
            await retry_with_backoff(operation, config)
        assert calls == 3

    async def test_custom_retryable_predicate(
        self, fast_config: ResilienceConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom retryable predicate overrides the default classification."""
        calls = 0

        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        class CustomTransientError(Exception):
            """Error only retryable via the custom predicate."""

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise CustomTransientError("custom transient")

        with pytest.raises(CustomTransientError):
            await retry_with_backoff(
                operation,
                fast_config,
                retryable=lambda exc: isinstance(exc, CustomTransientError),
            )
        assert calls == 4
