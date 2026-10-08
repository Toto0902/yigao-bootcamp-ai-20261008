"""Unit tests for QueryOrchestrator.

This module tests the orchestrator's coordination of the query pipeline,
including per-database executor routing, retry logic, input length guard,
confidence threshold handling, and integration with all components.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    LLMError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    QueryRequest,
    ResultValidationResult,
    ReturnType,
)
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.resilience.circuit_breaker import CircuitState
from pg_mcp.services.orchestrator import QueryOrchestrator


def make_orchestrator(
    sql_generator: object = None,
    sql_validator: object = None,
    sql_executors: dict | None = None,
    result_validator: object = None,
    schema_cache: object = None,
    pools: dict | None = None,
    resilience_config: ResilienceConfig | None = None,
    validation_config: ValidationConfig | None = None,
    **kwargs: object,
) -> QueryOrchestrator:
    """Create a QueryOrchestrator with sensible mock defaults for tests."""
    pools = pools if pools is not None else {"test_db": MagicMock()}
    if sql_executors is None:
        sql_executors = {name: MagicMock() for name in pools}
    return QueryOrchestrator(
        sql_generator=sql_generator if sql_generator is not None else MagicMock(),
        sql_validator=sql_validator if sql_validator is not None else MagicMock(),
        sql_executors=sql_executors,
        result_validator=result_validator if result_validator is not None else MagicMock(),
        schema_cache=schema_cache if schema_cache is not None else MagicMock(),
        pools=pools,
        resilience_config=(
            resilience_config if resilience_config is not None else ResilienceConfig()
        ),
        validation_config=validation_config
        if validation_config is not None
        else ValidationConfig(),
        **kwargs,
    )


class TestDatabaseResolution:
    """Test database name resolution logic."""

    @pytest.fixture
    def orchestrator(self) -> QueryOrchestrator:
        """Create orchestrator with mocked components."""
        return make_orchestrator(pools={"db1": MagicMock(), "db2": MagicMock()})

    def test_resolve_database_specified_valid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified valid database."""
        result = orchestrator._resolve_database("db1")
        assert result == "db1"

    def test_resolve_database_specified_invalid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified but invalid database."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database("nonexistent")

        assert "not found" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]
        assert "db2" in exc_info.value.details["available_databases"]

    def test_resolve_database_auto_select_single(self) -> None:
        """Test auto-selecting when only one database available."""
        orchestrator = make_orchestrator(pools={"only_db": MagicMock()})

        result = orchestrator._resolve_database(None)
        assert result == "only_db"

    def test_resolve_database_auto_select_multiple_fails(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        """Test that auto-select fails when multiple databases available."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "multiple databases" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]

    def test_resolve_database_no_databases(self) -> None:
        """Test error when no databases configured."""
        orchestrator = make_orchestrator(pools={})

        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "no databases configured" in str(exc_info.value).lower()


class TestExecutorRouting:
    """Test per-database executor routing (multi-database compliance)."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="db1",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_executor_respected_for_specified_database(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Request for db2 must be executed by db2's executor, not db1's."""
        executor_db1 = AsyncMock()
        executor_db1.execute.return_value = ([{"id": 1}], 1)
        executor_db2 = AsyncMock()
        executor_db2.execute.return_value = ([{"id": 2}], 1)

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"db1": executor_db1, "db2": executor_db2},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"db1": MagicMock(), "db2": MagicMock()},
        )

        request = QueryRequest(
            question="Get users",
            database="db2",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        executor_db2.execute.assert_called_once_with("SELECT * FROM users;")
        executor_db1.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_executor_raises_error(self, mock_schema: DatabaseSchema) -> None:
        """Test error when no executor registered for resolved database."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={},  # No executors registered
            schema_cache=mock_cache,
            pools={"db1": MagicMock()},
        )

        request = QueryRequest(
            question="Test query",
            database="db1",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "no sql executor" in response.error.message.lower()


class TestQuestionLengthValidation:
    """Test question length guard before LLM invocation."""

    @pytest.mark.asyncio
    async def test_question_too_long_rejected(self) -> None:
        """Questions exceeding max_question_length are rejected before LLM call."""
        mock_generator = AsyncMock()

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            validation_config=ValidationConfig(max_question_length=10),
        )

        request = QueryRequest(
            question="This question is definitely longer than ten characters",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "question_too_long"
        mock_generator.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_question_within_limit_accepted(self) -> None:
        """Questions within limit proceed to LLM generation."""
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_cache = MagicMock()
        mock_cache.get.return_value = DatabaseSchema(
            database_name="test_db", tables=[], version="15.0"
        )

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            schema_cache=mock_cache,
            validation_config=ValidationConfig(max_question_length=100),
        )

        request = QueryRequest(
            question="Short question",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        mock_generator.generate.assert_called_once()


class TestSQLGenerationWithRetry:
    """Test SQL generation with retry logic."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_generate_sql_success_first_attempt(self, mock_schema: DatabaseSchema) -> None:
        """Test successful SQL generation on first attempt."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None  # No exception = valid

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            resilience_config=ResilienceConfig(max_retries=3),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert validation_result.is_select is True
        mock_generator.generate.assert_called_once()
        mock_validator.validate_or_raise.assert_called_once_with("SELECT * FROM users;")

    @pytest.mark.asyncio
    async def test_generate_sql_retry_on_validation_failure(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test retry logic when validation fails."""
        # Setup mocks - first attempt fails validation, second succeeds
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = [
            "SELECT * FROM user;",  # First attempt (wrong table name)
            "SELECT * FROM users;",  # Second attempt (correct)
        ]

        mock_validator = MagicMock()
        # First call raises error, second call succeeds
        mock_validator.validate_or_raise.side_effect = [
            SQLParseError('relation "user" does not exist'),
            None,  # Success on second attempt
        ]

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            resilience_config=ResilienceConfig(max_retries=3),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert mock_generator.generate.call_count == 2
        assert mock_validator.validate_or_raise.call_count == 2

        # Verify retry included error feedback
        second_call = mock_generator.generate.call_args_list[1]
        assert second_call.kwargs["previous_attempt"] == "SELECT * FROM user;"
        assert 'relation "user" does not exist' in second_call.kwargs["error_feedback"]

    @pytest.mark.asyncio
    async def test_generate_sql_fails_after_max_retries(self, mock_schema: DatabaseSchema) -> None:
        """Test failure after exhausting all retries."""
        # Setup mocks - all attempts fail validation
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError(
            "DELETE statements are not allowed"
        )

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            resilience_config=ResilienceConfig(max_retries=2),
        )

        # Execute and verify exception
        with pytest.raises(SecurityViolationError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Delete all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "DELETE statements are not allowed" in str(exc_info.value)
        # Should attempt max_retries + 1 times (initial + retries)
        assert mock_generator.generate.call_count == 3
        assert orchestrator.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_generate_sql_circuit_breaker_open(self, mock_schema: DatabaseSchema) -> None:
        """Test that open circuit breaker prevents SQL generation."""
        orchestrator = make_orchestrator(
            sql_generator=AsyncMock(),
            resilience_config=ResilienceConfig(circuit_breaker_threshold=1),
        )

        # Manually open the circuit breaker
        orchestrator.circuit_breaker._state = CircuitState.OPEN
        orchestrator.circuit_breaker._failure_count = 5

        # Attempt should fail immediately
        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "temporarily unavailable" in str(exc_info.value).lower()
        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_sql_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors during generation."""
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = RuntimeError("Unexpected error")

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            resilience_config=ResilienceConfig(max_retries=1),
        )

        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "unexpectedly" in str(exc_info.value).lower()
        assert orchestrator.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_generate_sql_shared_circuit_breaker(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test that injected circuit breaker instance is used (shared with server)."""
        from pg_mcp.resilience.circuit_breaker import CircuitBreaker

        shared_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            circuit_breaker=shared_breaker,
        )

        await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Success recorded on the shared breaker
        assert shared_breaker.failure_count == 0
        assert orchestrator.circuit_breaker is shared_breaker

    @pytest.mark.asyncio
    async def test_generate_sql_rate_limited(self, mock_schema: DatabaseSchema) -> None:
        """Test that LLM generation goes through the rate limiter."""
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
        )

        await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # The release task decrements active_count asynchronously; poll until done
        for _ in range(100):
            stats = orchestrator.rate_limiter.get_all_stats()
            if stats["llm"]["active_count"] == 0:
                break
            await asyncio.sleep(0.01)
        stats = orchestrator.rate_limiter.get_all_stats()
        # LLM limiter should have processed exactly one request and released it
        assert stats["llm"]["total_requests"] == 1
        assert stats["llm"]["active_count"] == 0


class TestResultValidation:
    """Test result validation logic."""

    @pytest.mark.asyncio
    async def test_validate_results_success(self) -> None:
        """Test successful result validation."""
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = ResultValidationResult(
            confidence=85,
            explanation="Results match the question well",
            suggestion=None,
            is_acceptable=True,
        )

        orchestrator = make_orchestrator(
            result_validator=mock_validator,
            validation_config=ValidationConfig(enabled=True),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 85
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_results_disabled(self) -> None:
        """Test that validation is skipped when disabled."""
        mock_validator = AsyncMock()

        orchestrator = make_orchestrator(
            result_validator=mock_validator,
            validation_config=ValidationConfig(enabled=False),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100
        mock_validator.validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_results_failure_uses_configured_threshold(self) -> None:
        """Test that validation failures fall back to min_confidence_score, not 100."""
        mock_validator = AsyncMock()
        mock_validator.validate.side_effect = Exception("Validation failed")

        orchestrator = make_orchestrator(
            result_validator=mock_validator,
            validation_config=ValidationConfig(enabled=True, min_confidence_score=70),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 70


class TestExecuteQueryFlow:
    """Test complete query execution flow."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_execute_query_sql_only(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=SQL."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            schema_cache=mock_cache,
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT * FROM users;"
        assert response.validation is not None
        assert response.validation.is_valid is True
        assert response.data is None  # No execution for SQL-only
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_with_results(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=RESULT."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT id, name FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.return_value = (
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            2,  # total count
        )

        mock_result_validator = AsyncMock()
        mock_result_validator.validate.return_value = ResultValidationResult(
            confidence=90,
            explanation="Good results",
            suggestion=None,
            is_acceptable=True,
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": mock_executor},
            result_validator=mock_result_validator,
            schema_cache=mock_cache,
            validation_config=ValidationConfig(enabled=True),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT id, name FROM users;"
        assert response.data is not None
        assert response.data.row_count == 2
        assert len(response.data.rows) == 2
        assert response.data.columns == ["id", "name"]
        assert response.confidence == 90
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_emits_metrics(self, mock_schema: DatabaseSchema) -> None:
        """Test that query requests emit metrics with status and database."""
        from pg_mcp.observability.metrics import MetricsCollector

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        metrics = MetricsCollector()

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            schema_cache=mock_cache,
            metrics=metrics,
        )

        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        metric_value = (
            metrics.query_requests.labels(status="success", database="test_db")._value.get()
        )
        assert metric_value == 1

    @pytest.mark.asyncio
    async def test_execute_query_schema_not_cached(self) -> None:
        """Test loading schema when not in cache."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Not in cache
        mock_cache.load = AsyncMock(return_value=mock_schema)

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_pool = MagicMock()

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            schema_cache=mock_cache,
            pools={"test_db": mock_pool},
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify schema was loaded
        mock_cache.load.assert_called_once_with("test_db", mock_pool)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_execute_query_schema_load_fails(self) -> None:
        """Test handling of schema load failure."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.load = AsyncMock(side_effect=Exception("DB connection failed"))

        orchestrator = make_orchestrator(
            schema_cache=mock_cache,
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "schema" in response.error.message.lower()
        assert response.generated_sql is None

    @pytest.mark.asyncio
    async def test_execute_query_validation_error(self) -> None:
        """Test handling of SQL validation errors."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            schema_cache=mock_cache,
            resilience_config=ResilienceConfig(max_retries=1),
        )

        # Execute
        request = QueryRequest(
            question="Delete all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "DELETE not allowed" in response.error.message
        assert response.error.code == "security_violation"

    @pytest.mark.asyncio
    async def test_execute_query_execution_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of SQL execution errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.side_effect = DatabaseError("Query execution failed")

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": mock_executor},
            schema_cache=mock_cache,
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "execution failed" in response.error.message.lower()
        assert response.error.code == "database_error"

    @pytest.mark.asyncio
    async def test_execute_query_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.side_effect = RuntimeError("Unexpected error")

        orchestrator = make_orchestrator(
            schema_cache=mock_cache,
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert response.error.code == "internal_error"
        assert "internal server error" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_execute_query_auto_select_database(self, mock_schema: DatabaseSchema) -> None:
        """Test auto-selecting database when only one available."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = make_orchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            schema_cache=mock_cache,
            pools={"only_db": MagicMock()},  # Only one database
        )

        # Execute without specifying database
        request = QueryRequest(
            question="Test query",
            database=None,  # No database specified
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        # Verify schema was fetched for auto-selected database
        mock_cache.get.assert_called_once_with("only_db")
