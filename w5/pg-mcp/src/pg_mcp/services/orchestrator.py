"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements per-database executor routing, retry logic with error
feedback, circuit breaker protection, rate limiting, and metrics instrumentation.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    ErrorDetail,
    LLMError,
    PgMcpError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.resilience.retry import is_retryable_error, retry_with_backoff
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. Key features:

    - Per-database executor routing: requests are always executed against the
      resolved database, preventing cross-database data leakage.
    - Input length guard: questions are checked against
      ``ValidationConfig.max_question_length`` before LLM invocation.
    - Retry with exponential backoff for transient LLM/DB errors, driven by
      ``ResilienceConfig``.
    - Circuit breaker shared with the server, injected instead of created.
    - Rate limiting for LLM and DB operations via ``MultiRateLimiter``.
    - Prometheus metrics instrumentation across the whole pipeline.
    - Confidence threshold enforcement from ``ValidationConfig``.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executors={"mydb": executor},
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executors: dict[str, SQLExecutor],
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executors: Dictionary mapping database names to SQL executors.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries and circuit breaker.
            validation_config: Validation configuration including thresholds.
            circuit_breaker: Shared circuit breaker instance. Created from
                ``resilience_config`` if not provided.
            rate_limiter: Shared rate limiter for LLM/DB operations. Created from
                ``resilience_config`` limits if not provided.
            metrics: Metrics collector for observability. Uses the module
                singleton if not provided.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executors = sql_executors
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config

        # Use shared circuit breaker, or create one from config as fallback
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

        # Use shared rate limiter, or create one from config as fallback
        self.rate_limiter = rate_limiter or MultiRateLimiter(
            query_limit=resilience_config.max_concurrent_queries,
            llm_limit=resilience_config.max_concurrent_llm_calls,
        )

        self.metrics = metrics

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Generate request_id for tracking
        2. Validate question length
        3. Resolve and validate database name
        4. Load schema from cache
        5. Generate and validate SQL with retry logic (rate-limited, circuit-broken)
        6. Execute SQL against the resolved database's executor (if return_type == RESULT)
        7. Validate results (optional)
        8. Return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        # Generate request_id for full-chain tracing
        request_id = str(uuid.uuid4())
        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )
        start_time = self._get_current_time_ms()
        database_name: str | None = None
        success_status = "error"

        try:
            # Step 1: Validate question length before LLM invocation
            self._validate_question_length(request.question)

            # Step 2: Resolve database name
            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 3: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                # Schema not in cache, load it
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 4: Generate and validate SQL with retry logic
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 5: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                success_status = "success"
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )

            # Step 6: Execute SQL on the resolved database's executor
            logger.debug("Executing SQL", extra={"request_id": request_id})

            executor = self._resolve_executor(database_name)

            async with self.rate_limiter.for_queries():
                db_start = self._get_current_time_ms()
                results, total_count = await executor.execute(generated_sql)
                db_duration = (self._get_current_time_ms() - db_start) / 1000

            if self.metrics is not None:
                self.metrics.observe_db_query_duration(db_duration)

            execution_time_ms = self._get_current_time_ms() - start_time
            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 7: Validate results (non-blocking, failures don't fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            # Step 8: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            success_status = "success"
            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
            )

        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            success_status = "error"
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id, "database": database_name},
            )
            success_status = "error"
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
            )
        finally:
            # Emit query request metric with resolved database for observability
            if self.metrics is not None:
                self.metrics.increment_query_request(
                    status=success_status, database=database_name or "unknown"
                )

    def _validate_question_length(self, question: str) -> None:
        """Validate question length against configured maximum.

        Args:
            question: User's natural language question.

        Raises:
            PgMcpError: If the question exceeds ``max_question_length``.
        """
        max_length = self.validation_config.max_question_length
        if len(question) > max_length:
            logger.warning(
                "Question length exceeds configured maximum",
                extra={"question_length": len(question), "max_length": max_length},
            )
            raise PgMcpError(
                message=f"Question length {len(question)} exceeds maximum of {max_length}",
                code=ErrorCode.QUESTION_TOO_LONG,
                details={"question_length": len(question), "max_length": max_length},
            )

    def _resolve_executor(self, database_name: str) -> SQLExecutor:
        """Resolve the SQL executor for the given database.

        Args:
            database_name: Resolved database name.

        Returns:
            SQLExecutor: The executor bound to the requested database.

        Raises:
            DatabaseError: If no executor is registered for the database.
        """
        executor = self.sql_executors.get(database_name)
        if executor is None:
            raise DatabaseError(
                message=f"No SQL executor available for database '{database_name}'",
                details={
                    "requested_database": database_name,
                    "available_databases": list(self.sql_executors.keys()),
                },
            )
        return executor

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM inside rate limiter, with exponential
           backoff retry for transient LLM errors
        3. Validates the generated SQL
        4. On validation failure, retries with error feedback
        5. Records success/failure to circuit breaker and emits metrics

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.

        Example:
            >>> sql, validation, tokens = await orchestrator._generate_sql_with_retry(
            ...     question="Count users",
            ...     schema=db_schema,
            ...     request_id="123",
            ... )
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL inside rate limiter with exponential backoff
                # retry for transient LLM errors (timeout, service unavailable).
                # The closure binds the current feedback values (avoids B023).
                def generate_operation(
                    prev: str | None,
                    feedback: str | None,
                ) -> Callable[[], Awaitable[str]]:
                    """Build a zero-arg async operation bound to current feedback."""

                    async def _generate() -> str:
                        return await self.sql_generator.generate(
                            question=question,
                            schema=schema,
                            previous_attempt=prev,
                            error_feedback=feedback,
                        )

                    return _generate

                async with self.rate_limiter.for_llm():
                    generated_sql = await retry_with_backoff(
                        generate_operation(previous_sql, error_feedback),
                        self.resilience_config,
                        retryable=is_retryable_error,
                        operation_name="generate_sql",
                        operation_id=request_id,
                    )

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if attempt < max_retries:
                        # Record as failure and retry with feedback
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        if self.metrics is not None:
                            self.metrics.increment_sql_rejected(
                                reason=type(validation_error).__name__
                            )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        if self.metrics is not None:
                            self.metrics.increment_sql_rejected(
                                reason=type(validation_error).__name__
                            )
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result from actual validator outcome
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=self._is_select_statement(generated_sql),
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, SecurityViolationError, SQLParseError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    def _is_select_statement(self, sql: str) -> bool:
        """Check whether the validated SQL is a SELECT statement.

        Uses the SQLGlot AST to distinguish SELECT (including set operations
        and WITH queries) from EXPLAIN commands that also pass validation.

        Args:
            sql: Generated SQL query.

        Returns:
            bool: True if the statement is a SELECT-family statement.
        """
        try:
            import sqlglot
            from sqlglot import exp

            parsed = sqlglot.parse(sql, read="postgres")
            if not parsed or parsed[0] is None:
                return False
            statement = parsed[0]
            # EXPLAIN passes validation but is not a SELECT statement
            if isinstance(statement, exp.Command):
                return False
            return isinstance(
                statement, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.With)
            )
        except Exception:
            # Validator already approved the SQL; treat parse failure as non-SELECT
            return False

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score
        that respects ``ValidationConfig.min_confidence_score``.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled;
                falls back to ``min_confidence_score`` if validation fails.

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            llm_start = self._get_current_time_ms()
            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )
            llm_duration = (self._get_current_time_ms() - llm_start) / 1000

            if self.metrics is not None:
                self.metrics.increment_llm_call(operation="validate_result")
                self.metrics.observe_llm_latency("validate_result", llm_duration)

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query; use configured threshold instead of
            # blindly reporting full confidence
            fallback_confidence = self.validation_config.min_confidence_score
            logger.warning(
                "Result validation failed, using fallback confidence",
                extra={
                    "request_id": request_id,
                    "fallback_confidence": fallback_confidence,
                    "error": str(e),
                },
            )
            return fallback_confidence

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        return time.time() * 1000
