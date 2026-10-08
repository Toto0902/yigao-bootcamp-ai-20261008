"""E2E verification via MCP stdio protocol.

Verifies without an LLM API key:
  1. tools/list returns the query tool
  2. Unknown database name is rejected by multi-db routing
  3. Over-length question is rejected by validation
  4. Prometheus metrics endpoint serves collected counters
"""

import asyncio
import json

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(
    command="uv",
    args=["run", "python", "-m", "pg_mcp"],
    cwd=r"d:\work_spaces\work_spaces_trae\geektime-bootcamp-ai-master\w5\pg-mcp",
)

REQUEST_TIMEOUT = 30.0


def rpc(rpc_id: int, method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}}


async def call_tool(session: ClientSession, name: str, arguments: dict) -> dict:
    result = await session.call_tool(name, arguments)
    text = result.content[0].text if result.content else "{}"
    return json.loads(text)


async def main() -> None:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=REQUEST_TIMEOUT)
            print("[1] MCP session initialized")

            tools = await asyncio.wait_for(session.list_tools(), timeout=REQUEST_TIMEOUT)
            tool_names = [t.name for t in tools.tools]
            print(f"[2] tools/list -> {tool_names}")
            assert "query" in tool_names, "query tool missing"

            # Unknown database must be rejected before reaching any LLM
            res = await call_tool(
                session,
                "query",
                {"question": "count users", "database": "nonexistent_db"},
            )
            assert res["success"] is False, res
            assert res["error"]["code"] == "database_error", res["error"]
            assert "not found" in res["error"]["message"], res["error"]
            print(f"[3] unknown db rejected: {res['error']['message']}")
            print(f"    available: {res['error']['details']['available_databases']}")

            # Over-length question must be rejected by validation
            res = await call_tool(
                session,
                "query",
                {"question": "x" * 10001, "database": "blog_small"},
            )
            assert res["success"] is False, res
            # Rejected at QueryRequest model layer (invalid_request) or
            # orchestrator layer (question_too_long); both are valid guards
            code = res["error"]["code"].lower()
            assert code in ("question_too_long", "invalid_request"), res["error"]
            print(f"[4] long question rejected: {code}")

            # Metrics endpoint must serve counters while server is running
            metrics = await httpx.AsyncClient().get("http://localhost:9090/metrics")
            body = metrics.text
            assert "pg_mcp_query_requests_total" in body, "query counter missing"
            for line in body.splitlines():
                if line.startswith("pg_mcp_query_requests_total"):
                    print(f"[5] metrics: {line}")

    print("ALL E2E CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
