"""Keep the stdio MCP server alive without an MCP client.

The stdio transport terminates when stdin reaches EOF, which happens
immediately when the server is launched from a plain terminal. This wrapper
holds a pipe open as the child's stdin so the server (and its Prometheus
metrics endpoint) keeps running for manual inspection.

Stop with Ctrl+C.
"""

import os
import subprocess
import sys


def main() -> None:
    read_fd, write_fd = os.pipe()
    child = subprocess.Popen([sys.executable, "-m", "pg_mcp"], stdin=read_fd)
    os.close(read_fd)  # child owns the read end now; write end stays open
    try:
        child.wait()
    except KeyboardInterrupt:
        child.terminate()
        child.wait(timeout=10)
    finally:
        os.close(write_fd)


if __name__ == "__main__":
    main()
