"""`python -m src` — serve the API, run eval, or launch the terminal CLI."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn

        from src.core.config import settings

        uvicorn.run(
            "src.api.routes:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
        )
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        from src.eval.cli import main as eval_main

        return eval_main(sys.argv[2:])

    from src.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
