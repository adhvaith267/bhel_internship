"""`python -m bhel_internship` — serve the API, run eval, or launch the terminal CLI."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn

        from bhel_internship.core.config import settings

        uvicorn.run(
            "bhel_internship.api.routes:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
        )
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        from bhel_internship.eval.cli import main as eval_main

        return eval_main(sys.argv[2:])

    from bhel_internship.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
