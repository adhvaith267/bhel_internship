"""`python -m src` — start the web server."""

from __future__ import annotations

import uvicorn
from src.core.config import settings


def main() -> None:
    uvicorn.run(
        "src.api.routes:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
