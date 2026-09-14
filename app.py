"""Uvicorn launcher entrypoint.

Run with ``python app.py``. Equivalent to ``python main.py``.
"""

import uvicorn

from docurag.core.config import DEBUG, HOST, PORT

if __name__ == "__main__":
    uvicorn.run("docurag.api.routes:app", host=HOST, port=PORT, reload=DEBUG)
