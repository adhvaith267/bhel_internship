"""Uvicorn launcher entrypoint.

Run with ``python app.py``. Equivalent to ``python main.py``.
"""

import uvicorn

from bhel_internship.core.config import DEBUG, HOST, PORT

if __name__ == "__main__":
    uvicorn.run("bhel_internship.api.routes:app", host=HOST, port=PORT, reload=DEBUG)
