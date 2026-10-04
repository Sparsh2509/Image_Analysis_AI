"""Application entrypoint for deployment platforms (Render, Railway, etc.)."""

import os
import uvicorn

if __name__ == "__main__":
    # Render assigns the port via the $PORT environment variable (default 10000)
    port = int(os.environ.get("PORT", 10000))
    # Must bind to 0.0.0.0 for external container routing
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
