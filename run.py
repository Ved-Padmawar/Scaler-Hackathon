"""
Run the Business Chat OpenEnv FastAPI server locally.
Usage: poetry run python run.py
"""

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )
