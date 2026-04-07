"""
HTTP client for Business Chat OpenEnv.
Connects to the running FastAPI server via HTTP.
"""

import requests
from typing import Optional


class BusinessChatEnvClient:
    """Simple HTTP client for the Business Chat OpenEnv server."""

    def __init__(self, base_url: str = "http://localhost:7860", session_id: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self._headers = {"Content-Type": "application/json", "X-Session-Id": session_id}

    def reset(self, task_type: Optional[str] = None, business_type: Optional[str] = None) -> dict:
        payload = {}
        if task_type:
            payload["task_type"] = task_type
        if business_type:
            payload["business_type"] = business_type
        response = requests.post(f"{self.base_url}/reset", json=payload, headers=self._headers)
        response.raise_for_status()
        return response.json()

    def step(self, action: dict) -> dict:
        response = requests.post(f"{self.base_url}/step", json={"action": action}, headers=self._headers)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict:
        response = requests.get(f"{self.base_url}/state", headers=self._headers)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
