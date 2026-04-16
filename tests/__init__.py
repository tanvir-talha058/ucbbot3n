"""Test package bootstrap for the UCB Bank chatbot suite.

The local environment's Starlette/FastAPI test client can hang under Python
3.13, so the test package installs a small ASGI-backed replacement before the
test modules import `fastapi.testclient.TestClient`.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any


def _patch_testclient() -> None:
    try:
        import httpx
        import fastapi.testclient as fastapi_testclient
        import starlette.testclient as starlette_testclient
    except Exception:
        return

    class PatchedTestClient:
        def __init__(
            self,
            app: Any,
            base_url: str = "http://testserver",
            raise_server_exceptions: bool = True,
            root_path: str = "",
            backend: str | None = None,
            backend_options: dict[str, Any] | None = None,
            cookies: Any = None,
            headers: dict[str, str] | None = None,
            follow_redirects: bool = True,
            **_: Any,
        ) -> None:
            self.app = app
            self.base_url = base_url
            self.raise_server_exceptions = raise_server_exceptions
            self.root_path = root_path
            self.backend = backend
            self.backend_options = backend_options or {}
            self.cookies = cookies
            self.headers = {"user-agent": "testclient", **(headers or {})}
            self.follow_redirects = follow_redirects

        async def _request_async(self, method: str, url: str, **kwargs: Any):
            transport = httpx.ASGITransport(
                app=self.app,
                root_path=self.root_path,
                raise_app_exceptions=self.raise_server_exceptions,
            )
            async with httpx.AsyncClient(
                transport=transport,
                base_url=self.base_url,
                cookies=self.cookies,
                headers=self.headers,
                follow_redirects=self.follow_redirects,
            ) as client:
                return await client.request(method, url, **kwargs)

        def request(self, method: str, url: str, **kwargs: Any):
            return asyncio.run(self._request_async(method, url, **kwargs))

        def get(self, url: str, **kwargs: Any):
            return self.request("GET", url, **kwargs)

        def post(self, url: str, **kwargs: Any):
            return self.request("POST", url, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def close(self) -> None:
            return None

    fastapi_testclient.TestClient = PatchedTestClient
    starlette_testclient.TestClient = PatchedTestClient


_patch_testclient()

# Prevent the Ollama auto-start helper from spawning processes during tests.
os.environ.setdefault("OLLAMA_AUTOSTART", "false")
