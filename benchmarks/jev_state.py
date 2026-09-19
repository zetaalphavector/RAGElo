"""Sends Jev a JSON object as the state, which TypeSafe recommends over a string for most requests.

RAGElo prompts are strings, so the state leaves the evaluator as JSON text and becomes an object on the wire.
"""

from __future__ import annotations

import json

import httpx


class JsonStateTransport(httpx.AsyncBaseTransport):
    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        body["state"] = json.loads(body["state"])
        headers = {name: value for name, value in request.headers.items() if name.lower() != "content-length"}
        return await self._transport.handle_async_request(
            httpx.Request(request.method, request.url, headers=headers, json=body)
        )
