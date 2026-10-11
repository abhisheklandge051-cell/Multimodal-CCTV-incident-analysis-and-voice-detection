"""
backend/api/websocket_manager.py
WebSocket connection manager — broadcast JSON events to all connected React clients.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("ws_manager")


class ConnectionManager:
    """Manages active WebSocket connections and provides broadcast helpers."""

    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        logger.info(f"WS client connected. Total connections: {len(self._active)}")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._active:
            self._active.remove(ws)
        logger.info(f"WS client disconnected. Total connections: {len(self._active)}")

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send JSON payload to all connected clients; silently remove dead connections."""
        payload = json.dumps(data, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._active):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_from_thread(self, loop: asyncio.AbstractEventLoop, data: dict[str, Any]) -> None:
        """Thread-safe broadcast — call from background (non-async) threads."""
        asyncio.run_coroutine_threadsafe(self.broadcast(data), loop)

    @property
    def connection_count(self) -> int:
        return len(self._active)


# Singleton used across the application
manager = ConnectionManager()
