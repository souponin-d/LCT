"""Временная заглушка для анализа батчей, поступающих от эмулятора по WebSocket."""
from __future__ import annotations

import json
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

app = FastAPI(title="LCT Analysis Stub", version="0.1.0")

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


_frontend_clients: Set[WebSocket] = set()


async def _broadcast_to_frontend(payload: Dict[str, Any]) -> None:
    """Отправить ``processed``-сообщение всем подключённым фронтам."""

    disconnected: list[WebSocket] = []
    for client in list(_frontend_clients):
        try:
            await client.send_json(payload)
        except Exception:  # pragma: no cover - best-effort рассылка
            disconnected.append(client)

    for client in disconnected:
        _frontend_clients.discard(client)


def _iso_now() -> str:
    """Вернуть текущий момент времени в формате ISO 8601 с суффиксом Z."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_processed_payload(batch: Dict[str, Any], started_at: float) -> Dict[str, Any]:
    """Сформировать ответ с заглушками на основе входного батча."""

    items = batch.get("items", {}) if isinstance(batch, dict) else {}
    predicted = {
        name: [0 for _ in values] if isinstance(values, list) else []
        for name, values in items.items()
    }
    bpm_values = items.get("bpm", []) if isinstance(items.get("bpm"), list) else []
    uterus_values = items.get("uterus", []) if isinstance(items.get("uterus"), list) else []
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    return {
        "sent_at": _iso_now(),
        "latency_ms": latency_ms,
        "batch": items,
        "predicted": predicted,
        "artefacts": [
            [[100, 150], "spike"],
            [[300, 350], "dropout"],
        ],
        "metrics": [
            [round(sum(bpm_values) / len(bpm_values), 1) if bpm_values else 0.0, "mean bpm"],
            [round(sum(uterus_values) / len(uterus_values), 1) if uterus_values else 0.0, "mean uterus"],
        ],
    }


@app.websocket("/ws/analyze")
async def analyze_websocket(websocket: WebSocket) -> None:
    """Получать батчи по WebSocket и отвечать заглушкой ``processed_data``."""

    await websocket.accept()
    try:
        while True:
            try:
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            started_at = time.perf_counter()
            try:
                batch = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid payload"})
                continue

            processed = _build_processed_payload(batch, started_at)
            await websocket.send_json(processed)
            await _broadcast_to_frontend(processed)
    finally:
        with suppress(RuntimeError):
            await websocket.close()


@app.websocket("/ws/processed")
async def processed_stream(websocket: WebSocket) -> None:
    """Стрим обработанных данных для фронтенда по WebSocket."""

    await websocket.accept()
    _frontend_clients.add(websocket)
    try:
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        _frontend_clients.discard(websocket)
        with suppress(RuntimeError):
            await websocket.close()


__all__ = ["app"]
