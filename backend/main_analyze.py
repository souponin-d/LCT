"""Временная заглушка для анализа батчей, поступающих от эмулятора по WebSocket."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


app = FastAPI(title="LCT Analysis Stub", version="0.1.0")


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
            print(json.dumps(processed, ensure_ascii=False, indent=2))
    finally:
        with suppress(RuntimeError):
            await websocket.close()


__all__ = ["app"]
