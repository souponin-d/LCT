"""FastAPI application acting as a thin proxy between the real-time emulator and the Svelte frontend.

At this stage the backend exposes minimal endpoints that will be expanded later:
* `/health` – quick readiness/liveness probe.
* `/ingest` – accepts batches forwarded by the emulator (JSON payloads) and stores them
  in an in-memory queue for the frontend to poll.
* `/stream` – placeholder endpoint to retrieve the most recent batch. In the future this can
  be upgraded to Server-Sent Events or WebSockets for real-time updates.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class BatchPayload(BaseModel):
    """Represents a single real-time batch forwarded by the emulator.

    Attributes
    ----------
    batch_id:
        Sequential identifier allowing the frontend to detect gaps.
    sent_at:
        UTC timestamp indicating when the batch was produced.
    batch_size:
        Number of samples included in the batch.
    time_step:
        Delay between batches, expressed in milliseconds.
    items:
        Mapping of signal names (``bpm``, ``uterus``) to their collected samples.
    """

    batch_id: str
    sent_at: datetime
    batch_size: int
    time_step: int
    items: dict[str, list[float]]


app = FastAPI(title="LCT Real-Time Proxy", version="0.1.0")

# For the bootstrap phase we maintain a bounded queue in memory so the frontend can poll.
BATCH_BUFFER: Deque[BatchPayload] = deque(maxlen=10)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Basic service healthcheck endpoint."""

    return {"status": "ok"}


@app.post("/ingest", status_code=202)
def ingest_batch(batch: BatchPayload) -> dict[str, str]:
    """Receive a batch from the emulator and keep it in a short-lived buffer."""

    BATCH_BUFFER.append(batch)
    return {"status": "accepted"}


@app.get("/stream")
def stream_latest() -> BatchPayload:
    """Return the most recent batch stored in the buffer."""

    try:
        latest: Optional[BatchPayload] = BATCH_BUFFER[-1]
    except IndexError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=404, detail="No batches available yet") from exc

    return latest


__all__ = ["app", "BatchPayload", "ingest_batch", "stream_latest"]
