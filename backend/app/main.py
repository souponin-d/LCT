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
from typing import Deque, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class BatchPayload(BaseModel):
    """Represents the structure of the batches emitted by the emulator.

    Attributes
    ----------
    timestamps:
        Millisecond timestamps associated with the sampled values.
    signals:
        Dictionary mapping signal names to lists of numeric samples.
    artifacts:
        List of [x1, x2] coordinate pairs that highlight detected artefacts.
    predictions:
        Arbitrary payload with model predictions (values will be defined later).
    metadata:
        Additional metadata (optional) to help debug or visualise the batch.
    """

    timestamps: list[int]
    signals: dict[str, list[float]]
    artifacts: list[tuple[float, float]]
    predictions: dict[str, list[float] | float | int | str | None]
    metadata: dict[str, str | int | float | None] | None = None


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
