"""Utility for emulating real-time data streaming from archived recordings.

The emulator reads historical samples from JSON/JSONL/CSV files and emits batches of
measurements with a configurable cadence.  This allows the rest of the system to be
developed against deterministic data before integrating with the actual hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator


@dataclass(slots=True)
class RealTimeConfig:
    """Runtime configuration for the real-time emulator."""

    archive_path: Path
    time_step_ms: int
    batch_size: int
    loop_forever: bool = False
    output_path: Path = Path("batch.json")
    max_batches: int | None = None


class RealTimeEmulator:
    """Replay archived sensor readings as if they were arriving in real time."""

    def __init__(self, config: RealTimeConfig) -> None:
        self.config = config
        self._records = tuple(self._load_records(config.archive_path))
        if not self._records:
            raise ValueError(
                "Archive appears to be empty – populate `data/archive` with JSON/CSV files first."
            )
        self._cursor = 0

    # ------------------------------------------------------------------
    # Data loading helpers
    def _load_records(self, path: Path) -> Iterator[dict[str, Any]]:
        """Load archive content from a file or a directory.

        Supports JSON arrays, JSON Lines files (`*.jsonl`), and CSV files.
        """

        if path.is_dir():
            for file_path in sorted(path.iterdir()):
                if file_path.suffix.lower() in {".json", ".jsonl", ".csv"}:
                    yield from self._load_records(file_path)
            return

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "records" in payload:
                payload = payload["records"]
            if not isinstance(payload, list):
                raise ValueError(f"JSON archive {path} must contain a list of records")
            for record in payload:
                if not isinstance(record, dict):
                    raise ValueError(f"JSON record in {path} is not an object: {record!r}")
                yield record
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    yield self._normalise_csv_row(row)
        else:
            raise ValueError(f"Unsupported archive format: {path.suffix}")

    def _normalise_csv_row(self, row: dict[str, str]) -> dict[str, Any]:
        """Convert CSV rows into the standard record structure."""

        timestamp = int(float(row.get("timestamp", "0")))
        # Everything except the timestamp is treated as a signal
        signals: dict[str, float] = {}
        for key, value in row.items():
            if key == "timestamp":
                continue
            try:
                signals[key] = float(value)
            except (TypeError, ValueError):
                continue
        return {"timestamp": timestamp, "signals": signals}

    # ------------------------------------------------------------------
    # Streaming helpers
    def _next_chunk(self) -> list[dict[str, Any]]:
        if self._cursor >= len(self._records):
            if self.config.loop_forever:
                self._cursor = 0
            else:
                return []
        end = min(self._cursor + self.config.batch_size, len(self._records))
        chunk = list(self._records[self._cursor:end])
        self._cursor = end
        return chunk

    def _format_batch(self, chunk: Iterable[dict[str, Any]]) -> dict[str, Any]:
        timestamps: list[int] = []
        signals: dict[str, list[float]] = defaultdict(list)
        artifacts: list[list[float]] = []
        predictions: dict[str, list[Any]] = defaultdict(list)
        metadata: dict[str, Any] = {}

        for record in chunk:
            timestamp = record.get("timestamp")
            if timestamp is not None:
                timestamps.append(int(timestamp))

            for signal_name, value in record.get("signals", {}).items():
                try:
                    signals[signal_name].append(float(value))
                except (TypeError, ValueError):
                    continue

            for coords in record.get("artifacts", []) or []:
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    artifacts.append([float(coords[0]), float(coords[1])])

            for key, value in (record.get("predictions") or {}).items():
                predictions[key].append(value)

            metadata.update(record.get("metadata") or {})

        return {
            "timestamps": timestamps,
            "signals": signals,
            "artifacts": artifacts,
            "predictions": predictions,
            "metadata": metadata or None,
        }

    def stream_batches(self) -> Generator[dict[str, Any], None, None]:
        """Yield formatted batches according to the configured cadence."""

        emitted = 0
        while True:
            if self.config.max_batches is not None and emitted >= self.config.max_batches:
                return

            chunk = self._next_chunk()
            if not chunk:
                return

            batch = self._format_batch(chunk)
            emitted += 1
            yield batch
            time.sleep(self.config.time_step_ms / 1000)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Continuously write batches to ``config.output_path`` in JSON format."""

        for batch in self.stream_batches():
            self.config.output_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")


def parse_args() -> RealTimeConfig:
    parser = argparse.ArgumentParser(description="Real-time archive playback emulator")
    parser.add_argument("archive", type=Path, help="Path to the archive file or directory")
    parser.add_argument("--time-step", type=int, default=1000, help="Cadence between batches in milliseconds")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of samples per batch")
    parser.add_argument(
        "--loop", action="store_true", help="Restart from the beginning when the archive is exhausted"
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Optionally cap the number of batches produced"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("batch.json"), help="Destination file for the generated batch"
    )
    args = parser.parse_args()
    return RealTimeConfig(
        archive_path=args.archive,
        time_step_ms=args.time_step,
        batch_size=args.batch_size,
        loop_forever=args.loop,
        max_batches=args.max_batches,
        output_path=args.output,
    )


def main() -> None:
    config = parse_args()
    emulator = RealTimeEmulator(config)
    emulator.run()


if __name__ == "__main__":
    main()
