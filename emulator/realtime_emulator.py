"""Utility for emulating real-time data streaming from archived recordings.

The emulator replays historical measurements stored in CSV files located inside the
numbered folders of the ``data`` directory.  Each folder contains two sub-directories:
``bpm`` and ``uterus``.  Files inside these folders share the same prefix and differ only
by the ``_1`` / ``_2`` suffix that indicates the source sensor.  The emulator loads
paired files and produces combined samples that contain both signals.
"""
from __future__ import annotations

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

    dataset_path: Path
    time_step_ms: int
    batch_size: int


class RealTimeEmulator:
    """Replay archived sensor readings as if they were arriving in real time."""

    def __init__(self, config: RealTimeConfig) -> None:
        self.config = config
        self._records = tuple(self._load_records(config.dataset_path))
        if not self._records:
            raise ValueError(
                "Dataset appears to be empty – ensure the numbered folder contains paired CSV files."
            )
        self._cursor = 0

    # ------------------------------------------------------------------
    # Data loading helpers
    def _load_records(self, folder: Path) -> Iterator[dict[str, Any]]:
        """Load paired BPM and uterus measurements from the provided folder."""

        bpm_folder = folder / "bpm"
        uterus_folder = folder / "uterus"

        if not bpm_folder.is_dir() or not uterus_folder.is_dir():
            raise ValueError(
                "Dataset folder must contain `bpm` and `uterus` subdirectories with CSV files."
            )

        prefixes = sorted({self._extract_prefix(path.name) for path in bpm_folder.glob("*.csv")})
        for prefix in prefixes:
            bpm_path = bpm_folder / f"{prefix}_1.csv"
            uterus_path = uterus_folder / f"{prefix}_2.csv"
            if not bpm_path.exists() or not uterus_path.exists():
                continue
            yield from self._merge_pair(bpm_path, uterus_path)

    def _extract_prefix(self, filename: str) -> str:
        stem = Path(filename).stem
        if "_" not in stem:
            return stem
        return stem.rsplit("_", 1)[0]

    def _read_signal_csv(self, file_path: Path) -> Iterator[tuple[int, float]]:
        with file_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if "time_sec" not in reader.fieldnames or "value" not in reader.fieldnames:
                raise ValueError(f"Unexpected CSV format in {file_path}")
            for row in reader:
                try:
                    timestamp_ms = int(round(float(row["time_sec"]) * 1000))
                    value = float(row["value"])
                except (TypeError, ValueError):
                    continue
                yield timestamp_ms, value

    def _merge_pair(self, bpm_path: Path, uterus_path: Path) -> Iterator[dict[str, Any]]:
        bpm_stream = list(self._read_signal_csv(bpm_path))
        uterus_stream = list(self._read_signal_csv(uterus_path))
        dataset_kind = bpm_path.parents[2].name if len(bpm_path.parents) >= 3 else "unknown"
        folder_id = bpm_path.parents[1].name if len(bpm_path.parents) >= 2 else ""
        record_id = bpm_path.stem.rsplit("_", 1)[0]
        for (timestamp_ms, bpm_value), (_, uterus_value) in zip(bpm_stream, uterus_stream):
            yield {
                "timestamp": timestamp_ms,
                "signals": {
                    "bpm": bpm_value,
                    "uterus": uterus_value,
                },
                "metadata": {
                    "dataset": dataset_kind,
                    "folder": folder_id,
                    "record": record_id,
                },
            }

    # ------------------------------------------------------------------
    # Streaming helpers
    def _next_chunk(self) -> list[dict[str, Any]]:
        if self._cursor >= len(self._records):
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

        while True:
            chunk = self._next_chunk()
            if not chunk:
                return

            batch = self._format_batch(chunk)
            yield batch
            if self._cursor < len(self._records):
                time.sleep(self.config.time_step_ms / 1000)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Print batches as JSON objects."""

        for batch in self.stream_batches():
            print(json.dumps(batch, indent=2, ensure_ascii=False), flush=True)


# Local configuration -------------------------------------------------------
# The dataset path must point to one of the numbered folders inside either the
# ``data/hypoxia`` or ``data/regular`` directories, for example ``data/hypoxia/1``.
DATASET_PATH = Path("data/hypoxia/1")
TIME_STEP_MS = 1000
BATCH_SIZE = 10


def main() -> None:
    config = RealTimeConfig(dataset_path=DATASET_PATH, time_step_ms=TIME_STEP_MS, batch_size=BATCH_SIZE)
    emulator = RealTimeEmulator(config)
    emulator.run()


if __name__ == "__main__":
    main()
