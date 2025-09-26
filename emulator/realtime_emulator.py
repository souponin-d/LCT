"""Utility for emulating real-time data streaming from archived recordings.

The emulator replays historical samples stored in CSV archives and emits batches of
measurements with a configurable cadence.  This allows the rest of the system to be
developed against deterministic data before integrating with the actual hardware.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator


# ``dataclass`` gained the ``slots`` parameter only in Python 3.10.
# Older interpreters (e.g. Python 3.9 used on some developer machines) raise a
# ``TypeError`` when the argument is provided.  The wrapper below keeps slot
# generation enabled where possible, while remaining compatible with Python 3.9.
def dataclass_with_optional_slots(_cls=None, /, *, slots: bool = True, **kwargs):
    """Return a ``dataclass`` decorator that sets ``slots`` when supported."""

    if sys.version_info >= (3, 10) and slots:
        kwargs["slots"] = True

    def wrap(cls):
        return dataclass(cls, **kwargs)

    if _cls is None:
        return wrap
    return wrap(_cls)


@dataclass_with_optional_slots
class RealTimeConfig:
    """Runtime configuration for the real-time emulator."""

    archive_path: Path
    time_step_ms: int
    batch_size: int


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
        """Load archive content from a directory with ``bpm`` and ``uterus`` CSV files."""

        if not path.is_dir():
            raise ValueError(
                "Archive path must point to a directory containing 'bpm' and 'uterus' subdirectories"
            )

        bpm_dir = path / "bpm"
        uterus_dir = path / "uterus"

        if not bpm_dir.is_dir() or not uterus_dir.is_dir():
            raise ValueError(
                "Archive directory must include both 'bpm' and 'uterus' subdirectories with CSV files"
            )

        bpm_files = {
            self._strip_suffix(csv_path.name): csv_path for csv_path in bpm_dir.glob("*.csv")
        }
        uterus_files = {
            self._strip_suffix(csv_path.name): csv_path for csv_path in uterus_dir.glob("*.csv")
        }

        common_stems = sorted(bpm_files.keys() & uterus_files.keys())
        if not common_stems:
            raise ValueError(
                "No matching CSV file pairs were found in the 'bpm' and 'uterus' directories"
            )

        for stem in common_stems:
            bpm_file = bpm_files[stem]
            uterus_file = uterus_files[stem]
            yield from self._load_csv_pair(bpm_file, uterus_file)

    def _strip_suffix(self, filename: str) -> str:
        """Remove the trailing channel suffix from a CSV filename."""

        if "_" not in filename:
            return filename
        return filename.rsplit("_", 1)[0]

    def _load_csv_pair(self, bpm_path: Path, uterus_path: Path) -> Iterator[dict[str, Any]]:
        """Combine paired BPM and uterus CSV files into unified records."""

        bpm_rows = self._read_csv(bpm_path)
        uterus_rows = self._read_csv(uterus_path)

        bpm_series = self._rows_to_series(bpm_rows, bpm_path.name)
        uterus_series = self._rows_to_series(uterus_rows, uterus_path.name)

        common_times = sorted(bpm_series.keys() & uterus_series.keys())
        if not common_times:
            raise ValueError(
                f"No overlapping timestamps between {bpm_path.name} and {uterus_path.name}"
            )

        for time_sec in common_times:
            yield self._combine_series(
                time_sec,
                bpm_series[time_sec],
                uterus_series[time_sec],
                bpm_path.name,
                uterus_path.name,
            )

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            return list(reader)

    def _rows_to_series(self, rows: list[dict[str, str]], filename: str) -> dict[float, float]:
        series: dict[float, float] = {}
        for row in rows:
            try:
                time_sec = float(row.get("time_sec", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid time value in {filename}: {row!r}") from exc

            try:
                value = float(row.get("value", "nan"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value in {filename}: {row!r}") from exc

            series[time_sec] = value

        return series

    def _combine_series(
        self,
        time_sec: float,
        bpm_value: float,
        uterus_value: float,
        bpm_name: str,
        uterus_name: str,
    ) -> dict[str, Any]:
        try:
            timestamp = int(float(time_sec) * 1000)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid timestamp derived from {bpm_name} and {uterus_name}: {time_sec!r}"
            ) from exc

        return {
            "timestamp": timestamp,
            "signals": {
                "bpm": float(bpm_value),
                "uterus": float(uterus_value),
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
            if self.config.time_step_ms > 0:
                time.sleep(self.config.time_step_ms / 1000)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Continuously print batches in JSON format."""

        for batch in self.stream_batches():
            print(json.dumps(batch, indent=2, ensure_ascii=False))

ARCHIVE_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "hypoxia" / "1"
# Adjust ``ARCHIVE_DIRECTORY`` to point at a numbered case folder (e.g. ``regular/3``).
TIME_STEP_MS = 0  # milliseconds between emitted batches
BATCH_SIZE = 120


def main() -> None:
    """Run the emulator using the configured archive and playback parameters."""

    config = RealTimeConfig(
        archive_path=ARCHIVE_DIRECTORY,
        time_step_ms=TIME_STEP_MS,
        batch_size=BATCH_SIZE,
    )
    emulator = RealTimeEmulator(config)
    emulator.run()


if __name__ == "__main__":
    main()
