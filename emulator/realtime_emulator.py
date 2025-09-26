"""Утилита для эмуляции потоковой передачи данных в реальном времени из архива.

Эмулятор воспроизводит исторические измерения, сохранённые в CSV, и формирует батчи
с заданной периодичностью. Это позволяет разрабатывать остальные части системы на
детерминированных данных до интеграции с реальным оборудованием.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator


# ``dataclass`` получила параметр ``slots`` только в Python 3.10.
# Более старые интерпретаторы (например, Python 3.9) выбрасывают ``TypeError`` при
# передаче аргумента, поэтому враппер ниже включает слоты там, где это возможно, и
# остаётся совместимым с Python 3.9.
def dataclass_with_optional_slots(_cls=None, /, *, slots: bool = True, **kwargs):
    """Вернуть декоратор ``dataclass``, устанавливающий ``slots`` при поддержке."""

    if sys.version_info >= (3, 10) and slots:
        kwargs["slots"] = True

    def wrap(cls):
        return dataclass(cls, **kwargs)

    if _cls is None:
        return wrap
    return wrap(_cls)


@dataclass_with_optional_slots
class RealTimeConfig:
    """Параметры запуска эмулятора реального времени."""

    archive_path: Path
    time_step_ms: int
    batch_size: int


class RealTimeEmulator:
    """Воспроизводить архивные данные так, будто они поступают онлайн."""

    def __init__(self, config: RealTimeConfig) -> None:
        self.config = config
        self._records = tuple(self._load_records(config.archive_path))
        if not self._records:
            raise ValueError(
                "Архив пуст – заполните `data/archive` CSV-файлами перед запуском."
            )
        self._cursor = 0

    # ------------------------------------------------------------------
    # Парсинг CSV
    def _load_records(self, path: Path) -> Iterator[dict[str, Any]]:
        """Загрузить архив из каталогов ``bpm`` и ``uterus`` с CSV-файлами."""

        if not path.is_dir():
            raise ValueError(
                "Путь архива должен указывать на каталог с подпапками 'bpm' и 'uterus'"
            )

        bpm_dir = path / "bpm"
        uterus_dir = path / "uterus"

        if not bpm_dir.is_dir() or not uterus_dir.is_dir():
            raise ValueError(
                "Каталог архива обязан содержать подпапки 'bpm' и 'uterus' с CSV-файлами"
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
                "Не найдено совпадающих пар CSV-файлов в папках 'bpm' и 'uterus'"
            )

        for stem in common_stems:
            bpm_file = bpm_files[stem]
            uterus_file = uterus_files[stem]
            yield from self._load_csv_pair(bpm_file, uterus_file)

    def _strip_suffix(self, filename: str) -> str:
        """Удалить суффикс канала из имени CSV-файла."""

        if "_" not in filename:
            return filename
        return filename.rsplit("_", 1)[0]

    def _load_csv_pair(self, bpm_path: Path, uterus_path: Path) -> Iterator[dict[str, Any]]:
        """Объединить пары CSV-файлов BPM и uterus в единые записи."""

        bpm_rows = self._read_csv(bpm_path)
        uterus_rows = self._read_csv(uterus_path)

        bpm_series = self._rows_to_series(bpm_rows, bpm_path.name)
        uterus_series = self._rows_to_series(uterus_rows, uterus_path.name)

        common_times = sorted(bpm_series.keys() & uterus_series.keys())
        if not common_times:
            raise ValueError(
                f"Отсутствует пересечение временных меток между {bpm_path.name} и {uterus_path.name}"
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
                raise ValueError(f"Недопустимое значение времени в {filename}: {row!r}") from exc

            try:
                value = float(row.get("value", "nan"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Недопустимое числовое значение в {filename}: {row!r}") from exc

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
                f"Недопустимая метка времени из {bpm_name} и {uterus_name}: {time_sec!r}"
            ) from exc

        return {
            "timestamp": timestamp,
            "signals": {
                "bpm": float(bpm_value),
                "uterus": float(uterus_value),
            },
        }

    # ------------------------------------------------------------------
    # Формирование батча
    def _next_chunk(self) -> list[dict[str, Any]]:
        chunk: list[dict[str, Any]] = []
        for _ in range(self.config.batch_size):
            chunk.append(self._records[self._cursor])
            self._cursor = (self._cursor + 1) % len(self._records)
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
        """Выдавать сформированные батчи с заданной периодичностью."""

        while True:
            chunk = self._next_chunk()
            batch = self._format_batch(chunk)
            yield batch
            if self.config.time_step_ms > 0:
                time.sleep(self.config.time_step_ms / 1000)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Непрерывно выводить батчи в формате JSON."""

        for batch in self.stream_batches():
            print(json.dumps(batch, indent=2, ensure_ascii=False))

# ----------------------------------------------------------------------
# Конфигурация
ARCHIVE_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "hypoxia" / "1"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Разобрать аргументы командной строки для настройки эмулятора."""

    parser = argparse.ArgumentParser(description="Эмулятор потоковой передачи данных")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=True,
        help="Количество точек в батче",
    )
    parser.add_argument(
        "-t",
        "--time-step",
        type=int,
        required=True,
        help="Периодичность отправки батчей в миллисекундах",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Запустить эмулятор с параметрами из командной строки."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = RealTimeConfig(
        archive_path=ARCHIVE_DIRECTORY,
        time_step_ms=args.time_step,
        batch_size=args.batch_size,
    )
    emulator = RealTimeEmulator(config)
    emulator.run()


if __name__ == "__main__":
    main()
