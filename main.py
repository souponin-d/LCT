"""Общий точка входа для запуска эмулятора и временного бэкенда одним процессом."""
from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from sys import version_info
from typing import Optional

import uvicorn

from backend.main_analyze import app
from emulator.realtime_emulator import ARCHIVE_DIRECTORY, RealTimeConfig, RealTimeEmulator


# ``slots`` parameter is available only starting from Python 3.10.
# Older Python versions used in some environments would raise an error if the
# argument is passed, so we include it conditionally.
_dataclass_kwargs = {"slots": True} if version_info >= (3, 10) else {}


@dataclass(**_dataclass_kwargs)
class PipelineSettings:
    """Настройки запуска связки эмулятора и бэкенда."""

    batch_size: int
    time_step_ms: int
    host: str
    port: int


def parse_args(argv: Optional[list[str]] = None) -> PipelineSettings:
    """Разобрать аргументы командной строки и получить настройки."""

    parser = argparse.ArgumentParser(description="Запустить эмулятор вместе с бэкендом")
    parser.add_argument("-b", "--batch-size", type=int, required=True, help="Размер батча")
    parser.add_argument(
        "-t",
        "--time-step",
        type=int,
        required=True,
        help="Интервал между батчами в миллисекундах",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Хост бэкенда")
    parser.add_argument("--port", type=int, default=8080, help="Порт бэкенда")
    args = parser.parse_args(argv)
    return PipelineSettings(
        batch_size=args.batch_size,
        time_step_ms=args.time_step,
        host=args.host,
        port=args.port,
    )


async def wait_for_server(host: str, port: int, timeout: float = 5.0) -> None:
    """Дождаться доступности сокета бэкенда."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            _reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            if loop.time() >= deadline:
                raise TimeoutError(f"Бэкенд не поднялся за {timeout} секунд")
            await asyncio.sleep(0.1)
        else:
            writer.close()
            await writer.wait_closed()
            return


async def run_pipeline(settings: PipelineSettings) -> None:
    """Поднять бэкенд и перенаправить в него батчи с эмулятора."""

    logging.info("Запускаем бэкенд на %s:%s", settings.host, settings.port)
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    try:
        await wait_for_server(settings.host, settings.port)
        emulator_config = RealTimeConfig(
            archive_path=ARCHIVE_DIRECTORY,
            time_step_ms=settings.time_step_ms,
            batch_size=settings.batch_size,
        )
        emulator = RealTimeEmulator(emulator_config)
        endpoint = f"ws://{settings.host}:{settings.port}/ws/analyze"
        logging.info("Подключаем эмулятор к %s", endpoint)
        await emulator.run_async(endpoint)
    finally:
        server.should_exit = True
        if not server_task.done():
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task


def main(argv: Optional[list[str]] = None) -> None:
    """Синхронная оболочка для запуска из командной строки."""

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    settings = parse_args(argv)
    try:
        asyncio.run(run_pipeline(settings))
    except KeyboardInterrupt:
        logging.info("Остановка по Ctrl+C")


if __name__ == "__main__":
    main()
