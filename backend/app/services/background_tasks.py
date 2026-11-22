"""Own fire-and-forget tasks until completion and retrieve their exceptions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)
_tasks: set[asyncio.Task] = set()


def create_background_task(coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task:
    task = asyncio.create_task(coroutine)
    _tasks.add(task)
    task.add_done_callback(_finished)
    return task


def _finished(task: asyncio.Task) -> None:
    _tasks.discard(task)
    if not task.cancelled():
        error = task.exception()
        if error is not None:
            # Provider exceptions can contain credential-bearing URLs.
            logger.error("后台任务异常终止 (%s)", type(error).__name__)
