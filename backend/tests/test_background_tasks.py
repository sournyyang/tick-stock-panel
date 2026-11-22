import asyncio
import gc
import weakref

import pytest

from app.services import background_tasks


@pytest.mark.asyncio
async def test_task_is_owned_until_completion():
    started = asyncio.Event()
    release = asyncio.Event()

    async def work():
        started.set()
        await release.wait()

    task = background_tasks.create_background_task(work())
    ref = weakref.ref(task)
    await started.wait()
    del task
    gc.collect()
    assert ref() is not None
    release.set()
    await ref()
    await asyncio.sleep(0)
    assert not background_tasks._tasks


@pytest.mark.asyncio
async def test_failure_is_observed_without_logging_credentials(caplog):
    async def fail():
        raise RuntimeError("https://example.invalid?token=private")

    task = background_tasks.create_background_task(fail())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert task.done()
    assert "RuntimeError" in caplog.text
    assert "private" not in caplog.text
    assert task not in background_tasks._tasks


@pytest.mark.asyncio
async def test_cancelled_task_is_released_without_error(caplog):
    task = background_tasks.create_background_task(asyncio.sleep(100))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert task not in background_tasks._tasks
    assert not caplog.records
