"""Optional router lifespan owns one non-trading monitoring worker."""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.services.screener import ScreenerService

from .service import CloseWatchService

logger = logging.getLogger(__name__)


class ConfigRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    strategy_id: str = Field(min_length=1, max_length=120)
    enabled: StrictBool
    start_time: str = Field(default='14:50', pattern=r'^14:[3-5][0-9]$')


def build_router():
    service = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal service
        lease = thread = None
        try:
            repo = getattr(app.state, 'repo', None)
            engine = getattr(app.state, 'strategy_engine', None)
            if repo is None or engine is None:
                raise ValueError('尾盘监控所需的策略引擎和仓库尚未就绪')
            root = Path(repo.store.data_dir) / 'user_data/extensions/close_watch'
            root.mkdir(parents=True, exist_ok=True)
            lease = sqlite3.connect(root/'runner.sqlite', timeout=0)
            lease.execute('BEGIN EXCLUSIVE')
            service = CloseWatchService(repo.store.data_dir, engine, ScreenerService(repo),
                                        getattr(app.state, 'quote_service', None))
            active = service

            def run():
                while not active.stopping.is_set():
                    try:
                        active.tick()
                    except Exception:  # Isolate the optional worker.
                        logger.exception('close watch evaluation failed')
                    active.stopping.wait(15)

            thread = threading.Thread(target=run, name='close-watch', daemon=True)
            thread.start()
        except Exception:  # An unavailable optional module must not stop the app.
            logger.exception('close watch extension unavailable')
            service = None
        try:
            yield
        finally:
            if service:
                service.stopping.set()
            if thread:
                thread.join(timeout=3)
            if lease:
                lease.close()
            service = None

    router = APIRouter(prefix='/api/custom/close-watch', tags=['custom-close-watch'], lifespan=lifespan)

    def call(method, *args):
        if service is None:
            raise HTTPException(503, '尾盘监控未就绪或已有其他监控进程，请检查后端状态')
        try:
            return getattr(service, method)(*args)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else '尾盘监控数据读取或保存失败'
            raise HTTPException(409, message) from exc

    @router.get('/state')
    def get_state():
        return call('get')

    @router.put('/config')
    def configure(body: ConfigRequest):
        return call('configure', body.strategy_id, body.enabled, body.start_time)

    return router
