"""策略监控结果异步追加到自选。合并待处理成员, 不在行情线程写盘。"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from app.services import watchlist

logger = logging.getLogger(__name__)


class MonitorWatchlistSync:
    def __init__(self, rules: Callable[[], dict[str, dict]], on_change: Callable[[], None]):
        self._rules = rules
        self._on_change = on_change
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._saved: dict[str, dict] = {}
        self._pools: dict[str, dict] = {}
        self._warnings: dict[str, str] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    def warning_for(self, rule_id: str) -> str | None:
        with self._lock:
            return self._warnings.get(rule_id)

    def submit(self, candidates: list[dict]) -> None:
        with self._lock:
            active = self._rules()
            for rid in list(self._saved):
                if active.get(rid) != self._saved[rid]["rule"]:
                    self._saved.pop(rid)
                    self._pools.pop(rid, None)
            for candidate in candidates:
                rule = candidate["rule"]
                rid = rule["id"]
                if active.get(rid) != rule:
                    continue
                saved = self._saved.get(rid)
                if saved and saved["rules_version"] != candidate["rules_version"]:
                    saved = None
                    self._saved.pop(rid)
                self._pools[rid] = candidate
                if saved:
                    # 仅缓存仍在池中的成功成员; 再次入池可重新加入用户手动移除的自选。
                    saved["symbols"] = saved["symbols"] & candidate["symbols"]
                symbols = candidate["symbols"] - (saved["symbols"] if saved else set())
                pending = self._pending.get(rid)
                if pending and self._same_run(pending, candidate):
                    symbols = symbols | pending["symbols"]
                if symbols:
                    self._pending[rid] = {**candidate, "symbols": symbols}
            if self._running or not self._pending:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name="monitor-watchlist", daemon=True)
            self._thread.start()

    @staticmethod
    def _same_run(left: dict, right: dict) -> bool:
        return left["rule"] == right["rule"] and left["rules_version"] == right["rules_version"]

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._running = False
                    return
                rid = next(iter(self._pending))
                candidate = self._pending.pop(rid)
            rule = candidate["rule"]
            # 排队期间规则可能停用、删除或切换策略; 不再写入过期结果。
            if self._rules().get(rid) != rule:
                continue
            rev = watchlist.revision()
            failed = False
            warning_changed = False
            try:
                watchlist.add_strategy_results(rule["strategy_id"], candidate["strategy_name"], list(candidate["symbols"]))
                with self._lock:
                    saved = self._saved.get(rid)
                    previous = saved["symbols"] if saved and self._same_run(saved, candidate) else set()
                    latest = self._pools.get(rid)
                    current = latest["symbols"] if latest and self._same_run(latest, candidate) else set()
                    self._saved[rid] = {**candidate, "symbols": (previous | candidate["symbols"]) & current}
                    warning_changed = self._warnings.pop(rid, None) is not None
            except Exception:  # noqa: BLE001
                logger.warning("策略监控自动加入自选失败: %s", rid, exc_info=True)
                with self._lock:
                    warning_changed = rid not in self._warnings
                    self._warnings[rid] = "自动加入自选失败，将在下轮监控时重试，请检查数据目录写入权限及磁盘空间"
                    pending = self._pending.get(rid)
                    if pending is None or self._same_run(pending, candidate):
                        symbols = candidate["symbols"] | (pending["symbols"] if pending else set())
                        self._pending[rid] = {**candidate, "symbols": symbols}
                    self._running = False
                    failed = True
            if warning_changed or watchlist.revision() != rev:
                self._on_change()
            if failed:
                return  # 保留失败成员, 下一轮提交重试, 避免磁盘故障时忙循环。
