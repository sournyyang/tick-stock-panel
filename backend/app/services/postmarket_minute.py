"""交易日盘后全市场当日分钟采集：检查点、延后重试及量额覆盖核对。"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from datetime import date, datetime, timedelta
from datetime import time as wall_time
from pathlib import Path

import polars as pl

from app.market_time import CN_TZ, cn_now
from app.services import kline_sync, minute_adjust, preferences, trading_day
from app.services.fs_utils import atomic_write_text
from app.tickflow.capabilities import Cap

logger = logging.getLogger(__name__)
START_TIME = wall_time(15, 45)
RETRY_DELAY = timedelta(minutes=15)
MAX_ATTEMPTS = 3
BATCH_SIZE = 8
ROUND_BUDGET_S = 600
MIN_UNIVERSE_SYMBOLS = 4000
_RUN_LOCK = threading.Lock()
_STOCK = re.compile(r"^(?:(?:60|68)\d{4}\.SH|(?:00|30)\d{4}\.SZ|(?:43|83|87|88|92)\d{4}\.BJ)$")
_COLS = ("symbol", "datetime", "open", "high", "low", "close", "volume", "amount")


def _path(data_dir: Path, day: date) -> Path:
    return Path(data_dir) / "postmarket_minute" / f"{day.isoformat()}.json"


def _load(data_dir: Path, day: date) -> dict:
    path = _path(data_dir, day)
    if not path.exists():
        return {"date": day.isoformat(), "symbols": {}}
    # 损坏检查点不得被静默重置，否则已采集标的会再次请求整个市场。
    return json.loads(path.read_text(encoding="utf-8"))


def session_status(data_dir: Path, day: date, *, include_symbols: bool = False) -> dict:
    report = _load(data_dir, day)
    report.setdefault("status", "not_started")
    if include_symbols:
        return report
    return {k: v for k, v in report.items() if k != "symbols"}


def ready_to_continue(data_dir: Path, day: date) -> bool:
    """有可立即采集的当天标的时直接续跑；冷却时释放任务槽。"""
    now = cn_now()
    if now.date() != day or gate_reason(now):
        return False
    report = _load(data_dir, day)
    if report.get("status") != "pending":
        return False
    if report.get("cooldown_until") and datetime.fromisoformat(report["cooldown_until"]) > now:
        return False
    return any(
        r["status"] not in ("complete", "suspended") and r["attempts"] < MAX_ATTEMPTS
        and (not r.get("next_retry_at") or datetime.fromisoformat(r["next_retry_at"]) <= now)
        for r in report["symbols"].values()
    )


def _save(data_dir: Path, report: dict, now: datetime) -> dict:
    records = list(report["symbols"].values())
    counts = {s: sum(r["status"] == s for r in records)
              for s in ("complete", "suspended", "failed", "pending", "unverified")}
    expected = len(records) - counts["suspended"]
    report.update(
        expected_symbols=expected, universe_symbols=len(records),
        data_symbols=counts["complete"] + counts["unverified"],
        coverage_pct=round(100 * counts["complete"] / expected, 2) if expected else 100.0,
        **{f"{s}_symbols": n for s, n in counts.items()},
        updated_at=now.isoformat(),
    )
    retryable = [r for r in records if r["status"] not in ("complete", "suspended")
                 and r["attempts"] < MAX_ATTEMPTS]
    report["status"] = ("complete" if counts["complete"] == expected
                        else "pending" if retryable else "incomplete")
    due = [r["next_retry_at"] for r in retryable if r.get("next_retry_at")]
    report["next_retry_at"] = min(due) if due else None
    path = _path(data_dir, date.fromisoformat(report["date"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(report, ensure_ascii=False))
    return {k: v for k, v in report.items() if k != "symbols"}


def _daily_reference(data_dir: Path, day: date) -> dict[str, dict]:
    paths = sorted((Path(data_dir) / "kline_daily" / f"date={day}").glob("*.parquet"))
    if not paths:
        return {}
    df = pl.read_parquet(paths, columns=["symbol", "volume", "amount", "close"])
    if df["symbol"].n_unique() != df.height:
        raise ValueError("当日日K存在重复股票，无法核对分钟覆盖")
    return {row["symbol"]: row for row in df.to_dicts()}


def validate_symbol(frame: pl.DataFrame, reference: dict | None) -> str | None:
    """不要求每股恰好240根：无成交分钟可缺行，完整性由当日日K量额核对。"""
    if frame.is_empty():
        return "empty"
    if not set(_COLS).issubset(frame.columns):
        return "missing_columns"
    if frame["datetime"].n_unique() != frame.height:
        return "duplicate_minutes"
    for name in _COLS[2:]:
        values = frame[name].to_list()
        if any(v is None or not math.isfinite(v) or v < 0
               or (name in _COLS[2:6] and v == 0) for v in values):
            return "invalid_values"
    if not frame.filter(
        (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
    ).is_empty():
        return "invalid_ohlc"
    if reference is None:
        return "daily_reference_missing"
    for name, floor in (("volume", 0.02), ("amount", 1.0)):
        expected = reference.get(name)
        if expected is None or not math.isfinite(expected) or expected <= 0:
            return "daily_reference_invalid"
        if abs(frame[name].sum() - expected) > max(floor, expected * 0.005):
            return f"{name}_mismatch"
    close = reference.get("close")
    if close is None or not math.isfinite(close) or close <= 0:
        return "daily_reference_invalid"
    if abs(frame.sort("datetime")["close"][-1] - close) > 0.011:
        return "close_mismatch"
    return None


def _today_frame(frame: pl.DataFrame, day: date, provider: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    frame = kline_sync._enforce_minute_beijing_wallclock(frame, source=provider)
    if not {"symbol", "datetime"}.issubset(frame.columns):
        return pl.DataFrame()
    frame = frame.with_columns(
        pl.col("datetime").cast(pl.Datetime("us")),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in _COLS[2:] if c in frame.columns],
    )
    return frame.filter(
        (pl.col("datetime").dt.date() == day)
        & (((pl.col("datetime").dt.time() >= wall_time(9, 30))
            & (pl.col("datetime").dt.time() <= wall_time(11, 30)))
           | ((pl.col("datetime").dt.time() >= wall_time(13, 0))
              & (pl.col("datetime").dt.time() <= wall_time(15, 0))))
    )


def _fetch(symbols: list[str], repo, capset, day: date) -> pl.DataFrame:
    name = preferences.get_minute_data_provider()
    start = datetime.combine(day, wall_time(9, 30), CN_TZ)
    end = datetime.combine(day, wall_time(15, 0), CN_TZ)
    if name != "tickflow":
        provider, fallback, error = kline_sync._resolve_minute_provider(name)
        if fallback:
            raise RuntimeError(error or f"分钟源 {name} 不可用")
        postmarket = getattr(provider, "get_minute_postmarket", None)
        if callable(postmarket):
            return postmarket(symbols, day)
        return provider.get_minute(symbols, start_time=start, end_time=end,
                                   asset_type="stock", freq="1m")
    if capset is None or not capset.has(Cap.KLINE_MINUTE_BATCH):
        raise RuntimeError("分钟数据能力不可用")
    return kline_sync.sync_minute_batch(
        symbols, start_time=start, end_time=end, batch_size=BATCH_SIZE, rpm=30,
        raw_basis=minute_adjust.minute_basis_is_raw(repo.store.data_dir),
    )


def _publish(repo) -> None:
    from app.api.data import invalidate_data_cache
    from app.jobs.daily_pipeline import _refresh_single_view

    _refresh_single_view(repo, "kline_minute")
    invalidate_data_cache("minute")


def run_session(repo, capset, symbols: list[str], *, now: datetime | None = None,
                on_progress=None) -> dict:
    """单轮有界执行，未采集标的优先；失败至少等待15分钟，最多三次。"""
    now = (now or cn_now()).astimezone(CN_TZ)
    reason = gate_reason(now)
    if reason:
        return {"status": reason}
    if not _RUN_LOCK.acquire(blocking=False):
        return {"status": "running"}
    try:
        return _run(repo, capset, symbols, now, on_progress)
    finally:
        _RUN_LOCK.release()


def gate_reason(now: datetime | None = None) -> str | None:
    now = (now or cn_now()).astimezone(CN_TZ)
    if not preferences.get_minute_sync_enabled():
        return "disabled"
    if now.time() < START_TIME:
        return "before_close"
    verdict = trading_day.is_trading_day(now)
    return None if verdict is True else "non_trading_day" if verdict is False else "calendar_unknown"


def _run(repo, capset, symbols: list[str], now: datetime, on_progress) -> dict:
    day, data_dir = now.date(), repo.store.data_dir
    reference = _daily_reference(data_dir, day)
    if not reference:
        return {"date": str(day), "status": "awaiting_daily"}
    report = _load(data_dir, day)
    report["provider"] = preferences.get_minute_data_provider()
    # 日K与已有股票池取并集，不能把较窄的 instruments 当作全市场。
    universe = sorted(s for s in set(symbols) | set(reference) if _STOCK.fullmatch(s))
    if not universe:
        return {"date": str(day), "status": "empty_universe"}
    if len(universe) < MIN_UNIVERSE_SYMBOLS:
        # 市场规模显著偏小（例如只有自选/单市场维表）时不得伪装全市场完成。
        return {"date": str(day), "status": "universe_incomplete", "universe_symbols": len(universe)}
    for symbol in universe:
        report["symbols"].setdefault(symbol, {"status": "pending", "attempts": 0})
    path = data_dir / "kline_minute" / f"date={day}" / "part.parquet"
    existing = pl.read_parquet(path) if path.exists() else pl.DataFrame()
    frames = ({key[0]: frame for key, frame in existing.partition_by("symbol", as_dict=True).items()}
              if not existing.is_empty() else {})
    for symbol, record in report["symbols"].items():
        daily = reference.get(symbol)
        if daily and daily["volume"] == 0 and daily["amount"] == 0:
            record.update(status="suspended", reason="daily_zero_volume_amount")
        elif symbol in frames and validate_symbol(_today_frame(frames[symbol], day, report["provider"]), daily) is None:
            record.update(status="complete", reason=None)
        elif record["status"] in ("complete", "suspended"):
            record.update(status="pending", attempts=0, reason="partition_changed")
        elif record["status"] == "unverified" and symbol not in frames:
            record.update(status="pending", reason="partition_missing")
    _save(data_dir, report, now)
    if report.get("cooldown_until") and datetime.fromisoformat(report["cooldown_until"]) > now:
        return session_status(data_dir, day)
    eligible = [s for s, r in report["symbols"].items()
                if r["status"] not in ("complete", "suspended") and r["attempts"] < MAX_ATTEMPTS
                and (not r.get("next_retry_at") or datetime.fromisoformat(r["next_retry_at"]) <= now)]
    eligible.sort(key=lambda s: (report["symbols"][s]["attempts"], s))
    started, empty_batches = time.monotonic(), 0
    for offset in range(0, len(eligible), BATCH_SIZE):
        # 不跨北京午夜继续采旧日期；下一交易日也不自动回补这个检查点。
        if time.monotonic() - started >= ROUND_BUDGET_S or cn_now().date() != day:
            break
        if on_progress:
            on_progress("sync_minute", int(100 * offset / max(1, len(eligible))),
                        f"当日分钟K {offset}/{len(eligible)}，完整覆盖 {report['complete_symbols']}/{report['expected_symbols']}")
        batch = eligible[offset:offset + BATCH_SIZE]
        error = None
        try:
            frame = _today_frame(_fetch(batch, repo, capset, day), day, report["provider"])
        except Exception as exc:  # noqa: BLE001 — 单批失败持久化，后续轮次重试
            logger.warning("postmarket minute batch failed (%d symbols): %s", len(batch), exc)
            error, frame = type(exc).__name__, pl.DataFrame()
        parts = ({key[0]: part for key, part in frame.partition_by("symbol", as_dict=True).items()}
                 if not frame.is_empty() else {})
        attempted_at = now + timedelta(seconds=time.monotonic() - started)
        accepted = []
        for symbol in batch:
            part = parts.get(symbol, pl.DataFrame())
            reason = error or validate_symbol(part, reference.get(symbol))
            record = report["symbols"][symbol]
            record["attempts"] += 1
            record.update(status="complete" if reason is None else "unverified"
                          if reason == "daily_reference_missing" else "failed",
                          reason=reason, rows=part.height,
                          next_retry_at=(attempted_at + RETRY_DELAY).isoformat())
            if reason in (None, "daily_reference_missing"):
                accepted.append(part.select(_COLS))
        if accepted:
            # 无日K参考的有效分钟也保留，但覆盖报告仍标为未核对，不能冒充完成。
            with repo._write_lock:
                kline_sync._write_minute_partition(pl.concat(accepted), data_dir / "kline_minute")
            _publish(repo)
        _save(data_dir, report, attempted_at)
        empty_batches = 0 if accepted else empty_batches + 1
        if empty_batches >= 2:
            # 连续整批失败先冷却，避免在上游失联时扫描数千次。
            report["cooldown_until"] = (attempted_at + RETRY_DELAY).isoformat()
            break
        if offset + BATCH_SIZE < len(eligible):
            time.sleep(1)
    result = _save(data_dir, report, now + timedelta(seconds=time.monotonic() - started))
    if on_progress:
        on_progress("done", 100, f"当日分钟K：完整覆盖 {result['complete_symbols']}/{result['expected_symbols']} "
                    f"({result['coverage_pct']}%)，失败 {result['failed_symbols']}，待采 {result['pending_symbols']}")
    logger.info("postmarket minute %s: coverage=%s%% complete=%d expected=%d status=%s",
                day, result["coverage_pct"], result["complete_symbols"], result["expected_symbols"], result["status"])
    return result
