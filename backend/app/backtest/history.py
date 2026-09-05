"""Persistent strategy-backtest history.

Each completed run stores a small metadata document for listing and a gzip
payload for restoring the exact result and configuration shown to the user.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HISTORY_SCHEMA_VERSION = 1
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SUMMARY_STATS = (
    "total_return",
    "annual_return",
    "max_drawdown",
    "sharpe",
    "win_rate",
    "profit_factor",
    "n_trades",
    "n_candidates",
)


def _history_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "backtest_results" / "strategy"


def _safe_run_id(run_id: object) -> str:
    value = str(run_id or "")
    if not _RUN_ID_RE.fullmatch(value):
        raise ValueError("invalid backtest run id")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def persist_strategy_backtest_result(data_dir: Path, result: dict[str, Any]) -> dict[str, Any] | None:
    """Persist a successful result and return its list metadata."""
    if not isinstance(result, dict) or result.get("error"):
        return None
    run_id = _safe_run_id(result.get("run_id"))
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    strategy_info = (
        result.get("strategy_info")
        if isinstance(result.get("strategy_info"), dict)
        else {}
    )
    strategy_id = str(config.get("strategy_id") or strategy_info.get("id") or "")
    if not strategy_id:
        return None

    created_at = datetime.now(UTC).isoformat()
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    metadata = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "strategy_id": strategy_id,
        "strategy_name": str(strategy_info.get("name") or strategy_id),
        "strategy_version": str(strategy_info.get("version") or "1.0.0"),
        "code_hash": str(strategy_info.get("code_hash") or ""),
        "asset_type": str(config.get("asset_type") or "stock"),
        "mode": str(config.get("mode") or "position"),
        "start": str(config.get("start") or ""),
        "end": str(config.get("end") or ""),
        "trade_count": len(result.get("trades") or []),
        "stats": {key: stats.get(key) for key in _SUMMARY_STATS if stats.get(key) is not None},
    }
    wrapper = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "created_at": created_at,
        "result": result,
    }
    root = _history_dir(data_dir)
    payload = gzip.compress(
        json.dumps(wrapper, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"),
        compresslevel=6,
    )
    _atomic_write(root / f"{run_id}.json.gz", payload)
    _atomic_write(
        root / f"{run_id}.meta.json",
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )
    return metadata


def list_strategy_backtest_history(
    data_dir: Path,
    *,
    strategy_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    root = _history_dir(data_dir)
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in root.glob("*.meta.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(item, dict):
            continue
        if strategy_id and item.get("strategy_id") != strategy_id:
            continue
        items.append(item)
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items[: max(1, min(int(limit), 200))]


def get_strategy_backtest_history(data_dir: Path, run_id: str) -> dict[str, Any] | None:
    safe_id = _safe_run_id(run_id)
    path = _history_dir(data_dir) / f"{safe_id}.json.gz"
    if not path.is_file():
        return None
    try:
        wrapper = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("result"), dict):
        return None
    return wrapper
