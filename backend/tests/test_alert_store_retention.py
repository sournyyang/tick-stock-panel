from __future__ import annotations

import json
import time

import pytest

from app.services import alert_store


@pytest.fixture(autouse=True)
def _isolate_store(monkeypatch):
    monkeypatch.setattr(alert_store, "_write_count", 0)
    monkeypatch.setattr(alert_store, "_publish_events", lambda events: None)


@pytest.mark.parametrize("batch", [False, True])
def test_rolling_cleanup_keeps_thirty_day_boundary(tmp_path, monkeypatch, batch):
    now = 1_800_000_000
    monkeypatch.setattr(time, "time", lambda: now)
    now_ms = now * 1000
    day_ms = 86_400_000
    events = [
        {"ts": now_ms - 30 * day_ms - 1, "rule_id": "expired"},
        {"ts": now_ms - 30 * day_ms, "rule_id": "boundary"},
        {"ts": now_ms - 29 * day_ms, "rule_id": "day29"},
        {"ts": now_ms - 8 * day_ms, "rule_id": "day8"},
        *[{"ts": now_ms - i, "rule_id": f"recent{i}"} for i in range(16)],
    ]
    if batch:
        alert_store.append_many(tmp_path, events)
    else:
        for event in events:
            alert_store.append(tmp_path, event)

    stored = [json.loads(line) for line in
              (tmp_path / "user_data" / "alerts.jsonl").read_text().splitlines()]
    assert {event["rule_id"] for event in stored} == {
        event["rule_id"] for event in events if event["rule_id"] != "expired"
    }
    recent = alert_store.list_recent(tmp_path)
    assert len(recent) == 19
    assert recent[-1]["rule_id"] == "boundary"


def test_rolling_cleanup_keeps_latest_hundred_thousand_records(tmp_path, monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr(time, "time", lambda: now)
    now_ms = now * 1000
    # Reverse timestamp order verifies retention selects newest events, not file tail.
    events = [{"ts": now_ms - i, "rule_id": str(i)} for i in range(100_001)]
    events.insert(0, {"ts": now_ms - 31 * 86_400_000, "rule_id": "expired"})
    alert_store.append_many(tmp_path, events)

    assert alert_store.count(tmp_path) == 100_000
    recent = alert_store.list_recent(tmp_path)
    assert len(recent) == 100_000
    assert recent[0]["rule_id"] == "0"
    assert recent[-1]["rule_id"] == "99999"
