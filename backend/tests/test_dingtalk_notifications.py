"""钉钉协议、配置及通知接线回归；所有网络和用户文件均隔离。"""
from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import secrets_store
from app.api import settings
from app.jobs import daily_pipeline
from app.services import dingtalk_adapter, preferences, quote_service
from app.strategy import monitor_rules

URL = "https://oapi.dingtalk.com/robot/send?access_token=test-token"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, "_path", lambda: tmp_path / "preferences.json")
    monkeypatch.setattr(secrets_store, "_path", lambda: tmp_path / "secrets.json")
    preferences._invalidate_cache()
    monkeypatch.setattr(secrets_store, "_cache", None)
    monkeypatch.setattr(secrets_store, "_cache_sig", None)
    yield
    preferences._invalidate_cache()


@pytest.mark.parametrize("url", [
    "", "http://oapi.dingtalk.com/robot/send?access_token=x",
    "https://oapi.dingtalk.com.evil.test/robot/send?access_token=x",
    "https://user@oapi.dingtalk.com/robot/send?access_token=x",
    "https://oapi.dingtalk.com:8443/robot/send?access_token=x",
    "https://oapi.dingtalk.com/robot/send/extra?access_token=x",
    "https://oapi.dingtalk.com/robot/send", URL + "#fragment",
    "https://oapi.dingtalk.com/robot/send?access_token=",
    URL + "&access_token=second", URL + "&sign=stale&timestamp=0",
])
def test_invalid_url_never_posts(url, monkeypatch):
    monkeypatch.setattr("httpx.post", lambda *a, **k: pytest.fail("must not post"))
    assert not dingtalk_adapter.is_valid_url(url)
    assert not dingtalk_adapter.send_text(url, "标题", "正文")


def test_signed_text_matches_dingtalk_protocol(monkeypatch):
    calls = []
    monkeypatch.setattr(dingtalk_adapter.time, "time", lambda: 1_700_000_000.125)

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json={"errcode": 0, "errmsg": "ok"})

    monkeypatch.setattr("httpx.post", post)
    assert dingtalk_adapter.send_text(URL, "价格预警", "浦发银行触发", "SEC-test")
    query = parse_qs(urlparse(calls[0][0]).query)
    expected = base64.b64encode(hmac.new(
        b"SEC-test", b"1700000000125\nSEC-test", hashlib.sha256,
    ).digest()).decode()
    assert query == {"access_token": ["test-token"], "timestamp": ["1700000000125"], "sign": [expected]}
    assert calls[0][1]["json"] == {
        "msgtype": "text", "text": {"content": "价格预警\n浦发银行触发"},
    }
    assert calls[0][1]["timeout"] == 5.0


def test_unsigned_markdown_and_utf8_truncation(monkeypatch):
    calls = []
    monkeypatch.setattr("httpx.post", lambda url, **kw: calls.append((url, kw)) or
                        httpx.Response(200, json={"errcode": 0}))
    assert dingtalk_adapter.send_markdown(URL, "每日复盘", "中文📈" * 10000)
    url, kwargs = calls[0]
    assert url == URL
    payload = kwargs["json"]
    assert payload["msgtype"] == "markdown"
    assert payload["markdown"]["title"] == "每日复盘"
    text = payload["markdown"]["text"]
    assert text.startswith("## 每日复盘\n\n")
    assert len(text.encode()) <= dingtalk_adapter.MAX_MESSAGE_BYTES
    assert text.endswith("报告过长，请在应用内查看完整内容。")
    assert "\ufffd" not in text


@pytest.mark.parametrize("response", [
    httpx.Response(200, json={"errcode": 310000, "errmsg": "sign not match"}),
    httpx.Response(200, json={"errcode": 130101}),
    httpx.Response(200, json={}), httpx.Response(200, json=[]),
    httpx.Response(200, text="not json"), httpx.Response(400), httpx.Response(429),
    httpx.Response(200, json={"errcode": False}),
])
def test_rejections_are_fail_closed_without_retry(response, monkeypatch):
    calls = []
    monkeypatch.setattr("httpx.post", lambda *a, **kw: calls.append(kw) or response)
    monkeypatch.setattr(dingtalk_adapter.time, "sleep", lambda _: pytest.fail("no retry"))
    assert not dingtalk_adapter.send_text(URL, "标题", "正文")
    assert len(calls) == 1


def test_network_retries_do_not_log_credentials(monkeypatch, caplog):
    calls, sleeps = [], []

    def post(url, **kwargs):
        calls.append(url)
        raise httpx.ConnectError(f"secret url: {url} SEC-test")

    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr(dingtalk_adapter.time, "sleep", sleeps.append)
    assert not dingtalk_adapter.send_text(URL, "标题", "正文", "SEC-test")
    assert len(calls) == 3
    assert sleeps == [1, 2]
    assert "test-token" not in caplog.text
    assert "SEC-test" not in caplog.text


def test_server_failure_can_retry_and_recover(monkeypatch):
    responses = iter([httpx.Response(503), httpx.Response(200, json={"errcode": 0})])
    monkeypatch.setattr("httpx.post", lambda *a, **kw: next(responses))
    monkeypatch.setattr(dingtalk_adapter.time, "sleep", lambda _: None)
    assert dingtalk_adapter.send_text(URL, "标题", "正文")


def test_single_attempt_network_failure_never_sleeps(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(args)
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr(dingtalk_adapter.time, "sleep", lambda _: pytest.fail("no backoff for diagnostic"))
    assert not dingtalk_adapter.send_text(URL, "标题", "正文", max_attempts=1)
    assert len(calls) == 1


def test_configuration_roundtrip_secret_preserve_and_clear():
    result = settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url=URL, secret="SEC-test"))
    assert result == {"dingtalk_webhook_url": URL, "dingtalk_webhook_secret_set": True}
    assert preferences.get_dingtalk_webhook_url() == URL
    assert "dingtalk_webhook_secret" not in preferences.load()
    assert secrets_store.get_dingtalk_webhook_secret() == "SEC-test"
    result = settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url=URL))
    assert result["dingtalk_webhook_secret_set"] is True
    result = settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url=URL, secret=""))
    assert result["dingtalk_webhook_secret_set"] is False
    settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url=URL, secret="SEC-test"))
    result = settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url=""))
    assert result == {"dingtalk_webhook_url": "", "dingtalk_webhook_secret_set": False}


def test_invalid_config_does_not_overwrite_saved_configuration():
    settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url=URL, secret="SEC-test"))
    with pytest.raises(HTTPException):
        settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url="https://evil.test", secret=""))
    assert preferences.get_dingtalk_webhook_url() == URL
    assert secrets_store.get_dingtalk_webhook_secret() == "SEC-test"


def test_api_contract_and_single_attempt_test_send(monkeypatch):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(settings.router)
    client = TestClient(app)
    response = client.put("/api/settings/preferences/dingtalk-webhook", json={"url": URL, "secret": "SEC-test"})
    assert response.status_code == 200
    assert "SEC-test" not in response.text
    prefs = client.get("/api/settings/preferences").json()
    assert prefs["dingtalk_webhook_url"] == URL
    assert prefs["dingtalk_webhook_secret_set"] is True
    assert "dingtalk_webhook_secret" not in prefs
    config = client.get("/api/settings/preferences/dingtalk-webhook")
    assert config.json() == {"url": URL, "secret": "SEC-test"}
    assert config.headers["cache-control"] == "no-store"
    calls = []
    monkeypatch.setattr(dingtalk_adapter, "send_text_result", lambda *a, **kw:
                        calls.append((a, kw)) or dingtalk_adapter.SendResult(True, ""))
    response = client.post("/api/settings/preferences/webhook-test", json={"channel": "dingtalk"})
    assert response.json()["ok"] is True
    assert calls[0][0][0] == URL
    assert calls[0][0][3] == "SEC-test"
    assert calls[0][1] == {"max_attempts": 1}
    assert client.put("/api/settings/preferences/dingtalk-webhook", json={"url": "file:///bad"}).status_code == 400


def test_test_send_unconfigured_invalid_and_failed(monkeypatch):
    req = settings.WebhookTestIn(channel="dingtalk")
    assert "尚未配置" in settings.test_webhook(req)["detail"]
    monkeypatch.setattr(preferences, "get_dingtalk_webhook_url", lambda: "https://evil.test")
    assert "地址非法" in settings.test_webhook(req)["detail"]
    monkeypatch.setattr(preferences, "get_dingtalk_webhook_url", lambda: URL)
    monkeypatch.setattr(dingtalk_adapter, "send_text_result", lambda *a, **kw:
                        dingtalk_adapter.SendResult(False, "钉钉推送失败：加签密钥不匹配"))
    assert settings.test_webhook(req) == {"ok": False, "detail": "钉钉推送失败：加签密钥不匹配"}


@pytest.mark.parametrize(("message", "hint"), [
    ("sign not match", "签名不匹配"),
    ("错误描述:机器人发送签名不匹配;解决方案:请确认签名和生成签名的时间戳", "签名不匹配"),
    ("invalid timestamp", "时间戳"),
    ("keywords not in content", "关键词"),
    ("ip 192.0.2.1 not in whitelist", "IP 白名单"),
    ("unknown rejection", "310000"),
])
def test_diagnostic_distinguishes_security_rejections_without_exposing_response(message, hint, monkeypatch, caplog):
    monkeypatch.setattr("httpx.post", lambda *a, **kw: httpx.Response(200, json={
        "errcode": 310000, "errmsg": f"{message} {URL} SEC-test",
    }))
    result = dingtalk_adapter.send_text_result(URL, "标题", "正文", "SEC-test", max_attempts=1)
    assert result.ok is False
    assert hint in result.detail
    for credential in ("test-token", "SEC-test", "192.0.2.1"):
        assert credential not in result.detail + caplog.text


def test_diagnostic_timeout_is_actionable_and_single_attempt(monkeypatch):
    calls = []

    def post(*a, **kw):
        calls.append(a)
        raise httpx.ConnectTimeout(f"{URL} SEC-test")

    monkeypatch.setattr("httpx.post", post)
    result = dingtalk_adapter.send_text_result(URL, "标题", "正文", max_attempts=1)
    assert not result.ok
    assert "连接超时" in result.detail
    assert "test-token" not in result.detail
    assert len(calls) == 1


def test_channel_persistence_and_legacy_defaults():
    assert preferences.get_dingtalk_webhook_url() == ""
    assert secrets_store.get_dingtalk_webhook_secret() == ""
    assert preferences.set_webhook_default_channels(["dingtalk", "sms", "dingtalk"]) == ["dingtalk"]
    assert preferences.set_review_push_channels(["dingtalk", "sms"]) == ["dingtalk"]
    rule = {"id": "r1", "name": "test", "type": "price", "conditions": [], "webhook_channels": ["dingtalk", "sms"]}
    assert monitor_rules.normalize(rule)["webhook_channels"] == ["dingtalk"]
    legacy = {**rule, "webhook_enabled": True}
    legacy.pop("webhook_channels")
    assert monitor_rules.normalize(legacy)["webhook_channels"] == ["feishu", "wecom"]


@pytest.mark.parametrize("selected,configured,expected", [(True, True, 1), (False, True, 0), (True, False, 0)])
def test_monitor_enqueues_only_selected_configured_channel(monkeypatch, selected, configured, expected):
    from app.services.quote_service import QuoteService

    calls = []

    class Executor:
        def submit(self, fn, *args):
            calls.append((fn, args))

    if configured:
        settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url=URL, secret="SEC-test"))
    monkeypatch.setattr(quote_service, "_WEBHOOK_EXECUTOR", Executor())
    monkeypatch.setattr("httpx.post", lambda *a, **kw: pytest.fail("must enqueue, not post"))
    engine = type("Engine", (), {"rules": {"r1": {"webhook_channels": ["dingtalk"] if selected else []}}})()
    event = {"rule_id": "r1", "source": "ladder", "symbol": "600000.SH", "name": "浦发银行", "message": "炸板预警"}
    QuoteService._maybe_send_webhook(object.__new__(QuoteService), [event], engine)
    assert len(calls) == expected
    if expected:
        assert calls[0][0] is dingtalk_adapter.send_text
        assert calls[0][1][0] == URL
        assert calls[0][1][1] == "连板梯队"
        assert "炸板预警" in calls[0][1][2]
        assert calls[0][1][3] == "SEC-test"


def test_review_uses_markdown_and_preserves_other_channels_on_failure(monkeypatch):
    settings.update_dingtalk_webhook(settings.DingtalkWebhookPrefsIn(url=URL, secret="SEC-test"))
    preferences.set_wecom_webhook_url("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test")
    preferences.set_review_push_channels(["dingtalk", "wecom"])
    calls = []
    monkeypatch.setattr(dingtalk_adapter, "send_markdown", lambda *a: calls.append(("dingtalk", a)) or False)
    monkeypatch.setattr("app.services.webhook_adapter.send_wecom_markdown", lambda *a: calls.append(("wecom", a)) or True)
    daily_pipeline._maybe_push_review("**报告正文**", {"as_of": "2026-10-13", "emotion_label": "中性"})
    assert [name for name, _ in calls] == ["dingtalk", "wecom"]
    assert calls[0][1] == (URL, "每日复盘", "**2026-10-13 · 情绪 中性**\n\n**报告正文**", "SEC-test")


def test_review_skips_unconfigured_dingtalk(monkeypatch):
    preferences.set_review_push_channels(["dingtalk"])
    monkeypatch.setattr(dingtalk_adapter, "send_markdown", lambda *a: pytest.fail("unconfigured"))
    daily_pipeline._maybe_push_review("正文", {})
