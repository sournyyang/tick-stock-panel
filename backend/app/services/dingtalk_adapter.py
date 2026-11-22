"""钉钉群自定义机器人投递适配：仅负责协议，不计算或去重告警。

协议参考：https://open.dingtalk.com/document/robots/custom-robot-access
签名参考：https://open.dingtalk.com/document/robots/customize-robot-security-settings
调用方在现有后台线程池中投递；失败返回 False，不影响站内通知。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlparse

logger = logging.getLogger(__name__)

# 给消息编码和平台限制留余量；超长报告引导回应用查看。
MAX_MESSAGE_BYTES = 18_000
_TRUNCATED_HINT = "\n\n报告过长，请在应用内查看完整内容。"


def is_valid_url(url: str) -> bool:
    """只接收官方机器人原始 Webhook，拒绝过期签名和非官方端点。"""
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query, keep_blank_values=True)
        return (
            parsed.scheme == "https"
            and parsed.netloc == "oapi.dingtalk.com"
            and parsed.path == "/robot/send"
            and not parsed.fragment
            and set(query) == {"access_token"}
            and len(query["access_token"]) == 1
            and bool(query["access_token"][0].strip())
        )
    except (TypeError, ValueError):
        return False


def _signed_url(url: str, secret: str) -> str:
    if not secret:
        return url
    timestamp = str(int(time.time() * 1000))
    digest = hmac.new(
        secret.encode(), f"{timestamp}\n{secret}".encode(), hashlib.sha256,
    ).digest()
    # urlencode 恰好编码一次，避免 Base64 中的 + 被解析为空格。
    return url + "&" + urlencode({
        "timestamp": timestamp, "sign": base64.b64encode(digest).decode("ascii"),
    })


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_MESSAGE_BYTES:
        return text
    budget = MAX_MESSAGE_BYTES - len(_TRUNCATED_HINT.encode("utf-8"))
    return encoded[:budget].decode("utf-8", errors="ignore") + _TRUNCATED_HINT


@dataclass(frozen=True)
class SendResult:
    ok: bool
    detail: str


def _business_failure(code: object, message: object) -> str:
    # 只映射已知原因，不把可能带凭据、签名或出口 IP 的响应原文返回给页面。
    message = str(message).lower()
    if "sign not match" in message or "签名不匹配" in message:
        reason = "签名不匹配，请从同一个机器人的安全设置重新复制 Webhook 和 SEC 加签密钥"
    elif "timestamp" in message or "时间戳" in message:
        reason = "时间戳无效，请检查服务器系统时间是否准确"
    elif "keyword" in message or "关键词" in message:
        reason = "消息未包含机器人要求的关键词，请检查机器人关键词安全设置"
    elif any(word in message for word in ("whitelist", "allowlist", "白名单")):
        reason = "出口 IP 不在机器人 IP 白名单中，请检查机器人安全设置"
    elif code in (300001, 40035):
        reason = "Webhook 无效或机器人已删除，请重新复制机器人地址"
    else:
        reason = "机器人拒绝请求，请检查 Webhook 和机器人安全设置"
    label = str(code) if type(code) is int else "invalid"
    return f"钉钉推送失败（errcode={label}）：{reason}"


def _post(url: str, payload: dict, secret: str, max_attempts: int) -> SendResult:
    import httpx

    detail = "钉钉推送失败：未发送请求"
    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.post(_signed_url(url, secret), json=payload, timeout=5.0)
            if response.status_code == 200:
                try:
                    result = response.json()
                except ValueError:
                    logger.warning("钉钉推送失败：响应不是 JSON")
                    return SendResult(False, "钉钉推送失败：响应不是有效 JSON")
                code = result.get("errcode") if isinstance(result, dict) else None
                if type(code) is int and code == 0:
                    return SendResult(True, "测试消息已发送，请检查钉钉群")
                # 业务拒绝（签名、关键词、限流等）立即返回，不重复刷群。
                # 不记录原始响应或异常文本，避免泄漏 access_token / 签名密钥。
                detail = _business_failure(code, result.get("errmsg", "") if isinstance(result, dict) else "")
                logger.warning("%s", detail)
                return SendResult(False, detail)
            logger.warning("钉钉推送 HTTP %s", response.status_code)
            detail = f"钉钉推送失败：HTTP {response.status_code}，请检查网络或稍后重试"
            if response.status_code < 500:
                return SendResult(False, detail)
        except Exception as exc:  # noqa: BLE001
            logger.warning("钉钉推送网络失败：%s", type(exc).__name__)
            if isinstance(exc, httpx.ConnectTimeout):
                detail = "钉钉连接超时：请检查后端服务器到钉钉的网络、代理和防火墙设置"
            elif isinstance(exc, httpx.TimeoutException):
                detail = "钉钉请求超时：请检查后端服务器网络或稍后重试"
            else:
                detail = "钉钉网络请求失败：请检查后端服务器网络、代理和证书设置"
        if attempt < max_attempts:
            time.sleep(min(2 ** (attempt - 1), 3))
    return SendResult(False, detail)


def send_text(url: str, title: str, body: str, secret: str = "", max_attempts: int = 3) -> bool:
    """监控告警/配置测试；HTTP 200 且 errcode=0 才算成功。"""
    return send_text_result(url, title, body, secret, max_attempts).ok


def send_text_result(url: str, title: str, body: str, secret: str = "", max_attempts: int = 3) -> SendResult:
    """与生产发送共用协议，配置测试额外返回可操作的脱敏原因。"""
    content = f"{title}\n{body}".strip()
    if not is_valid_url(url) or not content:
        return SendResult(False, "钉钉推送失败：Webhook 地址非法或消息为空")
    return _post(url, {"msgtype": "text", "text": {"content": _truncate(content)}}, secret, max_attempts)


def send_markdown(url: str, title: str, body: str, secret: str = "", max_attempts: int = 3) -> bool:
    """复盘报告；超长正文按 UTF-8 字节截断，并附应用内查看提示。"""
    if not is_valid_url(url) or not body.strip():
        return False
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": title, "text": _truncate(f"## {title}\n\n{body}".strip())},
    }
    return _post(url, payload, secret, max_attempts).ok
