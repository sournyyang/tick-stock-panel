"""Unit tests must not inherit a developer's accounts or data-source settings."""

from __future__ import annotations

import ipaddress
import socket

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    from app import secrets_store
    from app.config import settings
    from app.services import preferences

    monkeypatch.setattr(settings, "data_dir", tmp_path / "runtime")
    preferences._invalidate_cache()
    monkeypatch.setattr(secrets_store, "_cache", None)
    monkeypatch.setattr(secrets_store, "_cache_sig", None)
    yield
    preferences._invalidate_cache()


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    getaddrinfo = socket.getaddrinfo

    def local(host):
        if isinstance(host, bytes):
            host = host.decode()
        if host in (None, "", "localhost"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def check(host):
        if not local(host):
            raise RuntimeError("External networking is blocked in tests; mock the provider")

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            check(address[0])
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            check(address[0])
        return connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        check(host)
        return getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
