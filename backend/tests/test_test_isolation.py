import socket

import pytest

from app.config import settings
from app.services import preferences


def test_runtime_is_private_to_each_test(tmp_path):
    assert settings.data_dir == tmp_path / "runtime"
    assert preferences.load() == {}


def test_external_dns_and_connections_are_blocked():
    with pytest.raises(RuntimeError, match="External networking is blocked"):
        socket.getaddrinfo("example.invalid", 443)
    with socket.socket() as sock, pytest.raises(RuntimeError, match="External networking is blocked"):
        sock.connect(("192.0.2.1", 443))
