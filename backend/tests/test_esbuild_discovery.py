import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import ext_source_import


def binary(root, package, name="bin/esbuild"):
    path = (
        root
        / "node_modules/.pnpm"
        / f"@esbuild+{package}@0.21.5/node_modules/@esbuild"
        / package
        / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture", encoding="utf8")
    return path


@pytest.mark.parametrize(
    "system,machine,package,name",
    [
        ("Darwin", "arm64", "darwin-arm64", "bin/esbuild"),
        ("Linux", "x86_64", "linux-x64", "bin/esbuild"),
        ("Windows", "AMD64", "win32-x64", "esbuild.exe"),
    ],
)
def test_discovery_uses_host_platform(tmp_path, monkeypatch, system, machine, package, name):
    import platform

    binary(tmp_path, "aix-ppc64")
    expected = binary(tmp_path, package, name)
    monkeypatch.setattr(ext_source_import, "frontend_root", lambda: tmp_path)
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr("shutil.which", lambda _: None)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="0.21.5\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert ext_source_import.find_esbuild() == expected
    assert calls == [[str(expected), "--version"]]


def test_bad_launcher_is_skipped_and_other_platforms_are_never_executed(tmp_path, monkeypatch):
    import platform

    binary(tmp_path, "aix-ppc64")
    native = binary(tmp_path, "darwin-arm64")
    launcher = tmp_path / "node_modules/.bin/esbuild"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("invalid executable", encoding="utf8")
    monkeypatch.setattr(ext_source_import, "frontend_root", lambda: tmp_path)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr("shutil.which", lambda _: None)

    def run(args, **kwargs):
        if args[0] == str(launcher):
            raise OSError("wrong executable")
        assert args == [str(native), "--version"]
        return SimpleNamespace(returncode=0, stdout="0.21.5\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert ext_source_import.find_esbuild() == native


@pytest.mark.parametrize(
    "error,expected",
    [
        (OSError("cannot execute"), ext_source_import.SourceImportUnavailableError),
        (subprocess.TimeoutExpired("esbuild", 60), ext_source_import.SourceCompileError),
    ],
)
def test_compile_tool_failures_are_mapped_and_temporary_files_removed(
    tmp_path, monkeypatch, error, expected
):
    source = tmp_path / "extension.tsx"
    source.write_text("export default {}", encoding="utf8")
    monkeypatch.setattr(ext_source_import, "frontend_root", lambda: tmp_path)
    monkeypatch.setattr(ext_source_import, "find_esbuild", lambda: Path("esbuild"))

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(expected):
        ext_source_import.compile_source(source, "demo")
    assert not list((tmp_path / "node_modules").glob(".tsp-import-*"))
