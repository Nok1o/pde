"""The documented entry point starts a local server without installing packages."""

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


def test_python_main_starts_server_from_another_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    with (tmp_path / "server.log").open("w+") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                str(root / "main.py"),
                "--server.port",
                str(port),
                "--server.headless",
                "true",
                "--server.fileWatcherType",
                "none",
            ],
            cwd=tmp_path,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 45
            while server.poll() is None and time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/_stcore/health", timeout=1
                    ) as response:
                        if response.status == 200 and response.read() == b"ok":
                            return
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.1)
            log.seek(0)
            pytest.fail("main.py did not start a healthy server:\n" + log.read())
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
