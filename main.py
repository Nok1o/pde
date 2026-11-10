"""Start the application: python main.py [--server.port 8502].

Uses the project's prepared .venv when available. No automatic installation.
"""

import importlib.util
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    local_python = (
        root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    # On this computer dependencies are already in .venv; archives omit it.
    interpreter = str(local_python) if local_python.is_file() else sys.executable
    if not local_python.is_file() and importlib.util.find_spec("streamlit") is None:
        print("Install dependencies first: python -m pip install -e .", file=sys.stderr)
        return 1
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    os.chdir(root)
    command = [
        interpreter,
        "-m",
        "streamlit",
        "run",
        str(root / "app.py"),
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
        *sys.argv[1:],
    ]
    os.execv(interpreter, command)


if __name__ == "__main__":
    raise SystemExit(main())
