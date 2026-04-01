"""Simple launcher for the Streamlit frontend."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    app_path = root / "streamlit_app.py"

    if not app_path.exists():
        print(f"Could not find {app_path}")
        return 1

    if importlib.util.find_spec("streamlit") is None:
        print("Streamlit is not installed in this Python environment.")
        print("Install it with: pip install streamlit")
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
