#!/usr/bin/env python3
"""Cross-platform launcher for GridironIQ (Windows, macOS, Linux).

Starts the FastAPI backend (port 8000) and the Next.js frontend (port 3000)
together, creating the backend venv / installing frontend deps on first run.
Ctrl+C stops both.

Usage:
    python start.py          (or double-click "Start GridironIQ.bat" on Windows,
                              "Start GridironIQ.command" on macOS)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
IS_WINDOWS = os.name == "nt"


def venv_python() -> Path:
    return BACKEND / ".venv" / ("Scripts" if IS_WINDOWS else "bin") / (
        "python.exe" if IS_WINDOWS else "python"
    )


def ensure_backend() -> Path:
    py = venv_python()
    if not py.exists():
        print("Creating backend virtualenv (first run)...")
        venv.EnvBuilder(with_pip=True).create(BACKEND / ".venv")
        print("Installing backend dependencies (this takes a few minutes the first time)...")
        subprocess.check_call([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"])
        subprocess.check_call([str(py), "-m", "pip", "install", "-q", "-r", str(BACKEND / "requirements.txt")])
    return py


def npm_command() -> str:
    npm = shutil.which("npm.cmd" if IS_WINDOWS else "npm") or shutil.which("npm")
    if not npm:
        print("ERROR: npm not found. Install Node.js from https://nodejs.org and re-run.")
        sys.exit(1)
    return npm


def ensure_frontend(npm: str) -> None:
    if not (FRONTEND / "node_modules").exists():
        print("Installing frontend dependencies (first run)...")
        subprocess.check_call([npm, "install"], cwd=FRONTEND)


def main() -> None:
    py = ensure_backend()
    npm = npm_command()
    ensure_frontend(npm)

    print("\nStarting backend  -> http://localhost:8000")
    backend_proc = subprocess.Popen(
        [str(py), "-m", "uvicorn", "app.main:app", "--port", "8000", "--reload"],
        cwd=BACKEND,
    )
    print("Starting frontend -> http://localhost:3000")
    frontend_proc = subprocess.Popen([npm, "run", "dev", "--", "--port", "3000"], cwd=FRONTEND)

    print("\nGridironIQ is running. Open http://localhost:3000  (Ctrl+C stops both)\n")
    try:
        while True:
            for proc, name in ((backend_proc, "backend"), (frontend_proc, "frontend")):
                code = proc.poll()
                if code is not None:
                    print(f"\n{name} exited with code {code}; shutting down.")
                    raise KeyboardInterrupt
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        for proc in (frontend_proc, backend_proc):
            if proc.poll() is None:
                proc.terminate()
        for proc in (frontend_proc, backend_proc):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
