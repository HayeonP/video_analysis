#!/usr/bin/env python3
"""Public entry point; always re-executes inside the project virtual environment."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def environment_signature() -> str:
    digest = hashlib.sha256()
    for name in ("setup.sh", "requirements.txt", "pyproject.toml"):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def main() -> int:
    marker = VENV / ".ready"
    if sys.argv[1:] == ["--mark-environment-ready"]:
        if Path(sys.prefix).resolve() != VENV.resolve():
            raise SystemExit("가상환경 내부에서만 설치 완료를 기록할 수 있습니다.")
        marker.write_text(environment_signature(), encoding="utf-8")
        return 0
    if not marker.exists() or marker.read_text().strip() != environment_signature():
        subprocess.run(["bash", str(ROOT / "setup.sh")], check=True)
    if Path(sys.prefix).resolve() != VENV.resolve():
        environment = dict(os.environ)
        environment.pop("PYTHONHOME", None)
        environment["VIRTUAL_ENV"] = str(VENV)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PATH"] = str(VENV / "bin") + os.pathsep + environment.get("PATH", "")
        python = str(VENV / "bin/python")
        os.execve(python, [python, str(ROOT / "run.py"), *sys.argv[1:]], environment)
    from video_events.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
