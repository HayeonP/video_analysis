#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    venv = ROOT / ".venv"
    if not (venv / ".ready").exists():
        subprocess.run(["bash", str(ROOT / "runtime/setup.sh")], check=True)
    if Path(sys.prefix).resolve() != venv.resolve():
        env = dict(os.environ, VIRTUAL_ENV=str(venv), PYTHONNOUSERSITE="1")
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        python = str(venv / "bin/python")
        os.execve(python, [python, str(ROOT / "runtime/run.py"), *sys.argv[1:]], env)
    from low_freq_event.cli import main
    raise SystemExit(main())

