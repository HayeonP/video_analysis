from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_fingerprint(path: Path, *, relative_to: Path | None = None) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    result = {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}
    if relative_to is not None:
        result["relative_path"] = os.path.relpath(path.resolve(), relative_to.resolve())
    return result
