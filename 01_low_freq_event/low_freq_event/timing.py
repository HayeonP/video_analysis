"""Crash-tolerant accumulated active time; offline gaps never count as execution."""
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
import uuid
from .storage import atomic_json, read_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunTimer:
    def __init__(self, path: Path, interval: float = 2.0):
        self.path, self.interval = path, interval
        self.data = read_json(path) if path.exists() else {"sessions": []}
        for session in self.data["sessions"]:
            if session["status"] == "running":
                session["status"] = "interrupted"
                session["ended_at"] = session["last_heartbeat_at"]
        self.session = {"id": uuid.uuid4().hex, "started_at": utc_now(), "ended_at": None,
                        "elapsed_sec": 0.0, "status": "running"}
        self.data["sessions"].append(self.session)
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.flush()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.flush()

    def flush(self):
        with self.lock:
            self.session["elapsed_sec"] = round(time.monotonic() - self.started, 3)
            self.session["last_heartbeat_at"] = utc_now()
            self.data["total_active_sec"] = round(sum(s["elapsed_sec"] for s in self.data["sessions"]), 3)
            self.data["heartbeat_interval_sec"] = self.interval
            atomic_json(self.path, self.data)

    def close(self, status: str):
        self.stop.set()
        self.thread.join()
        self.session["status"] = status
        self.session["ended_at"] = utc_now()
        self.flush()
        return self.data
