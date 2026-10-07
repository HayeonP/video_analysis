from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import signal
import subprocess
import sys
from threading import Thread
from urllib.request import Request, urlopen

import psutil
import pytest

from low_freq_event.viewer_server import RangeRequestHandler, bind_server


@contextmanager
def occupied_port(ignore_term=False, host="127.0.0.1"):
    script = """
import signal, socket, sys
if sys.argv[1] == 'ignore':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((sys.argv[2], 0))
    listener.listen()
    print(listener.getsockname()[1], flush=True)
    while True:
        signal.pause()
"""
    process = subprocess.Popen([sys.executable, "-u", "-c", script, "ignore" if ignore_term else "stop", host],
                               stdout=subprocess.PIPE, text=True)
    try:
        port = int(process.stdout.readline())
        yield process, port
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


@pytest.mark.parametrize("ignore_term", [False, True])
def test_replaces_listener_and_only_forces_unresponsive_process(ignore_term, capsys):
    with occupied_port(ignore_term) as (old, port), occupied_port() as (unrelated, other_port):
        with bind_server("127.0.0.1", port, SimpleHTTPRequestHandler, stop_timeout=.1) as server:
            assert server.server_port == port
            old.wait(timeout=5)
            # psutil.wait_procs may reap our child before Popen.wait observes its status.
            assert not psutil.pid_exists(old.pid)
            assert ("강제 종료" in capsys.readouterr().out) == ignore_term
            assert unrelated.poll() is None
            assert other_port != port


def test_available_address_does_not_stop_listener_on_another_interface():
    with occupied_port(host="127.0.0.2") as (other, port):
        with bind_server("127.0.0.1", port, SimpleHTTPRequestHandler) as server:
            assert server.server_port == port
            assert other.poll() is None


def test_wildcard_bind_replaces_loopback_listener():
    with occupied_port() as (old, port):
        with bind_server("0.0.0.0", port, SimpleHTTPRequestHandler, stop_timeout=.1):
            old.wait(timeout=5)
            assert not psutil.pid_exists(old.pid)


def test_does_not_terminate_its_own_process():
    with ThreadingHTTPServer(("127.0.0.1", 0), SimpleHTTPRequestHandler) as existing:
        with pytest.raises(RuntimeError, match="자기 자신"):
            bind_server("127.0.0.1", existing.server_port, SimpleHTTPRequestHandler)


@pytest.mark.parametrize("name", ["viewer.html", "frame.jpg", "video.mp4"])
def test_result_switch_serves_new_asset_even_with_older_mtime(tmp_path, name):
    asset = tmp_path / name
    asset.write_bytes(b"old result")
    handler = partial(RangeRequestHandler, directory=str(tmp_path))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/{name}"
            with urlopen(url) as response:
                assert response.read() == b"old result"
                modified = response.headers["Last-Modified"]
                assert response.headers["Cache-Control"] == "no-store"
            asset.write_bytes(b"corrected result")
            os.utime(asset, (1, 1))
            request = Request(url, headers={"If-Modified-Since": modified})
            with urlopen(request) as response:
                assert response.status == 200
                assert response.read() == b"corrected result"
            with urlopen(Request(url, headers={"Range": "bytes=0-8"})) as response:
                assert response.status == 206
                assert response.read() == b"corrected"
        finally:
            server.shutdown()
            thread.join(timeout=5)
