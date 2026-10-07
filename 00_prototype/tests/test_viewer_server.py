"""CPU-only HTTP checks for seeking without downloading the entire video."""
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
import os

import pytest

from video_events.viewer_server import RangeRequestHandler


@pytest.fixture
def viewer_http(tmp_path):
    (tmp_path / "video.mp4").write_bytes(bytes(range(256)) * 16)
    (tmp_path / "empty.mp4").write_bytes(b"")
    (tmp_path / "viewer.html").write_text("<h1>review</h1>")
    handler = partial(RangeRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    yield server.server_port
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def get(port, path="/video.mp4", *, method="GET", headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("requested,start,end", [("bytes=257-511", 257, 511), ("bytes=4090-", 4090, 4095),
                                                  ("bytes=-6", 4090, 4095), ("bytes=4090-99999", 4090, 4095)])
def test_viewer_server_serves_only_requested_video_bytes(viewer_http, requested, start, end):
    status, headers, body = get(viewer_http, headers={"Range": requested})
    assert status == 206
    assert headers["Content-Range"] == f"bytes {start}-{end}/4096"
    assert headers["Content-Type"] == "video/mp4"
    assert int(headers["Content-Length"]) == len(body) == end - start + 1
    assert body == (bytes(range(256)) * 16)[start:end + 1]


def test_viewer_server_head_and_regular_assets(viewer_http):
    status, headers, body = get(viewer_http, method="HEAD", headers={"Range": "bytes=2-10"})
    assert status == 206 and headers["Content-Length"] == "9" and not body
    status, headers, body = get(viewer_http, path="/viewer.html")
    assert status == 200 and body == b"<h1>review</h1>"
    assert headers["Accept-Ranges"] == "bytes"
    status, _, body = get(viewer_http, headers={"Range": "bytes=2-10", "If-Range": '"old"'})
    assert status == 200 and len(body) == 4096


@pytest.mark.parametrize("path,requested", [("/video.mp4", "bytes=4096-"), ("/video.mp4", "bytes=5-2"),
                                           ("/video.mp4", "bytes=-0"), ("/empty.mp4", "bytes=0-")])
def test_viewer_server_reports_unsatisfiable_ranges(viewer_http, path, requested):
    status, headers, body = get(viewer_http, path, headers={"Range": requested})
    assert status == 416 and headers["Content-Range"].startswith("bytes */") and not body


def test_replaced_page_ignores_old_browser_cache_even_with_older_mtime(viewer_http, tmp_path):
    status, headers, _ = get(viewer_http, path="/viewer.html")
    assert status == 200 and headers["Cache-Control"] == "no-store"
    modified = headers["Last-Modified"]
    page = tmp_path / "viewer.html"
    timestamp = page.stat().st_mtime
    page.write_text("<h1>current result</h1>")
    os.utime(page, (timestamp - 3600, timestamp - 3600))
    status, headers, body = get(viewer_http, path="/viewer.html", headers={"If-Modified-Since": modified})
    assert status == 200 and body == b"<h1>current result</h1>"
    assert headers["Cache-Control"] == "no-store"
