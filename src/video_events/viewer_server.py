"""Static viewer server with LAN access and byte ranges for video seeking."""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import socket
from urllib.parse import quote, urlsplit


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, home_page: str | None = None, **kwargs):
        self.home_page = home_page
        super().__init__(*args, **kwargs)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self._byte_range = None
        if urlsplit(self.path).path == "/" and self.home_page:
            self.send_response(302)
            self.send_header("Location", self.home_page)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        header = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        # Unknown validators and multipart ranges fall back to a complete response.
        if not match or not any(match.groups()) or self.headers.get("If-Range"):
            return super().send_head()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        try:
            stream = open(path, "rb")
        except OSError:
            return super().send_head()
        try:
            stat = os.fstat(stream.fileno())
            size = stat.st_size
            first, last = match.groups()
            start = int(first) if first else max(0, size - int(last))
            end = min(int(last), size - 1) if first and last else size - 1
            if start >= size or end < start:
                stream.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            stream.seek(start)
            self._byte_range = (start, end)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return stream
        except BaseException:
            stream.close()
            raise

    def copyfile(self, source, outputfile):
        try:
            if self._byte_range is None:
                return super().copyfile(source, outputfile)
            start, end = self._byte_range
            remaining = end - start + 1
            while remaining:
                block = source.read(min(64 * 1024, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel downloads when the user seeks to another time.


def lan_address() -> str | None:
    """Ask the OS which source address its default route uses; send no packets."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 80))
            return probe.getsockname()[0]
    except OSError:
        return None


def serve(directory: Path, port: int = 5000, host: str = "0.0.0.0", *, home: Path | None = None) -> None:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"파일을 제공할 폴더가 없습니다: {directory}")
    if not 0 <= port <= 65535:
        raise ValueError("포트는 0~65535 범위여야 합니다.")
    # Keep asset URLs relative to the actual HTML by redirecting the root URL.
    # Prefer this project's current Sintel experiment; otherwise open the latest viewer.
    if home is not None:
        home = home.resolve()
        if not home.is_relative_to(directory) or not home.is_file():
            raise ValueError("서버의 첫 화면은 프로젝트 폴더 안에 있는 HTML이어야 합니다. 외부 경로는 --generate-only를 사용하세요.")
    else:
        viewers = [directory / "outputs/sintel/viewer.html", directory / "viewer.html"]
        viewers.extend(sorted((directory / "outputs").glob("*/viewer.html"),
                              key=lambda path: path.stat().st_mtime, reverse=True))
        home = next((path for path in viewers if path.is_file()), None)
    home_page = "/" + quote(home.relative_to(directory).as_posix(), safe="/") if home else None
    handler = partial(RangeRequestHandler, directory=str(directory), home_page=home_page)
    with ThreadingHTTPServer((host, port), handler) as server:
        if host in {"0.0.0.0", "127.0.0.1", "localhost"}:
            print(f"서버 PC에서 열기: http://localhost:{server.server_port}", flush=True)
        address = lan_address() if host == "0.0.0.0" else host
        if address and address not in {"127.0.0.1", "localhost"}:
            print(f"같은 공유기의 다른 PC에서 열기: http://{address}:{server.server_port}", flush=True)
        print("Ctrl+C로 뷰어 서버를 종료합니다.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n뷰어 서버를 종료했습니다.")
