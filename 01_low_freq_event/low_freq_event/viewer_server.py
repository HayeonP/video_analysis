"""Static viewer server with LAN access and byte ranges for video seeking."""
from __future__ import annotations

import errno
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import os
from pathlib import Path
import re
import socket
import time
from urllib.parse import quote, urlsplit

import psutil


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, home_page: str | None = None, **kwargs):
        self.home_page = home_page
        super().__init__(*args, **kwargs)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        # Different analyses reuse /viewer.html, /video.mp4 and frame URLs.
        # A browser must not reuse assets from the previously served result.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        self._byte_range = None
        # A new result can have older mtimes than the previous one. Ignore
        # validators from pages cached before no-store was introduced as well.
        for header in ("If-Modified-Since", "If-None-Match"):
            if header in self.headers:
                del self.headers[header]
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


def bind_server(host: str, port: int, handler, *, stop_timeout: float = 2.0) -> ThreadingHTTPServer:
    """Replace listeners on the conflicting TCP address, then bind the new viewer."""
    try:
        return ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE or port == 0:
            raise

    addresses = {entry[4][0] for entry in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)}
    listeners = []
    for connection in psutil.net_connections(kind="tcp"):
        if connection.status != psutil.CONN_LISTEN or connection.laddr.port != port:
            continue
        address = ipaddress.ip_address(connection.laddr.ip)
        if isinstance(address, ipaddress.IPv6Address):
            address = address.ipv4_mapped or address
            if address.version == 6 and not address.is_unspecified:
                continue  # An IPv6-only local address cannot block this IPv4 server.
        if address.is_unspecified or "0.0.0.0" in addresses or str(address) in addresses:
            listeners.append(connection)
    pids = {connection.pid for connection in listeners}
    if None in pids:
        raise PermissionError(f"{host}:{port} 점유 프로세스의 PID를 확인할 권한이 없습니다.")
    if os.getpid() in pids:
        raise RuntimeError(f"현재 프로세스가 이미 {host}:{port}를 사용하고 있어 자기 자신을 종료할 수 없습니다.")

    processes = []
    for pid in sorted(pids):
        try:
            process = psutil.Process(pid)
            print(f"포트 {port} 사용 중 · 기존 프로세스 종료: PID {pid} ({process.name()})", flush=True)
            process.terminate()
            processes.append(process)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise PermissionError(f"포트 {port}의 프로세스 PID {pid}를 종료할 권한이 없습니다.") from exc
    _, alive = psutil.wait_procs(processes, timeout=stop_timeout)
    for process in alive:
        try:
            print(f"종료에 응답하지 않아 강제 종료합니다: PID {process.pid}", flush=True)
            process.kill()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise PermissionError(f"프로세스 PID {process.pid}를 강제 종료할 권한이 없습니다.") from exc
    if alive:
        psutil.wait_procs(alive, timeout=stop_timeout)

    # Socket teardown can lag behind process exit. Never terminate unrelated new
    # listeners that acquire the address while this replacement is in progress.
    deadline = time.monotonic() + 2.0
    while True:
        try:
            return ThreadingHTTPServer((host, port), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{host}:{port}를 비우지 못했습니다. 포트 점유 상태를 확인하세요.") from exc
            time.sleep(0.05)


def serve(directory: Path, port: int = 5001, host: str = "0.0.0.0", *, home: Path | None = None) -> None:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"파일을 제공할 폴더가 없습니다: {directory}")
    if not 0 <= port <= 65535:
        raise ValueError("포트는 0~65535 범위여야 합니다.")
    # Keep asset URLs relative to the actual HTML by redirecting the root URL.
    if home is not None:
        home = home.resolve()
        if not home.is_relative_to(directory) or not home.is_file():
            raise ValueError("서버의 첫 화면은 프로젝트 폴더 안에 있는 HTML이어야 합니다. 외부 경로는 --generate-only를 사용하세요.")
    else:
        candidate = directory / "viewer.html"
        home = candidate if candidate.is_file() else None
    home_page = "/" + quote(home.relative_to(directory).as_posix(), safe="/") if home else None
    handler = partial(RangeRequestHandler, directory=str(directory), home_page=home_page)
    with bind_server(host, port, handler) as server:
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
