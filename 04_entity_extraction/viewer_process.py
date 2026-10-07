"""Release the requested viewer port by stopping its existing listeners."""
import errno
import os
from pathlib import Path
import signal
import time


def listening_sockets(port):
    sockets = set()
    for table in (Path('/proc/net/tcp'), Path('/proc/net/tcp6')):
        if not table.exists():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == '0A' and int(fields[1].split(':')[-1], 16) == port:
                sockets.add(f'socket:[{fields[9]}]')
    return sockets


def owns_socket(process, sockets):
    try:
        for descriptor in (process / 'fd').iterdir():
            try:
                if os.readlink(descriptor) in sockets:
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def process_start(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def signal_process(pid, descriptor, started, signum):
    if descriptor is not None:
        signal.pidfd_send_signal(descriptor, signum)
    elif started is not None and process_start(pid) == started:
        os.kill(pid, signum)


def stop_port_listeners(port):
    sockets = listening_sockets(port)
    handles = []
    try:
        for process in Path('/proc').iterdir():
            if not process.name.isdigit() or int(process.name) == os.getpid():
                continue
            if not owns_socket(process, sockets):
                continue
            pid = int(process.name)
            try:
                # Prefer pidfds; some Python builds omit them. Check process start
                # time before signalling in that case to detect reused PIDs.
                started = process_start(pid)
                descriptor = (os.pidfd_open(pid) if hasattr(os, 'pidfd_open')
                              and hasattr(signal, 'pidfd_send_signal') else None)
                handles.append((pid, descriptor, started))
                if not owns_socket(process, sockets):
                    continue
                signal_process(pid, descriptor, started, signal.SIGTERM)
                print(f'포트 {port} 점유 프로세스 종료 요청: PID {pid}', flush=True)
            except ProcessLookupError:
                continue
        if not handles:
            raise OSError(errno.EADDRINUSE,
                          f'포트 {port} 점유 프로세스를 확인할 수 없습니다. 프로세스 조회·종료 권한을 확인하세요.')
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if not any(owns_socket(Path('/proc') / str(pid), sockets) for pid, _, _ in handles):
                break
            time.sleep(.05)
        for pid, descriptor, started in handles:
            if owns_socket(Path('/proc') / str(pid), sockets):
                try:
                    signal_process(pid, descriptor, started, signal.SIGKILL)
                    print(f'포트 {port} 점유 프로세스 강제 종료: PID {pid}', flush=True)
                except ProcessLookupError:
                    pass
    finally:
        for _, descriptor, _ in handles:
            if descriptor is not None:
                os.close(descriptor)


def bind_replacing_viewer(server_class, address, handler):
    try:
        return server_class(address, handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
    stop_port_listeners(address[1])
    deadline = time.monotonic() + 5
    while True:
        try:
            return server_class(address, handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or time.monotonic() >= deadline:
                raise
            time.sleep(.05)
