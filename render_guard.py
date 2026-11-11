"""Cross-process render admission and exclusive worker ownership."""
import hashlib
import json
import os
from pathlib import Path
import time


class RenderBusy(RuntimeError):
    pass


class RenderFileLock:
    def __init__(self, path, timeout=5):
        self.path, self.timeout, self.handle = Path(path), timeout, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.handle.write(b'0')
            self.handle.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self.handle.close()
                    self.handle = None
                    raise RenderBusy('Máy đang có tác vụ render. Vui lòng chờ tác vụ hiện tại hoàn tất.')
                time.sleep(.05)

    def __exit__(self, *args):
        if self.handle:
            self.handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def render_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def process_is_alive(pid):
    if pid <= 0:
        return False
    if os.name == 'nt':
        # os.kill(pid, 0) is CTRL_C_EVENT on Windows, not a safe liveness probe.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def active_render(registry_path, load_state, now=None):
    """Ignore completed jobs/dead workers, but retain freshly launched jobs."""
    try:
        registry = json.loads(Path(registry_path).read_text(encoding='utf-8'))
        state = load_state(registry['task_id'])
        if not state or state.get('status') not in ('queued', 'running'):
            return None
        current = time.time() if now is None else now
        pid = int(state.get('worker_pid') or 0)
        if pid:
            if not process_is_alive(pid):
                return None
        elif current - float(state.get('created_at') or 0) > 60:
            return None
        return registry
    except (OSError, ValueError, KeyError, TypeError):
        return None
