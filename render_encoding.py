"""One encoder decision per render; bounded CPU work after GPU failure."""
import subprocess
import threading
import time


class EncoderSession:
    def __init__(self, ffmpeg, log=None):
        self.ffmpeg, self.log = ffmpeg, log
        self.lock = threading.Lock()
        try:
            result = subprocess.run([
                ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin',
                '-f', 'lavfi', '-i', 'color=size=320x180:rate=30', '-frames:v', '2',
                '-c:v', 'h264_nvenc', '-preset', 'p1', '-f', 'null', '-',
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                errors='replace', timeout=15)
            self.gpu = result.returncode == 0
            details = result.stderr or ''
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.gpu, details = False, str(exc)
        if log:
            log('INFO' if self.gpu else 'WARNING',
                '[ENCODER] NVENC đã sẵn sàng.' if self.gpu else
                '[ENCODER] NVENC không dùng được; dùng CPU cho cả tác vụ, không thử lại trên từng cảnh.',
                details[-3000:])

    def run(self, common, gpu_args, cpu_args, label, timeout=600, cwd=None):
        started = time.monotonic()
        common = [common[0], '-hide_banner', '-loglevel', 'error', '-nostdin',
                  '-filter_threads', '1', '-filter_complex_threads', '1'] + common[1:]
        with self.lock:
            use_gpu = self.gpu
        def execute(args):
            try:
                return subprocess.run(common + args, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, text=True, errors='replace', timeout=timeout, cwd=cwd)
            except subprocess.TimeoutExpired:
                # A stalled NVENC call is also an encoder failure: allow the
                # existing CPU fallback, with its own bounded timeout.
                return subprocess.CompletedProcess(common + args, -1,
                    stderr=f'FFmpeg timeout after {timeout}s')
        result = execute(gpu_args if use_gpu else cpu_args)
        if result.returncode and use_gpu:
            with self.lock:
                first_failure = self.gpu
                self.gpu = False
            if first_failure and self.log:
                self.log('WARNING', f'[ENCODER] {label}: NVENC lỗi; các bước tiếp theo dùng CPU.',
                         (result.stderr or '')[-3000:])
            result = execute(cpu_args)
        if result.returncode:
            if self.log:
                self.log('ERROR', f'[ENCODE_FAILED] {label}; code={result.returncode}',
                         (result.stderr or '')[-6000:])
            raise RuntimeError(f'{label}: FFmpeg lỗi {result.returncode}; xem nhật ký chi tiết.')
        return time.monotonic() - started
