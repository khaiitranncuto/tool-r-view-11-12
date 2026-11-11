import json
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from render_guard import RenderFileLock, RenderBusy, active_render, process_is_alive
from render_encoding import EncoderSession
from video_renderer import _step3_concat_segments, get_ffmpeg_exe, get_video_info


class RenderPerformanceTests(unittest.TestCase):
    def test_gpu_timeout_falls_back_to_cpu_and_disables_gpu(self):
        outcomes = [SimpleNamespace(returncode=0, stderr=''),
                    subprocess.TimeoutExpired('ffmpeg', 5),
                    SimpleNamespace(returncode=0, stderr='')]
        with patch('render_encoding.subprocess.run', side_effect=outcomes) as run:
            encoder = EncoderSession('ffmpeg')
            encoder.run(['ffmpeg'], ['-c:v', 'h264_nvenc'], ['-c:v', 'libx264'], 'scene', timeout=5)
            self.assertFalse(encoder.gpu)
            self.assertIn('libx264', run.call_args.args[0])
            self.assertEqual(run.call_count, 3)

    def test_cpu_timeout_is_bounded_and_preserves_diagnostic(self):
        log = Mock()
        with patch('render_encoding.subprocess.run', side_effect=[
                SimpleNamespace(returncode=1, stderr='No GPU'), subprocess.TimeoutExpired('ffmpeg', 5)]) as run:
            encoder = EncoderSession('ffmpeg', log=log)
            with self.assertRaisesRegex(RuntimeError, 'scene'):
                encoder.run(['ffmpeg'], [], ['-c:v', 'libx264'], 'scene', timeout=5)
            self.assertEqual(run.call_count, 2)
            self.assertIn('timeout', log.call_args.args[2])

    def test_lock_excludes_another_process_and_releases_after_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            lockfile = str(Path(directory) / 'worker.lock')
            script = ('from render_guard import RenderFileLock,RenderBusy; import sys\n'
                      'try:\n with RenderFileLock(sys.argv[1],timeout=0): pass\n'
                      'except RenderBusy: sys.exit(7)\n')
            with RenderFileLock(lockfile):
                result = subprocess.run([sys.executable, '-c', script, lockfile], timeout=15)
                self.assertEqual(result.returncode, 7)
            with RenderFileLock(lockfile, timeout=0):
                pass

    def test_dead_and_completed_workers_do_not_block_new_render(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'active.json'
            path.write_text(json.dumps({'task_id': 'test', 'fingerprint': 'a'}), encoding='utf-8')
            self.assertIsNone(active_render(path, lambda _: {'status': 'success'}))
            self.assertIsNone(active_render(path, lambda _: {'status': 'running', 'created_at': 1}, now=100))
            with patch('render_guard.process_is_alive', return_value=False):
                self.assertIsNone(active_render(path, lambda _: {'status': 'running', 'worker_pid': 123}))
            self.assertEqual(active_render(path, lambda _: {
                'status': 'running', 'worker_pid': os.getpid()})['task_id'], 'test')

    def test_windows_liveness_probe_does_not_send_signals(self):
        with patch('render_guard.os.kill', side_effect=AssertionError('Must not signal a Windows worker')):
            if os.name == 'nt':
                self.assertTrue(process_is_alive(os.getpid()))
        self.assertFalse(process_is_alive(0))

    def test_http_duplicate_reuses_worker_and_other_render_is_rejected(self):
        from flask import Flask, jsonify, request
        import task_store
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(task_store, 'JOB_DIR', str(Path(directory) / '.jobs')):
            web = Flask(__name__)
            task_id = '11111111-1111-1111-1111-111111111111'
            def launch():
                task_store.save_task_state(task_id, {'status': 'running', 'worker_pid': os.getpid()})
                return jsonify({'success': True, 'task_id': task_id})
            tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render_video')
            node.decorator_list = []
            create = Mock(side_effect=launch)
            env = dict(os=os, request=request, jsonify=jsonify, app=web, WORKSPACE_DIR=directory,
                       load_task_state=task_store.load_task_state, _create_render_video_request=create)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<render route>', 'exec'), env)
            web.add_url_rule('/api/render-video', view_func=env['render_video'], methods=['POST'])
            client = web.test_client()
            first = client.post('/api/render-video', json={'video_name': 'a.mp4'})
            repeated = client.post('/api/render-video', json={'video_name': 'a.mp4'})
            other = client.post('/api/render-video', json={'video_name': 'b.mp4'})
            self.assertTrue(first.get_json()['success'])
            self.assertTrue(repeated.get_json()['reused'])
            self.assertEqual(repeated.get_json()['task_id'], task_id)
            self.assertEqual(other.status_code, 409)
            self.assertEqual(create.call_count, 1)
            task_store.save_task_state(task_id, {'status': 'success'})
            with RenderFileLock(Path(directory) / '.jobs/render_worker.lock'):
                busy = client.post('/api/render-video', json={'video_name': 'b.mp4'})
            self.assertEqual(busy.status_code, 409)
            self.assertEqual(create.call_count, 1)

    def test_failed_gpu_probe_is_not_repeated_for_each_segment(self):
        def execute(command, **kwargs):
            return SimpleNamespace(returncode=1 if 'h264_nvenc' in command else 0,
                                   stderr='No capable devices found')
        with patch('render_encoding.subprocess.run', side_effect=execute) as run:
            encoder = EncoderSession('ffmpeg')
            for _ in range(3):
                encoder.run(['ffmpeg', '-i', 'source'], ['-c:v', 'h264_nvenc', 'out'],
                            ['-c:v', 'libx264', 'out'], 'scene')
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(sum('h264_nvenc' in command for command in commands), 1)
            self.assertEqual(sum('libx264' in command for command in commands), 3)

    def test_gpu_failure_disables_gpu_for_remaining_stages(self):
        outcomes = [SimpleNamespace(returncode=0, stderr=''),
                    SimpleNamespace(returncode=1, stderr='encoder failed'),
                    SimpleNamespace(returncode=0, stderr=''), SimpleNamespace(returncode=0, stderr='')]
        with patch('render_encoding.subprocess.run', side_effect=outcomes) as run:
            encoder = EncoderSession('ffmpeg')
            for _ in range(2):
                encoder.run(['ffmpeg'], ['-c:v', 'h264_nvenc'], ['-c:v', 'libx264'], 'stage')
            self.assertFalse(encoder.gpu)
            self.assertIn('libx264', run.call_args_list[-1].args[0])
            self.assertEqual(run.call_count, 4)

    def test_concat_copy_failure_uses_selected_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = Mock()
            with patch('video_renderer.subprocess.run', return_value=SimpleNamespace(returncode=1, stderr='bad copy')):
                _step3_concat_segments(['a.mp4', 'b.mp4'], str(Path(directory) / 'out.mp4'),
                                      directory, encoder=encoder)
            self.assertEqual(encoder.run.call_count, 1)
            self.assertIn('fps=30,setpts=N/(30*TB)', encoder.run.call_args.args[0])

    def test_real_stream_copy_preserves_duration_frame_count_and_decodes(self):
        ffmpeg = get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as directory:
            segments = []
            for index, color in enumerate(('red', 'blue')):
                path = str(Path(directory) / f'{index}.mp4')
                subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'lavfi', '-i', f'color=c={color}:s=320x180:r=30:d=1',
                    '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', path],
                    check=True, timeout=30)
                segments.append(path)
            output = str(Path(directory) / 'joined.mp4')
            encoder = Mock()
            _step3_concat_segments(segments, output, directory, encoder=encoder)
            encoder.run.assert_not_called()
            duration, width, height = get_video_info(output)
            self.assertAlmostEqual(duration, 2.0, delta=.05)
            self.assertEqual((width, height), (320, 180))
            decoded = subprocess.run([ffmpeg, '-hide_banner', '-i', output, '-f', 'null', '-'],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors='replace', timeout=30)
            self.assertEqual(decoded.returncode, 0)
            self.assertEqual(int(re.findall(r'frame=\s*(\d+)', decoded.stderr)[-1]), 60)


if __name__ == '__main__':
    unittest.main()
