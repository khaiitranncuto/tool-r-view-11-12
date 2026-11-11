"""Small offline encode: real FFmpeg and subtitles, synthetic audio, no TTS network."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from structured_render import make_render_plan, render_structured_video
from tts_engine import get_ffmpeg_exe, get_audio_duration


class StructuredRenderSmoke(unittest.TestCase):
    def test_real_mp4_and_resume_without_resynthesizing_or_recutting(self):
        ffmpeg = get_ffmpeg_exe()
        class CpuEncoder:
            def __init__(self, *args):
                pass
            def run(self, common, gpu_args, cpu_args, label, timeout=600, cwd=None):
                result = subprocess.run(common + cpu_args, capture_output=True, text=True,
                                        timeout=timeout, cwd=cwd)
                if result.returncode:
                    raise RuntimeError(label + ': ' + result.stderr[-3000:])
                return 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            result = subprocess.run([ffmpeg, '-y', '-f', 'lavfi', '-i',
                'color=c=blue:s=160x90:r=30:d=4', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', str(source)], capture_output=True)
            self.assertEqual(result.returncode, 0)
            payload = {'parts': [{'scenes': [{'scene_id': 'S0001', 'tts': 'Mở cửa. Rồi bước vào.',
                'locked_visual_shots': [{'start': '00:00', 'end': '00:02'},
                                        {'start': '00:02', 'end': '00:04'}],
                'visual_links': [{'trigger_text': ''}, {'trigger_text': 'Rồi bước vào'}]}]}]}
            plan = make_render_plan(payload, 4)
            def synthetic_tts(texts, voice, rate, temp_dir, *args, **kwargs):
                work = Path(temp_dir)
                audio = work / 'voice.wav'
                subprocess.run([ffmpeg, '-y', '-f', 'lavfi', '-i',
                    'sine=frequency=440:sample_rate=24000:duration=1', str(audio)],
                    capture_output=True, check=True)
                (work / '_USE_MASTER_TTS_').write_text(str(audio))
                (work / 'submaker_subtitles.ass').write_text(
                    '[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n'
                    '[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, '
                    'OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, '
                    'Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n'
                    'Style: Default,Arial,32,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n'
                    '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
                    'Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Offline render test\n', encoding='utf-8')
                return [str(audio)], [1.0], [[{'start': 0, 'end': .45, 'content': 'Mở cửa'},
                                            {'start': .5, 'end': 1, 'content': 'Rồi bước vào'}]]
            options = {'aspect_ratio': '16:9', 'mirror': False, 'zoom_factor': 1,
                       'enable_zoom_crop': False, 'enable_subtitles': True, 'render_workers': 1}
            updates = []
            progress = lambda p, message: updates.append((p, message))
            with patch('tts_engine.generate_all_tts', side_effect=synthetic_tts) as tts, \
                 patch('render_encoding.EncoderSession', CpuEncoder):
                url = render_structured_video('offline-first', str(source), '', .12,
                                              'voice', '+20%', plan, options, progress)
                self.assertEqual(tts.call_count, 1)
                output = root / url.lstrip('/')
                self.assertTrue(output.exists())
                self.assertAlmostEqual(get_audio_duration(str(output)), 1.0, delta=.15)
                with patch('video_renderer.create_master_video_raw', side_effect=AssertionError('must reuse video')):
                    render_structured_video('offline-resume', str(source), '', .12,
                                            'voice', '+20%', plan, options, progress)
                self.assertEqual(tts.call_count, 1)
                self.assertEqual(updates[-1][0], 100)
                cache = next((root / '.render_cache').iterdir())
                self.assertTrue((cache / 'audio_visual_edl.json').exists())
                self.assertTrue(list(cache.glob('seg_*.complete.json')))


if __name__ == '__main__':
    unittest.main()
