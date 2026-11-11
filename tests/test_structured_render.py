import asyncio
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
import unittest
from unittest.mock import AsyncMock, Mock, patch

from structured_render import make_render_plan, prepare_plan_inputs, strict_visual_edl, cache_key
from tts_engine import _recover_failed_tts_groups


class StructuredRenderTests(unittest.TestCase):
    def payload(self):
        return {'characters': [{'id': 'P01', 'name': 'Nobita'}], 'parts': [{'scenes': [{
            'scene_id': 'S0001', 'tts': 'Nobita mở cửa. Sau đó cậu bước vào nhà.', 'characters': ['P01'],
            'locked_visual_shots': [{'shot_id': 'SHOT_000003', 'start': '00:12', 'end': '00:18'},
                                    {'shot_id': 'SHOT_000005', 'start': '00:24', 'end': '00:30'}],
            'visual_links': [{'visual_id': 'V001', 'trigger_text': ''},
                             {'visual_id': 'V002', 'trigger_text': 'Sau đó cậu'}]}]}]}

    def test_json_inputs_preserve_paragraph_identity_and_source_gap(self):
        plan = make_render_plan(self.payload(), 60)
        texts, starts, ends, groups = prepare_plan_inputs(plan)
        self.assertEqual(texts, ['Nobita mở cửa. Sau đó cậu bước vào nhà.'])
        self.assertEqual(plan['situations'][0]['character_ids'], ['P01'])
        self.assertEqual(starts, [12])
        self.assertEqual(ends, [30])
        self.assertEqual([(g['start'], g['end']) for g in groups[0]], [(12, 18), (24, 30)])
        self.assertNotIn('[TIME', json.dumps(plan))

    def test_invalid_ranges_and_missing_trigger_reject_without_repair(self):
        for kind in ['overlap', 'outside', 'trigger', 'duplicate']:
            payload = self.payload()
            scene = payload['parts'][0]['scenes'][0]
            if kind == 'overlap':
                scene['locked_visual_shots'][1]['start'] = '00:16'
            elif kind == 'outside':
                scene['locked_visual_shots'][1]['end'] = '02:00'
            elif kind == 'trigger':
                scene['visual_links'][1]['trigger_text'] = 'not in speech'
            else:
                payload['parts'][0]['scenes'].append(copy.deepcopy(scene))
            with self.assertRaises(ValueError):
                make_render_plan(payload, 60)

    def test_audio_led_edl_requires_cues_and_never_extends_selected_shot(self):
        plan = make_render_plan(self.payload(), 60)
        texts, _, _, groups = prepare_plan_inputs(plan)
        with self.assertRaisesRegex(ValueError, 'thiếu mốc từ'):
            strict_visual_edl(texts, [4], groups, [[]])
        cues = [[{'start': 0, 'end': 1.8, 'content': 'Nobita mở cửa'},
                 {'start': 2, 'end': 4, 'content': 'Sau đó cậu bước vào nhà'}]]
        rows = strict_visual_edl(texts, [4], groups, cues)
        self.assertAlmostEqual(sum(r['duration'] for r in rows), 4)
        self.assertEqual([r['start'] for r in rows], [12, 24])
        with self.assertRaisesRegex(ValueError, 'thiếu hình'):
            strict_visual_edl(texts, [20], groups, cues)

    def test_extension_fills_speech_but_stops_at_next_locked_shot_and_eof(self):
        from video_renderer import build_visual_edit_plan
        plan = build_visual_edit_plan([10, 3], [0, 12], [6, 15], 20,
                                      strict_visual_lock=True, extend_to_next_shot=True)
        self.assertEqual(plan[0]['source_duration'], 10)
        self.assertEqual(plan[0]['hold_duration'], 0)
        capped = build_visual_edit_plan([15, 3], [0, 12], [6, 15], 20,
                                        strict_visual_lock=True, extend_to_next_shot=True)
        self.assertEqual(capped[0]['source_duration'], 12)
        self.assertEqual(capped[0]['visual_window_end'], 12)
        eof = build_visual_edit_plan([8], [16], [18], 20,
                                     strict_visual_lock=True, extend_to_next_shot=True)
        self.assertEqual(eof[0]['source_duration'], 4)
        rows = strict_visual_edl(['Lời đọc dài hơn hình.'], [10],
                                [[{'text': 'Lời đọc dài hơn hình.', 'start': 0, 'end': 6}]],
                                [[]], source_duration=20)
        self.assertEqual(rows[0]['end'], 10)
        with self.assertRaisesRegex(ValueError, 'cuối video'):
            strict_visual_edl(['Lời đọc dài hơn hình.'], [10],
                             [[{'text': 'Lời đọc dài hơn hình.', 'start': 16, 'end': 18}]],
                             [[]], source_duration=20)

    def test_bounded_recovery_has_three_recovery_rounds_no_split(self):
        failed = [(0, ('voice', [(0, 'test paragraph')]), RuntimeError('initial failure'))]
        async def run():
            with patch('tts_engine.process_tts_group', new=AsyncMock(side_effect=RuntimeError('service down'))) as process, \
                 patch('tts_engine.asyncio.sleep', new=AsyncMock()) as sleep, \
                 patch('tts_engine._recover_tts_group_as_short_pieces', new=AsyncMock()) as split, \
                 patch('tts_engine._invalidate_tts_group_cache', return_value=False):
                recovered, pending, _ = await _recover_failed_tts_groups(
                    failed, '+20%', '.', 'ffmpeg', recovery_delays=(0, 0, 180), split_failed=False)
                self.assertEqual(recovered, [])
                self.assertEqual(len(pending), 1)
                self.assertEqual(process.await_count, 3)
                self.assertEqual([c.args[0] for c in sleep.await_args_list], [0, 0, 180])
                split.assert_not_awaited()
        asyncio.run(run())

    def test_cache_changes_on_voice_settings_plan_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            source.write_bytes(b'source')
            plan = make_render_plan(self.payload(), 60)
            def key(**kwargs):
                return cache_key(kwargs.get('plan', plan), str(source), '', .12,
                                 kwargs.get('voice', 'voice'), '+20%', kwargs.get('options', {}))
            before = key()
            self.assertNotEqual(before, key(voice='another'))
            self.assertNotEqual(before, key(options={'mirror': True}))
            edited = copy.deepcopy(plan)
            edited['situations'][0]['text'] += ' Cậu nhìn quanh.'
            self.assertNotEqual(before, key(plan=edited))
            source.write_bytes(b'different source')
            self.assertNotEqual(before, key())

    def test_frozen_legacy_snapshot_is_intact(self):
        backup = Path(__file__).resolve().parents[1] / '.render_backups' / 'legacy_before_structured_v2'
        manifest = json.loads((backup / 'manifest.json').read_text(encoding='utf-8'))
        for name, digest in manifest.items():
            self.assertEqual(hashlib.sha256((backup / name).read_bytes()).hexdigest(), digest, name)

    def test_worker_routes_to_requested_engine_without_silent_fallback(self):
        source = Path(__file__).resolve().parents[1] / 'app.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render_video_bg')
        legacy = Mock(return_value='/static/output/legacy.mp4')
        updates = Mock()
        env = {'os': os, 'time': time, 'traceback': traceback, 'WORKSPACE_DIR': '.',
               'render_review_video_ffmpeg': legacy, 'export_capcut_package': Mock(),
               'append_task_log': Mock(), '_task_log_path': lambda task: 'test.log',
               'update_task_fields': updates,
               'copy_result_to_export_directory': lambda *args: ('saved.mp4', '')}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     '<render dispatch>', 'exec'), env)
        args = ('task', 'source.mp4', '', .12, 'voice', '+20%', ['old timeline'])
        env['render_video_bg'](*args, {}, 'full_mp4')
        legacy.assert_called_once()
        legacy.reset_mock()
        plan = {'version': 2, 'situations': []}
        with patch('structured_render.render_structured_video', return_value='/static/output/new.mp4') as new:
            env['render_video_bg'](*args, {'render_engine': 'structured_v2', 'structured_plan': plan}, 'full_mp4')
            self.assertIs(new.call_args.args[6], plan)
        legacy.assert_not_called()
        with patch('structured_render.render_structured_video', side_effect=ValueError('missing timing')):
            env['render_video_bg'](*args, {'render_engine': 'structured_v2', 'structured_plan': plan}, 'full_mp4')
        legacy.assert_not_called()
        self.assertEqual(updates.call_args.kwargs['status'], 'failed')
        self.assertIn('Render cũ', updates.call_args.kwargs['message'])


if __name__ == '__main__':
    unittest.main()
