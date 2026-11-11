import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from recap_visual_media import evidence_manifest, visual_batches, build_visual_pack, inspect_writing_batch, parse_writing_response, evidence_writing_batches


class RecapVisualMediaTests(unittest.TestCase):
    def test_writing_parser_unwraps_all_complete_scene_containers_without_inventing_ids(self):
        rows = [{'event_id': 'E0017', 'tts': 'Doraemon mở cửa.'},
                {'event_id': 'E0018', 'tts': 'Nobita bước vào.'}]
        for raw in ([{'scenes': rows}], [{'scenes': rows[:1]}, {'scenes': rows[1:]}],
                    {'parts': [{'scenes': rows[:1]}, {'scenes': rows[1:]}]}):
            actual, report = inspect_writing_batch(parse_writing_response(json.dumps(raw)), ['E0017', 'E0018'])
            self.assertEqual(actual, rows)
            self.assertTrue(report['valid'])
        _, report = inspect_writing_batch(parse_writing_response(json.dumps([
            {'scenes': [rows[0], rows[0]]}])), ['E0017', 'E0018'])
        self.assertFalse(report['valid'])
        self.assertEqual(report['duplicate_ids'], ['E0017'])
        _, report = inspect_writing_batch(parse_writing_response('[{"scenes":[{"tts":"Missing ID"}]}]'), ['E0017'])
        self.assertFalse(report['valid'])
        with self.assertRaisesRegex(ValueError, 'mixed'):
            parse_writing_response(json.dumps([{'scenes': rows}, rows[0]]))

    def test_text_batches_ignore_large_backend_metadata_but_bound_evidence_size(self):
        cards = [dict(card, event_id=f'E{i + 1:04d}', narration_focus='Mở cửa',
                      observed_facts=[{'text': 'Nobita mở cửa.'}],
                      coverage_registry='x' * 50000) for i, card in enumerate(self.cards(2))]
        batches = list(evidence_writing_batches(cards))
        self.assertEqual([len(b) for b in batches], [2])
        cards[0]['observed_facts'] = [{'text': 'x' * 25000}]
        self.assertEqual([len(b) for b in evidence_writing_batches(cards)], [1, 1])

    def test_writing_parser_keeps_every_row_in_direct_array(self):
        rows = [{'event_id': f'E{i:04d}', 'tts': 'A longer sentence' if i == 28 else 'Short',
                 'visual_links': [{'visual_id': 'V001', 'trigger_text': ''}]} for i in range(25, 31)]
        for raw in (json.dumps(rows), '```json\n' + json.dumps(rows) + '\n```',
                    'Here is the result:\n' + json.dumps(rows), json.dumps({'scenes': rows})):
            result, report = inspect_writing_batch(parse_writing_response(raw), [r['event_id'] for r in rows])
            self.assertEqual(result, rows)
            self.assertTrue(report['valid'])

    def test_writing_parser_preserves_validation_and_rejects_truncated_arrays(self):
        row = {'event_id': 'E0001', 'tts': 'First'}
        self.assertEqual(parse_writing_response(json.dumps([row])), {'scenes': [row]})
        for raw in ('[' + json.dumps(row) + ',', '{"scenes":[' + json.dumps(row), ''):
            with self.assertRaisesRegex(ValueError, 'VISUAL_WRITING_JSON_INVALID'):
                parse_writing_response(raw)
        _, report = inspect_writing_batch(parse_writing_response(json.dumps([row, row])), ['E0001', 'E0002'])
        self.assertFalse(report['valid'])
        self.assertEqual(report['duplicate_ids'], ['E0001'])
        self.assertEqual(report['missing_ids'], ['E0002'])

    def test_batch_diagnostic_identifies_missing_extra_duplicate_and_scene_only_ids(self):
        _, report = inspect_writing_batch({'scenes': [
            {'event_id': 'E0001'}, {'event_id': 'E0001'}, {'scene_id': 'S0002'},
            {'event_id': 'E9999'}, 'bad row']}, ['E0001', 'E0002'])
        self.assertFalse(report['valid'])
        self.assertEqual(report['actual_count'], 5)
        self.assertEqual(report['missing_ids'], ['E0002'])
        self.assertEqual(report['duplicate_ids'], ['E0001'])
        self.assertEqual(report['unexpected_ids'], ['E9999'])
        self.assertEqual(report['invalid_rows'][0]['scene_id'], 'S0002')
        self.assertEqual(report['invalid_rows'][1]['row'], 5)
        self.assertIn('scene_count_mismatch', report['errors'])

    def test_batch_diagnostic_handles_wrong_types_and_does_not_normalize(self):
        for payload in ([], {}, {'scenes': 'invalid'}, {'parts': [None]}):
            _, report = inspect_writing_batch(payload, ['E0001'])
            self.assertEqual(report['errors'], ['scenes_missing_or_not_list'])
        _, report = inspect_writing_batch({'scenes': [{'event_id': ['E0001']}]}, ['E0001'])
        self.assertEqual(report['missing_ids'], ['E0001'])
        rows, report = inspect_writing_batch({'scenes': [{'event_id': 'E0001'}]}, ['E0001'])
        self.assertTrue(report['valid'])
        _, report = inspect_writing_batch({'scenes': [{'event_id': 'e1'}]}, ['E0001'])
        self.assertFalse(report['valid'])

    def cards(self, count=2):
        return [{'scene_id': f'S{i + 1:04d}', 'visual_sources': [
            {'shot_id': f'SHOT_{i + 1:06d}', 'start': f'00:0{i}', 'end': f'00:0{i + 1}'}]}
                for i in range(count)]

    def test_manifest_uses_global_source_offset_and_preserves_ids(self):
        cards = self.cards()
        original = copy.deepcopy(cards)
        rows = evidence_manifest(cards, 60)
        self.assertEqual([r['source_start'] for r in rows], [60, 61])
        self.assertEqual([r['pack_start'] for r in rows], [0, 1])
        self.assertEqual(rows[1]['shot_ids'], ['SHOT_000002'])
        self.assertEqual(cards, original)

    def test_invalid_labels_or_ranges_are_rejected_before_ffmpeg(self):
        for key, value in [('scene_id', "S1':x=0"), ('scene_id', '')]:
            cards = self.cards()
            cards[0][key] = value
            with self.assertRaises(ValueError):
                evidence_manifest(cards)
        for a, b in [(-1, 1), (0, 0), (0, 16), (float('nan'), 1)]:
            cards = self.cards()
            cards[0]['visual_sources'][0].update(start_seconds=a, end_seconds=b)
            with self.assertRaises(ValueError):
                evidence_manifest(cards)

    def test_batch_limits_preserve_every_scene_and_shot(self):
        cards = self.cards(8)
        batches = list(visual_batches(cards, max_scenes=3, max_seconds=2, max_shots=3))
        self.assertEqual([len(b) for b in batches], [2, 2, 2, 2])
        self.assertEqual([c for batch in batches for c in batch], cards)

    def test_real_ffmpeg_pack_decodes_and_cache_is_reused(self):
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            self.skipTest('imageio_ffmpeg unavailable')
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source.mp4'
            result = subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=10:duration=3',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output, manifest = build_visual_pack(str(source), str(Path(root) / 'cache'), self.cards(), ffmpeg)
            self.assertEqual(len(manifest), 2)
            self.assertGreater(Path(output).stat().st_size, 1000)
            before = Path(output).stat().st_mtime_ns
            result = subprocess.run([ffmpeg, '-v', 'error', '-i', output, '-f', 'null', '-'], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            cached, _ = build_visual_pack(str(source), str(Path(root) / 'cache'), self.cards(), ffmpeg)
            self.assertEqual(output, cached)
            self.assertEqual(Path(cached).stat().st_mtime_ns, before)


if __name__ == '__main__':
    unittest.main()
