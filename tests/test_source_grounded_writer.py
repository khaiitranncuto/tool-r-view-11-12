import copy
import json
import unittest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch

from source_grounded_writer import bind_written_scenes, is_source_grounded, writing_prompt, write_validated_scenes
from recap_pipeline import coerce_single_part_response, clamp_payload_narrations, merge_single_part_responses
from recap_visual_media import evidence_manifest
from transcript_pipeline import filter_transcript_evidence


class SourceWriterTests(unittest.TestCase):
    def alias_fixture(self):
        response = self.response()
        response['scenes'][0]['sentences'][0]['text'] = 'Jaian m\u1edf c\u1eeda ph\u00f2ng.'
        response['scenes'][0]['sentences'][1]['text'] = 'Jaian b\u01b0\u1edbc v\u00e0o trong.'
        return response, [{'id': 'P01', 'name': 'Gian', 'aliases': ['Jaian']}]

    def test_names_are_normalized_before_proof_and_anchor_lock(self):
        response, characters = self.alias_fixture()
        event = self.event()
        rows = bind_written_scenes(response, [event], characters=characters)
        scene = rows[0]
        self.assertNotIn('Jaian', scene['tts'])
        self.assertTrue(all(p['text'].startswith('Gian ') for p in scene['sentence_proofs']))
        self.assertTrue(scene['visual_links'][1]['trigger_text'].startswith('Gian '))
        self.assertIn(scene['visual_links'][1]['trigger_text'], scene['tts'])
        payload = json.loads(coerce_single_part_response(json.dumps({'scenes': rows}), 1, 12, 40,
            locked_event_by_id={'E0001': event}, locked_characters=characters))
        normalized = clamp_payload_narrations(payload)
        self.assertTrue(is_source_grounded(normalized['parts'][0]['scenes'][0]))
        self.assertEqual(clamp_payload_narrations(normalized), normalized)

    def test_cached_locks_survive_alias_and_visual_cleanup(self):
        response, characters = self.alias_fixture()
        event = self.event()
        rows = bind_written_scenes(response, [event])  # Old cache predates name normalization.
        payload = json.loads(coerce_single_part_response(json.dumps({'scenes': rows}), 1, 12, 40,
            locked_event_by_id={'E0001': event}, locked_characters=characters))
        normalized = clamp_payload_narrations(payload)
        scene = normalized['parts'][0]['scenes'][0]
        self.assertEqual(scene, payload['parts'][0]['scenes'][0])
        self.assertTrue(is_source_grounded(scene))
        responses = [coerce_single_part_response(json.dumps({'scenes': rows}), i, 12, 40,
            locked_event_by_id={'E0001': event}, locked_characters=characters) for i in (1, 2, 3)]
        merged = merge_single_part_responses(responses, [0, 12, 24, 36], 120)
        final = clamp_payload_narrations(merged)
        self.assertTrue(all(is_source_grounded(s) for p in final['parts'] for s in p['scenes']))

    def test_clamp_rejects_changed_locks_instead_of_resigning_them(self):
        event = self.event()
        rows = bind_written_scenes(self.response(), [event])
        payload = json.loads(coerce_single_part_response(json.dumps({'scenes': rows}), 1, 12, 40,
            locked_event_by_id={'E0001': event}))
        for field in ('tts', 'locked_visual_shots', 'sentence_proofs'):
            changed = copy.deepcopy(payload)
            scene = changed['parts'][0]['scenes'][0]
            if field == 'tts':
                scene[field] += ' Changed.'
            else:
                scene[field] = []
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'SOURCE_WRITER_LOCK_CHANGED'):
                clamp_payload_narrations(changed)

    def test_prompt_only_contains_the_proof_object_schema(self):
        prompt = writing_prompt([self.event()], {}, [])
        self.assertNotIn('"sentences":["Cau thu nhat."]', prompt)
        self.assertIn('"sentences":[{"text":', prompt)

    def test_malformed_writer_shapes_raise_validation_errors(self):
        for payload in ('text', {'scenes': 'text'}, {'scenes': ['text']}):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, 'SOURCE_WRITER_SCHEMA_INVALID'):
                bind_written_scenes(payload, [self.event()])
        for unit in ('A sentence.', None, {'text': 'A sentence.', 'fact_ids': [['F1']], 'visual_id': 'V001'},
                     {'text': 'A sentence.', 'fact_ids': ['F1'], 'visual_id': ['V001']}):
            payload = self.response()
            payload['scenes'][0]['sentences'][0] = unit
            with self.subTest(unit=unit), self.assertRaisesRegex(ValueError, 'SOURCE_WRITER_SCHEMA_INVALID'):
                bind_written_scenes(payload, [self.event()])

    def test_string_sentences_are_repaired_without_inventing_proof(self):
        bad = self.response()
        bad['scenes'][0]['sentences'] = [unit['text'] for unit in bad['scenes'][0]['sentences']]
        generate = unittest.mock.Mock(side_effect=[json.dumps(bad), json.dumps(self.response())])
        save = unittest.mock.Mock()
        rows = write_validated_scenes([self.event()], [], generate, save)
        self.assertEqual(generate.call_count, 2)
        self.assertIn('SOURCE_WRITER_SCHEMA_INVALID', generate.call_args.args[1])
        self.assertEqual(save.call_count, 2)
        self.assertTrue(is_source_grounded(rows[0]))

    def test_invalid_cached_candidate_can_be_repaired(self):
        bad = self.response()
        bad['scenes'][0]['sentences'] = ['First sentence.', 'Second sentence.']
        generate = unittest.mock.Mock(return_value=json.dumps(self.response()))
        self.assertTrue(is_source_grounded(write_validated_scenes([self.event()], [bad], generate)[0]))
        self.assertEqual(generate.call_count, 1)

    def test_invalid_proof_retries_are_bounded_and_valid_cache_is_reused(self):
        bad = self.response()
        bad['scenes'][0]['sentences'][0]['fact_ids'] = ['FOREIGN']
        generate = unittest.mock.Mock(return_value=json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'SOURCE_WRITER_PROOF_INVALID'):
            write_validated_scenes([self.event()], [], generate)
        self.assertEqual(generate.call_count, 2)
        generate.reset_mock()
        rows = write_validated_scenes([self.event()], [self.response(), bad], generate)
        self.assertTrue(is_source_grounded(rows[0]))
        generate.assert_not_called()

    def test_time_and_space_are_not_character_gian(self):
        from recap_pipeline import character_name_in_text
        self.assertFalse(character_name_in_text('Gian', 'Doraemon dùng cỗ máy thời gian.'))
        self.assertFalse(character_name_in_text('Gian', 'Họ vượt qua không gian.'))
        self.assertTrue(character_name_in_text('Gian', 'Gian chạy đến giúp Nobita.'))

    def event(self):
        return {'event_id': 'E0001', 'scene_id': 'S0001', 'sentence_quota': 2,
                'situation_narration': True, 'source_link_contract_version': 1,
                'grounded_plan_locked': True, 'narration_focus': 'Open and enter',
                'observed_facts': [{'fact_id': 'F1', 'text': 'Open door', 'shot_ids': ['SHOT_000001']},
                                   {'fact_id': 'F2', 'text': 'Enter room', 'shot_ids': ['SHOT_000002']}],
                'visual_sources': [{'shot_id': 'SHOT_000001', 'fact_ids': ['F1'], 'event': 'Open door',
                                    'start': '00:00', 'end': '00:06'},
                                   {'shot_id': 'SHOT_000002', 'fact_ids': ['F2'], 'event': 'Enter room',
                                    'start': '00:06', 'end': '00:12'}]}

    def response(self):
        return {'scenes': [{'event_id': 'E0001', 'sentences': [
            {'text': 'Nobita mở cửa phòng.', 'fact_ids': ['F1'], 'visual_id': 'V001'},
            {'text': 'Cậu bước vào bên trong.', 'fact_ids': ['F2'], 'visual_id': 'V002'}]}]}

    def test_freezes_prose_and_backend_links_through_real_serializers(self):
        event = self.event()
        rows = bind_written_scenes(self.response(), [event])
        self.assertTrue(is_source_grounded(rows[0]))
        self.assertEqual(rows[0]['visual_links'][1]['trigger_text'], 'Cậu bước')
        raw = coerce_single_part_response(json.dumps({'scenes': rows}), 1, 12, 40,
                                         locked_event_by_id={'E0001': event})
        payload = clamp_payload_narrations(json.loads(raw))
        scene = payload['parts'][0]['scenes'][0]
        self.assertTrue(is_source_grounded(scene))
        responses = [coerce_single_part_response(json.dumps({'scenes': rows}), i, 12, 40,
                     locked_event_by_id={'E0001': event}) for i in (1, 2, 3)]
        merged = merge_single_part_responses(responses, [0, 12, 24, 36], 120)
        self.assertTrue(is_source_grounded(merged['parts'][0]['scenes'][0]))
        scene['tts'] += ' Changed.'
        self.assertFalse(is_source_grounded(scene))

    def test_foreign_proof_and_unseen_action_are_rejected_before_lock(self):
        for key, value in [('fact_ids', ['FOREIGN']), ('visual_id', 'V999'), ('visual_id', 'V002')]:
            response = self.response()
            response['scenes'][0]['sentences'][0][key] = value
            with self.assertRaises(ValueError):
                bind_written_scenes(response, [self.event()])
        response = self.response()
        response['scenes'][0]['sentences'][0]['evidence_kind'] = 'dialogue'
        with self.assertRaisesRegex(ValueError, 'DIALOGUE_MISSING'):
            bind_written_scenes(response, [self.event()])

    def test_evidence_pack_keeps_visual_ids_and_can_include_full_situation(self):
        event = self.event()
        event['locked_visual_shots'] = [dict(event['visual_sources'][1], visual_id='V002')]
        selected = evidence_manifest([event])
        self.assertEqual(selected[0]['visual_id'], 'V002')
        full = evidence_manifest([dict(event, evidence_scope='situation')])
        self.assertEqual([r['shot_ids'][0] for r in full], ['SHOT_000001', 'SHOT_000002'])

    def test_sparse_repeated_whisper_noise_is_not_used_as_story(self):
        rows = [{'start_seconds': i * 30, 'end_seconds': (i + 1) * 30,
                 'text': 'おめでとうございます。'} for i in range(3)]
        rows.append({'start_seconds': 91, 'end_seconds': 93, 'text': 'We should go home.'})
        clean, report = filter_transcript_evidence(rows)
        self.assertEqual([r['text'] for r in clean], ['We should go home.'])
        self.assertEqual(report['removed_noise_segments'], 3)

    def test_clip_is_reused_across_api_switch_without_reencoding(self):
        from grounded_pipeline import indexed_clip
        from tts_engine import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / 'source.mp4')
            output = str(Path(directory) / 'indexed.mp4')
            generated = subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'lavfi', '-i', 'color=c=blue:s=160x90:r=2:d=3',
                '-c:v', 'libx264', source], capture_output=True)
            self.assertEqual(generated.returncode, 0)
            indexed_clip(source, output, 0, 2, ffmpeg, fps=2)
            with patch('grounded_pipeline.subprocess.run', side_effect=AssertionError('clip recut')):
                self.assertEqual(indexed_clip(source, output, 0, 2, ffmpeg, fps=2), output)

    def test_excess_sentences_are_compounded_locally_without_changing_words(self):
        from narrative_contract import words, sentences
        response = self.response()
        response['scenes'][0]['sentences'].append(
            {'text': 'Cậu đứng lại trong phòng.', 'fact_ids': ['F2'], 'visual_id': 'V002'})
        result = bind_written_scenes(response, [self.event()])[0]
        self.assertEqual(len(sentences(result['tts'])), 2)
        self.assertEqual(words(result['tts']), sum(words(u['text']) for u in response['scenes'][0]['sentences']))


if __name__ == '__main__':
    unittest.main()
