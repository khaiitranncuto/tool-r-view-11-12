"""Exercise evidence-first writing and selective visual repair offline."""
import ast
import copy
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from recap_pipeline import build_single_part_video_prompt, build_targeted_narration_repair_prompt
from recap_visual_media import visual_batches, evidence_manifest, evidence_writing_batches


class ImageFirstWritingTests(unittest.TestCase):
    def test_actual_app_sends_flat_schema_and_binds_flat_response(self):
        from source_grounded_writer import is_source_grounded
        write, events, _, prepared = self.writer(situations=True, quotas=True)
        for i, e in enumerate(events):
            e.update(source_link_contract_version=1, evidence_sentence_plan_version=1, paragraph_plan_version=2)
            e['observed_facts'][0].update(fact_id=f'F{i}', shot_ids=[f'SHOT_{i + 1:06d}'])
            e['sentence_slots'] = [{'slot_id': f'{e["event_id"]}_C001'}]
        calls = []
        def generate(client, model, contents, config, label, **kwargs):
            state = write.__globals__['TASKS']['offline']['active_writing_context']
            self.assertEqual(state['writer_format'], 'flat')
            self.assertEqual(config.response_json_schema['required'], ['sentences'])
            self.assertNotIn('enum', json.dumps(config.response_json_schema))
            calls.append(state['event_ids'])
            return json.dumps({'sentences': [dict(event_id=e['event_id'], text='Nobita mở cửa phòng.',
                fact_ids=[e['observed_facts'][0]['fact_id']]) for e in events if e['event_id'] in state['event_ids']]})
        write.__globals__['generate_gemini_text_with_recovery'] = generate
        with patch.dict('os.environ', {'RECAP_WRITER_FORMAT': 'flat'}):
            first = json.loads(write(False, [], None, None))
            again = json.loads(write(False, [], None, None))
        self.assertEqual(first, again)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(prepared), 0)
        self.assertTrue(all(is_source_grounded(r) for r in first['scenes']))

    def test_planned_writer_is_text_only_and_resumes_without_uploading_video(self):
        write, events, _, prepared = self.writer(situations=True, quotas=True)
        for i, event in enumerate(events):
            event['source_link_contract_version'] = 1
            event['evidence_sentence_plan_version'] = 1
            event['paragraph_plan_version'] = 2
            event['observed_facts'][0].update(fact_id=f'F{i}', shot_ids=[f'SHOT_{i + 1:06d}'])
            event['sentence_slots'] = [{'slot_id': f'{event["event_id"]}_C001'}]
        requests = []
        def generate(client, model, contents, config, label, **kwargs):
            self.assertEqual(len(contents), 1)
            self.assertIsInstance(contents[0], str)
            self.assertIn('MOT DOAN KE LIEN MACH', contents[0])
            self.assertNotIn('MANIFEST:', contents[0])
            self.assertIsNotNone(config.response_json_schema)
            self.assertEqual(kwargs['capacity_max_retries'], 0)
            ids = write.__globals__['TASKS']['offline']['active_writing_context']['event_ids']
            requests.append(ids)
            return json.dumps({'scenes': [{'event_id': e['event_id'], 'paragraph': 'Nobita mở cửa phòng.',
                'sentence_evidence': [{'fact_ids': [e['observed_facts'][0]['fact_id']], 'evidence_kind': 'visual'}]}
                for e in events if e['event_id'] in ids]})
        write.__globals__['generate_gemini_text_with_recovery'] = generate
        result = json.loads(write(False, [], None, None))
        self.assertEqual(len(result['scenes']), 20)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(prepared), 0)

        # A damaged bound cache can be rebuilt from the validated raw draft
        # without resending media or consuming another model request.
        task = write.__globals__['TASKS']['offline']
        cache_key = next(iter(task['evidence_writing_batches']))
        task['evidence_writing_batches'][cache_key][0]['tts'] += ' Changed.'
        repaired = json.loads(write(False, [], None, None))
        self.assertEqual(repaired, result)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(prepared), 0)
        self.assertTrue(all(r.get('source_grounded_signature') for r in result['scenes']))
        again = json.loads(write(False, [], None, None))
        self.assertEqual(again, result)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(prepared), 0)

    def writer(self, fail_second=False, situations=False, quotas=False, alter_mapping=False):
        # Compile the actual nested function inside its counter closure. This
        # avoids starting the Flask worker or needing any Gemini credentials.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'write_visual_batches')
        factory = ast.parse('def factory():\n    pass1_calls = 0\n').body[0]
        factory.body.extend([node, ast.Return(value=ast.Name(id='write_visual_batches', ctx=ast.Load()))])
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        events = [{'event_id': f'E{i:04d}', 'scene_id': f'S{i:04d}',
            'grounded_plan_locked': True, 'narration_focus': 'Nobita mo cua',
            'observed_facts': [{'text': 'Nobita mo cua'}], 'visual_sources': [
                {'shot_id': f'SHOT_{i:06d}', 'start': f'00:{i:02d}', 'end': f'00:{i + 1:02d}'}]}
                  for i in range(1, 21)]
        if situations:
            for event in events:
                event['situation_narration'] = True
                if quotas:
                    event.update(sentence_quota=1, tts_target_words=20, tts_min_words=15, tts_max_words=25)
        task, calls, prepared, current = {}, [], [], []
        draft_calls = []
        chapter_calls = []
        media = object()
        def prepare(cards, offset, label):
            current[:] = cards
            prepared.append([e['event_id'] for e in cards])
            return media, evidence_manifest(cards, offset), 'pack:' + ','.join(prepared[-1])
        def generate(client, model, contents, config, label, **kwargs):
            ids = task['active_writing_context']['event_ids']
            if isinstance(contents[0], str) and 'VIET CHUONG RECAP THEO MACH TRUYEN' in contents[0]:
                self.assertNotIn('"visuals"', contents[0])
                chapter_calls.append(list(ids))
                task['test_chapter_calls'] = chapter_calls
                return json.dumps({'scenes': [{'event_id': eid,
                    'sentences': [' '.join(['word'] * 20) + '.']} for eid in ids]})
            if isinstance(contents[0], str) and 'BUOC 1: VIET DOAN TRUYEN LIEN MACH' in contents[0]:
                draft_calls.append(list(ids))
                return json.dumps({'paragraphs': ['Nobita mo cua, roi buoc vao phong.']})
            if len(contents) == 2:
                self.assertIs(contents[0], media)
                self.assertIn('MANIFEST:', contents[1])
            else:
                self.assertEqual(len(contents), 1)
                self.assertIn('CHI DIEN DAT BANG CHUNG DA KHOA', contents[0])
                if ids in draft_calls:
                    self.assertIn('DOAN KE LIEN MACH', contents[0])
                    self.assertIn('Nobita mo cua, roi buoc vao phong.', contents[0])
            calls.append(ids)
            task['test_draft_calls'] = draft_calls
            if fail_second and len(calls) == 2:
                raise RuntimeError('simulated transport failure')
            if isinstance(contents[0], str) and 'BUOC GHEP HINH:' in contents[0]:
                prose = json.loads(contents[0].rsplit('\n', 1)[-1])
                return json.dumps({'scenes': [dict(r, tts=r['tts'] + (' Extra.' if alter_mapping else ''),
                    visual_links=[{'visual_id': 'V001', 'trigger_text': ''}]) for r in prose]})
            return json.dumps({'scenes': [{'event_id': e['event_id'], 'tts': 'Nobita mo cua',
                'visual_links': [{'visual_id': 'V001', 'trigger_text': ''}]} for e in events if e['event_id'] in ids]})
        def update(task_id, **fields):
            task.update(fields)
        env = dict(copy=copy, json=json, required_events=events, pass1_calls=0,
            visual_batches=visual_batches, evidence_writing_batches=evidence_writing_batches,
            story_context={'required_events': events}, story_plan={'characters': []},
            build_single_part_video_prompt=build_single_part_video_prompt,
            build_targeted_narration_repair_prompt=build_targeted_narration_repair_prompt,
            part_number=2, part_index=1, boundaries=[0, 30, 60, 90], style='normal', custom_prompt='',
            enable_multi_voice=False, ensure_visual_pack=prepare, full_video_outline={},
            selected_model='offline-model', thinking_effort='low', task_lock=threading.Lock(),
            TASKS={'offline': task}, task_id='offline', visual_client=object(), client=object(),
            writing_config=__import__('google.genai.types', fromlist=['GenerateContentConfig']).GenerateContentConfig(),
            generate_gemini_text_with_recovery=generate, progress_base=40, update_prog=lambda *a: None,
            record_gemini_usage=lambda *a: None, gemini_quota_pacer=None,
            is_gemini_capacity_503_error=lambda exc: '503' in str(exc),
            estimate_gemini_video_input_tokens=lambda *a: 100, update_task_fields=update,
            append_task_log=lambda *a: None, parse_json_object=json.loads)
        exec(compile(module, '<app image-first writer>', 'exec'), env)
        return env['factory'](), events, calls, prepared

    def test_writing_uses_text_evidence_and_resumes_completed_batches(self):
        write, events, calls, prepared = self.writer(fail_second=True)
        with self.assertRaisesRegex(RuntimeError, 'transport'):
            write(False, [], None, None)
        result = json.loads(write(False, [], None, None))
        self.assertEqual(len(result['scenes']), 20)
        self.assertEqual(calls, [[e['event_id'] for e in events[:18]],
                                 ['E0019', 'E0020'], ['E0019', 'E0020']])
        self.assertEqual(prepared, [])
        # A failed allocation reuses the paragraph draft as well as completed rows.
        self.assertEqual(len(write.__globals__['TASKS']['offline']['test_draft_calls']), 2)

    def test_targeted_repair_sends_only_requested_scene_video(self):
        write, events, calls, prepared = self.writer()
        base = {'parts': [{'scenes': [{'scene_id': 'S0007', 'event_id': 'E0007',
            'covered_event_ids': ['E0007'], 'tts': 'wrong action'}]}]}
        requirements = {'invalid_timestamp_scenes': [
            {'scene_id': 'S0007', 'reasons': ['unsupported_narration:wrong action']}],
            'invalid_tts_scenes': [], 'invalid_character_mentions': [], 'missing_event_ids': []}
        result = json.loads(write(True, ['E0007'], json.dumps(base), requirements))
        self.assertEqual([r['event_id'] for r in result['scenes']], ['E0007'])
        self.assertEqual(calls, [['E0007']])
        self.assertEqual(prepared, [['E0007']])

    def test_situations_write_final_prose_without_draft_remapping_call(self):
        write, events, calls, prepared = self.writer(situations=True)
        result = json.loads(write(False, [], None, None))
        self.assertEqual(len(result['scenes']), 20)
        self.assertEqual(len(calls), 2)
        self.assertEqual(write.__globals__['TASKS']['offline']['test_draft_calls'], [])
        self.assertEqual(prepared, [])

    def test_exact_quota_prose_precedes_visual_mapping_and_is_immutable(self):
        write, _, calls, _ = self.writer(situations=True, quotas=True)
        result = json.loads(write(False, [], None, None))
        self.assertEqual(len(result['scenes']), 20)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(write.__globals__['TASKS']['offline']['test_chapter_calls']), 2)
        broken, _, _, _ = self.writer(situations=True, quotas=True, alter_mapping=True)
        with self.assertRaisesRegex(ValueError, 'VISUAL_MAPPING_CHANGED_LOCKED_PROSE'):
            broken(False, [], None, None)

    def test_length_repair_does_not_upload_video(self):
        write, events, calls, prepared = self.writer()
        base = {'parts': [{'scenes': [{'scene_id': 'S0007', 'event_id': 'E0007',
            'covered_event_ids': ['E0007'], 'tts': 'Nobita mo cua'}]}]}
        requirements = {'invalid_tts_scenes': [{'scene_id': 'S0007', 'word_count': 3}],
                        'invalid_timestamp_scenes': [], 'invalid_character_mentions': [], 'missing_event_ids': []}
        result = json.loads(write(True, ['E0007'], json.dumps(base), requirements))
        self.assertEqual([r['event_id'] for r in result['scenes']], ['E0007'])
        self.assertEqual(prepared, [])

    def test_missing_verifier_result_is_not_a_rewrite_request(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'repair_event_ids')
        env = {'parse_json_object': json.loads, 'locked_event_by_id': {'E0001': {}, 'E0002': {}}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<repair gate>', 'exec'), env)
        payload = json.dumps({'parts': [{'scenes': [
            {'scene_id': 'S0001', 'covered_event_ids': ['E0001']},
            {'scene_id': 'S0002', 'covered_event_ids': ['E0002']}]}]})
        requirements = {'invalid_timestamp_scenes': [
            {'scene_id': 'S0001', 'reasons': ['narration_audit_pending:missing_evidence_verdict']},
            {'scene_id': 'S0002', 'reasons': ['unsupported_narration:invented action']}]}
        self.assertEqual(env['repair_event_ids'](payload, requirements), ['E0002'])


if __name__ == '__main__':
    unittest.main()
