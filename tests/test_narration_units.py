import copy
import json
import unittest

from narration_units import plan_narration_units, unit_response
from evidence_writer import build_sentence_plan, bind_planned_scenes, write_batch
from flat_writer import flat_prompt, flat_schema, flat_to_paragraphs
from narrative_contract import verify_script, sentences
from recap_pipeline import coerce_single_part_response, merge_single_part_responses, clamp_payload_narrations
from sentence_visual_contract import enforce_sentence_visual_contract, preflight_visual_plan
from source_grounded_writer import is_source_grounded


class NarrationUnitTests(unittest.TestCase):
    def plan(self):
        e = dict(event_id='E0001', scene_id='S0001', sentence_quota=1,
            start_seconds=0, end_seconds=12, grounded_plan_locked=True,
            observed_facts=[dict(fact_id='F1', text='Kukuru nhảy ra khỏi tủ.', shot_ids=['SHOT_000001']),
                dict(fact_id='F2', text='Cậu lao về phía nhóm bạn.', shot_ids=['SHOT_000002'])],
            visual_sources=[dict(shot_id='SHOT_000001', fact_ids=['F1'], start_seconds=0, end_seconds=6),
                dict(shot_id='SHOT_000002', fact_ids=['F2'], start_seconds=6, end_seconds=12)])
        return plan_narration_units(build_sentence_plan(dict(events_by_part=[[e]],
            sentence_contract={'target_count': 1}, scene_contract={})))

    def response(self):
        return {'sentences': [dict(event_id='E0001', unit_id='E0001_U001',
            text='Kukuru nhảy ra khỏi tủ. Cậu lao về phía nhóm bạn.')]}

    def test_one_unit_can_have_two_grammatical_sentences_and_two_shots(self):
        plan = self.plan()
        self.assertTrue(preflight_visual_plan(plan, 12)['passed'])
        e = plan['events_by_part'][0][0]
        self.assertEqual(e['sentence_slots'][0]['fact_ids'], ['F1', 'F2'])
        scene = bind_planned_scenes(unit_response(self.response(), [e]), [e])[0]
        self.assertEqual(len(sentences(scene['tts'])), 2)
        self.assertEqual(len(scene['sentence_proofs']), 1)
        self.assertEqual(len(scene['locked_visual_shots']), 2)
        self.assertTrue(is_source_grounded(scene))

    def test_unit_ids_survive_normalization_merge_and_export(self):
        e = self.plan()['events_by_part'][0][0]
        scene = bind_planned_scenes(unit_response(self.response(), [e]), [e])[0]
        normalized = coerce_single_part_response(json.dumps({'scenes': [scene]}), 1, 12, 40,
            locked_event_by_id={'E0001': e})
        payload = merge_single_part_responses([normalized], [0, 12], 40, single_timeline=True)
        payload = clamp_payload_narrations(payload)
        s = payload['parts'][0]['scenes'][0]
        self.assertTrue(is_source_grounded(s))
        s.update(grounding_audit_passed=True, grounding_audit_pending=False)
        report = enforce_sentence_visual_contract(payload, 1, 12)
        self.assertTrue(report['passed'])
        self.assertEqual(verify_script(s['tts'], 1, 40, unit_count=report['actual'])[0], 1)
        self.assertEqual(verify_script(s['tts'], 1, 40, unit_texts=[s['tts']])[0], 1)
        with self.assertRaisesRegex(ValueError, 'UNIT_TEXT_MISMATCH'):
            verify_script('Kukuru nhảy ra.', 1, 40, unit_texts=[s['tts']])

    def test_missing_duplicate_and_unknown_unit_ids_are_rejected(self):
        e = self.plan()['events_by_part'][0][0]
        for rows in [[], [dict(event_id='E0001', unit_id='BAD', text='Có hình.')], self.response()['sentences'] * 2]:
            mapped = unit_response({'sentences': rows}, [e])
            self.assertIn('IDS_INVALID', mapped['scenes'][0]['format_error'])

    def test_dispatch_and_cache_use_units_without_punctuation_count(self):
        e = self.plan()['events_by_part'][0][0]
        self.assertIn('unit_id', flat_schema([e])['properties']['sentences']['items']['required'])
        prompt = flat_prompt([e], {})
        self.assertIn('KHONG DEM BANG DAU CHAM/PHAY', prompt)
        saved, raw = [], []
        rows = write_batch([e], {}, lambda *_: json.dumps(self.response()),
            prompt_builder=flat_prompt, response_adapter=flat_to_paragraphs,
            save=lambda d: saved.extend(d), save_raw=lambda text, _: raw.append(text))
        again = write_batch([e], {}, lambda *_: self.fail('cache must be reused'), candidates=saved,
            prompt_builder=flat_prompt, response_adapter=flat_to_paragraphs)
        self.assertEqual(rows, again)
        self.assertEqual(json.loads(raw[0]), self.response())

    def test_english_cached_draft_is_repaired_in_vietnamese(self):
        event = self.plan()['events_by_part'][0][0]
        english = self.response()
        english['sentences'][0]['text'] = (
            'The boy jumps out of the cabinet and runs toward his friends.')
        bad_cache = unit_response(english, [event])
        prompts = []
        def generate(events, prompt):
            prompts.append(prompt)
            return json.dumps(self.response())
        rows = write_batch([event], {}, generate, candidates=[bad_cache],
            prompt_builder=flat_prompt, response_adapter=flat_to_paragraphs)
        self.assertEqual(len(prompts), 1)
        self.assertIn('NGÔN NGỮ BẮT BUỘC', prompts[0])
        self.assertIn('RECAP_LANGUAGE_INVALID', prompts[0])
        self.assertEqual(rows[0]['tts'], self.response()['sentences'][0]['text'])

    def test_persistently_english_output_is_not_accepted(self):
        event = self.plan()['events_by_part'][0][0]
        english = self.response()
        english['sentences'][0]['text'] = (
            'The boy jumps out of the cabinet and runs toward his friends.')
        calls = []
        def generate(events, prompt):
            calls.append(prompt)
            return json.dumps(english)
        with self.assertRaisesRegex(ValueError, 'RECAP_LANGUAGE_INVALID'):
            write_batch([event], {}, generate, prompt_builder=flat_prompt,
                        response_adapter=flat_to_paragraphs)
        self.assertEqual(len(calls), 2)

    def test_bad_raw_response_is_retained_before_adapter_rejects_it(self):
        e = self.plan()['events_by_part'][0][0]
        raw = []
        with self.assertRaises(ValueError):
            write_batch([e], {}, lambda *_: 'broken JSON', prompt_builder=flat_prompt,
                response_adapter=flat_to_paragraphs, save_raw=lambda text, _: raw.append(text))
        self.assertEqual(raw, ['broken JSON', 'broken JSON'])

    def test_absent_assigned_visual_blocks_export(self):
        e = self.plan()['events_by_part'][0][0]
        scene = bind_planned_scenes(unit_response(self.response(), [e]), [e])[0]
        scene.update(sentence_quota=1, grounded_plan_locked=True, grounding_audit_passed=True)
        scene['locked_visual_shots'].pop()
        with self.assertRaises(ValueError):
            enforce_sentence_visual_contract({'parts': [{'scenes': [scene]}]}, 1, 12)
