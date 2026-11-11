import copy
import unittest

from evidence_writer import bind_planned_scenes
from narrative_contract import allocate_sentences
from sentence_visual_contract import (inspect_sentence_visual_contract,
    enforce_sentence_visual_contract, preflight_visual_plan)
from source_grounded_writer import signature
from test_paragraph_narration import event, planned, response


class SentenceVisualContractTests(unittest.TestCase):
    def payload(self):
        events = planned([event()])
        scene = bind_planned_scenes(response(events), events)[0]
        scene.update(scene_id='S0001', sentence_quota=3, grounded_plan_locked=True,
                     grounding_audit_passed=True)
        return {'parts': [{'scenes': [scene]}]}

    def codes(self, payload, target=3, **kwargs):
        return {i['code'] for i in inspect_sentence_visual_contract(payload, target, **kwargs)['issues']}

    def test_all_sentences_have_source_proofs_and_exact_total(self):
        data = self.payload()
        original = copy.deepcopy(data)
        report = enforce_sentence_visual_contract(data, 3, 6)
        self.assertEqual(report['linked_sentence_count'], 3)
        self.assertTrue(report['passed'])
        self.assertEqual(data, original)

    def test_extra_or_missing_target_blocks_export(self):
        for target in (2, 4):
            self.assertIn('EXACT_SENTENCE_TARGET_MISMATCH', self.codes(self.payload(), target))
            with self.assertRaisesRegex(ValueError, 'EXPORT_BLOCKED'):
                enforce_sentence_visual_contract(self.payload(), target)

    def test_one_missing_sentence_image_blocks_whole_export(self):
        data = self.payload()
        scene = data['parts'][0]['scenes'][0]
        scene['sentence_proofs'][1]['visual_id'] = 'V_MISSING'
        scene['source_grounded_signature'] = signature(scene)
        self.assertIn('SENTENCE_VISUAL_MISSING', self.codes(data))

    def test_fallback_or_pending_audit_cannot_authorize_export(self):
        for change in [dict(visual_fallback={'reason': 'approximate'}),
                       dict(grounding_audit_pending=True), dict(grounding_audit_passed=False)]:
            data = self.payload()
            data['parts'][0]['scenes'][0].update(change)
            with self.assertRaisesRegex(ValueError, 'EXPORT_BLOCKED'):
                enforce_sentence_visual_contract(data, 3)

    def test_unrelated_or_outside_source_is_rejected_even_with_new_signature(self):
        data = self.payload()
        scene = data['parts'][0]['scenes'][0]
        scene['locked_visual_shots'][0]['shot_ids'] = ['SHOT_OTHER']
        scene['source_grounded_signature'] = signature(scene)
        self.assertIn('SENTENCE_VISUAL_UNRELATED', self.codes(data))
        data = self.payload()
        scene = data['parts'][0]['scenes'][0]
        scene['locked_visual_shots'][0]['end_seconds'] = 1000
        scene['source_grounded_signature'] = signature(scene)
        self.assertIn('SOURCE_WINDOW_INVALID', self.codes(data, duration=6))

    def test_wrong_proof_text_missing_video_and_edited_signature_block_export(self):
        data = self.payload()
        data['parts'][0]['scenes'][0]['sentence_proofs'][0]['text'] = 'Unsupported text.'
        self.assertIn('SENTENCE_PROOF_TEXT_MISMATCH', self.codes(data))
        self.assertIn('SOURCE_SIGNATURE_INVALID', self.codes(data))
        self.assertIn('SOURCE_VIDEO_MISSING', self.codes(self.payload(), source_path='missing-source-video.mp4'))

    def test_preflight_checks_inventory_before_writing(self):
        events = planned([event()])
        plan = dict(events_by_part=[events], sentence_contract={'target_count': 3})
        self.assertTrue(preflight_visual_plan(plan, 6)['passed'])
        plan['events_by_part'][0][0]['observed_facts'][0]['shot_ids'] = ['SHOT_OTHER']
        with self.assertRaisesRegex(ValueError, 'FACT_WITHOUT_SOURCE'):
            preflight_visual_plan(plan, 6)

    def test_wrong_source_times_cannot_hide_behind_valid_shot_ids(self):
        data = self.payload()
        scene = data['parts'][0]['scenes'][0]
        scene['locked_visual_shots'][0].update(start_seconds=100, end_seconds=106)
        catalog = {'SHOT_000001': {'start_seconds': 0, 'end_seconds': 6}}
        self.assertIn('SOURCE_WINDOW_DOES_NOT_MATCH_SHOT_IDS', self.codes(data, duration=120,
                                                                       source_catalog=catalog))

    def test_whisper_length_does_not_inflate_visual_sentence_budget(self):
        events = [event(1, 1), event(2, 1)]
        plan = dict(events_by_part=[events], scene_contract={}, situation_plan_version=4)
        normal = allocate_sentences(plan, 20, 400)
        events[0]['spoken_text'] = 'noisy transcription ' * 1000
        noisy = allocate_sentences(plan, 20, 400)
        self.assertEqual([e['sentence_quota'] for e in normal['events_by_part'][0]],
                         [e['sentence_quota'] for e in noisy['events_by_part'][0]])
