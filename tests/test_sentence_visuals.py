import copy
import json
import unittest

from sentence_visuals import select_sentence_shots, build_shot_catalog, sentence_candidates
from recap_pipeline import (coerce_single_part_response, payload_to_timeline, normalize_story_plan_from_evidence,
                            build_part_story_context, build_single_part_video_prompt, merge_single_part_responses)
from story_planning import run_two_stage_planning
from event_storyboard import validate_render_visual_grounding
from evidence_granularity import separate_event_granularity
from recap_pipeline import event_narration_capacity


class SentenceVisualTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{'events': [
            {'evidence_id': 'EV1', 'event': 'Nobita packs', 'cause': 'Mother scolds', 'consequence': 'He leaves',
             'start_seconds': 10, 'end_seconds': 30,
             'visual_sources': [{'start_seconds': 10, 'end_seconds': 20, 'event': 'Nobita packs',
                                 'verified_visual_shot': True}]},
            {'evidence_id': 'EV2', 'event': 'Doraemon opens a door', 'start_seconds': 40, 'end_seconds': 50,
             'visual_sources': [{'start_seconds': 40, 'end_seconds': 50, 'event': 'Doraemon opens a door',
                                 'verified_visual_shot': True}]},
        ]}]
        self.payload = {'characters': [], 'parts': [{'part': 1, 'scenes': [
            {'scene_id': 'S0001', 'tts': 'Nobita packs his bag.', 'planning_source_event_ids': ['EV1'],
             'start': '00:10', 'end': '00:30', 'visual_selection_pending': True},
            {'scene_id': 'S0002', 'tts': 'Doraemon opens a door.', 'planning_source_event_ids': ['EV2'],
             'start': '00:40', 'end': '00:50', 'visual_selection_pending': True},
        ]}]}

    def choose_first(self, cards):
        return {'matches': [{'scene_id': c['scene_id'], 'supported': True,
                 'links': [{'shot_id': c['candidates'][0]['shot_id'], 'trigger_text': ''}]} for c in cards]}

    def test_selects_observed_shots_after_narration_without_reopening_video(self):
        original = copy.deepcopy(self.payload)
        calls = []
        def select(cards):
            calls.append(cards)
            return self.choose_first(cards)
        def locate(*args):
            self.fail('Existing catalog should not reopen video')
        result = select_sentence_shots(self.payload, self.chunks, select, locate=locate)
        self.assertEqual(len(calls), 1)
        self.assertEqual([s['tts'] for s in result['parts'][0]['scenes']],
                         [s['tts'] for s in original['parts'][0]['scenes']])
        self.assertTrue(validate_render_visual_grounding(payload_to_timeline(result), 100))
        self.assertEqual(self.payload, original)

    def test_wrong_event_shot_is_not_offered_even_if_character_matches(self):
        catalog = build_shot_catalog(self.chunks)
        events = {e['evidence_id']: e for e in self.chunks[0]['events']}
        candidates = sentence_candidates(self.payload['parts'][0]['scenes'][0], catalog, events)
        self.assertEqual([r['event_ids'] for r in candidates], [['EV1']])

    def test_unknown_model_shot_never_becomes_a_timestamp(self):
        def select(cards):
            return {'matches': [{'scene_id': c['scene_id'], 'supported': True,
                                'links': [{'shot_id': 'SH99999', 'trigger_text': ''}]} for c in cards]}
        with self.assertRaisesRegex(ValueError, 'SENTENCE_SHOTS_PENDING'):
            select_sentence_shots(self.payload, self.chunks, select)

    def test_one_sentence_can_change_shot_at_literal_spoken_action(self):
        self.chunks[0]['events'][0]['visual_sources'].append({
            'start_seconds': 23, 'end_seconds': 28, 'event': 'Nobita leaves', 'verified_visual_shot': True})
        self.payload['parts'][0]['scenes'] = [dict(self.payload['parts'][0]['scenes'][0],
            tts='Nobita packs his bag, then he leaves the room.')]
        def select(cards):
            candidates = cards[0]['candidates']
            return {'matches': [{'scene_id': 'S0001', 'supported': True, 'links': [
                {'shot_id': candidates[0]['shot_id'], 'trigger_text': ''},
                {'shot_id': candidates[1]['shot_id'], 'trigger_text': 'then he leaves'},
            ]}]}
        result = select_sentence_shots(self.payload, self.chunks, select)
        timeline = payload_to_timeline(result)
        self.assertIn('[BEAT=', timeline)
        self.assertTrue(validate_render_visual_grounding(timeline, 100))

    def test_same_shot_for_adjacent_sentences_uses_only_observed_frames(self):
        scene = self.payload['parts'][0]['scenes'][0]
        self.payload['parts'][0]['scenes'] = [scene, dict(scene, scene_id='S0002', tts='Nobita closes his bag.')]
        result = select_sentence_shots(self.payload, self.chunks, self.choose_first)
        scenes = result['parts'][0]['scenes']
        self.assertEqual((scenes[0]['start'], scenes[0]['end']), ('00:00:10.00', '00:00:15.00'))
        self.assertEqual((scenes[1]['start'], scenes[1]['end']), ('00:00:15.00', '00:00:20.00'))
        self.assertTrue(validate_render_visual_grounding(payload_to_timeline(result), 100))

    def test_missing_shot_keeps_narration_and_can_be_located_after_writing(self):
        self.chunks[0]['events'][1]['visual_sources'] = []
        locations = []
        def locate(queries, attempt):
            locations.append(queries)
            self.assertEqual([q['scene_id'] for q in queries], ['S0002'])
            self.assertEqual(queries[0]['tts'], 'Doraemon opens a door.')
            return [{'event_ids': ['EV2'], 'start': 42, 'end': 48, 'action': 'Doraemon opens a door'}]
        result = select_sentence_shots(self.payload, self.chunks, self.choose_first, locate=locate)
        self.assertEqual(len(locations), 1)
        self.assertEqual(result['parts'][0]['scenes'][1]['start'], '00:00:42.00')

    def test_missing_shots_do_not_replay_text_selection_when_no_new_video_evidence(self):
        self.chunks[0]['events'][1]['visual_sources'] = []
        text_calls, locations, saved = [], [], []
        def select(cards):
            text_calls.append(cards)
            return self.choose_first(cards)
        def locate(queries, attempt):
            locations.append(attempt)
            return []
        with self.assertRaisesRegex(ValueError, 'SENTENCE_SHOTS_PENDING'):
            select_sentence_shots(self.payload, self.chunks, select, locate=locate, save=saved.append)
        self.assertEqual(locations, [1, 2])
        self.assertEqual(len(text_calls), 1)
        self.assertIn('S0001', saved[-1]['choices'])
        self.assertEqual(saved[-1]['pending_scene_ids'], ['S0002'])

    def test_transport_failure_resumes_only_unfinished_batches(self):
        base = self.payload['parts'][0]['scenes'][0]
        self.payload['parts'][0]['scenes'] = [dict(base, scene_id=f'S{i:04d}') for i in range(1, 14)]
        calls, saved = [], []
        def select(cards):
            calls.append(cards)
            if len(calls) == 2:
                raise RuntimeError('read timeout')
            return self.choose_first(cards)
        with self.assertRaisesRegex(RuntimeError, 'read timeout'):
            select_sentence_shots(self.payload, self.chunks, select, save=saved.append)
        self.assertEqual(len(saved[-1]['choices']), 12)
        resumed = []
        def resume(cards):
            resumed.extend(c['scene_id'] for c in cards)
            return self.choose_first(cards)
        select_sentence_shots(self.payload, self.chunks, resume, checkpoint=saved[-1])
        self.assertEqual(resumed, ['S0013'])

    def test_pending_narration_cannot_be_exported_as_a_fake_time_range(self):
        with self.assertRaisesRegex(ValueError, 'SENTENCE_SHOTS_PENDING'):
            payload_to_timeline(self.payload)

    def test_out_of_order_model_choice_is_repaired_with_text_only(self):
        self.chunks[0]['events'][0]['visual_sources'].append({
            'start_seconds': 23, 'end_seconds': 28, 'event': 'Nobita leaves', 'verified_visual_shot': True})
        scene = self.payload['parts'][0]['scenes'][0]
        self.payload['parts'][0]['scenes'] = [scene, dict(scene, scene_id='S0002', tts='Nobita leaves.')]
        calls = []
        def select(cards):
            calls.append(cards)
            if len(calls) == 1:
                return {'matches': [
                    {'scene_id': 'S0001', 'supported': True, 'links': [
                        {'shot_id': cards[0]['candidates'][1]['shot_id'], 'trigger_text': ''}]},
                    {'scene_id': 'S0002', 'supported': True, 'links': [
                        {'shot_id': cards[1]['candidates'][0]['shot_id'], 'trigger_text': ''}]},
                ]}
            self.assertEqual([c['scene_id'] for c in cards], ['S0002'])
            self.assertIsNotNone(cards[0]['source_constraints']['previous'])
            return {'matches': [{'scene_id': 'S0002', 'supported': True, 'links': [
                {'shot_id': cards[0]['candidates'][1]['shot_id'], 'trigger_text': ''}]}]}
        result = select_sentence_shots(self.payload, self.chunks, select)
        self.assertEqual(len(calls), 2)
        self.assertTrue(validate_render_visual_grounding(payload_to_timeline(result), 100))

    def test_invalid_trigger_does_not_silently_drop_second_action(self):
        self.chunks[0]['events'][0]['visual_sources'].append({
            'start_seconds': 23, 'end_seconds': 28, 'event': 'Nobita leaves', 'verified_visual_shot': True})
        self.payload['parts'][0]['scenes'] = [self.payload['parts'][0]['scenes'][0]]
        def select(cards):
            return {'matches': [{'scene_id': 'S0001', 'supported': True, 'links': [
                {'shot_id': cards[0]['candidates'][0]['shot_id'], 'trigger_text': ''},
                {'shot_id': cards[0]['candidates'][1]['shot_id'], 'trigger_text': 'not in narration'},
            ]}]}
        with self.assertRaisesRegex(ValueError, 'SENTENCE_SHOTS_PENDING'):
            select_sentence_shots(self.payload, self.chunks, select)


class DeferredNarrationPipelineTests(unittest.TestCase):
    def test_observed_fact_capacity_is_independent_of_missing_shot_duration(self):
        event = {'sentence_first': True, 'start_seconds': 10, 'end_seconds': 11,
                 'event': 'Nobita dong goi hanh ly', 'cause': 'Me yeu cau hoc bai',
                 'consequence': 'Cau roi nha di bui'}
        self.assertEqual(event_narration_capacity(event), 3)

    def test_missing_visual_does_not_demote_observed_event_but_uncertain_claim_stays_pending(self):
        event = {'evidence_id': 'EV1', 'start_seconds': 0, 'end_seconds': 180,
                 'event': 'Action', 'cause': 'Cause', 'consequence': 'Result'}
        uncertain = dict(event, evidence_id='EV2', granularity='uncertain')
        chunks, _ = separate_event_granularity([{'events': [event, uncertain]}], preserve_unlocated_events=True)
        self.assertEqual([e['evidence_id'] for e in chunks[0]['events']], ['EV1'])
        self.assertEqual(chunks[0]['event_review_candidates'][0]['evidence_id'], 'EV2')

    def test_plan_and_write_keep_events_without_shots_then_match_afterwards(self):
        chunks = [{'events': [{'evidence_id': f'EV{i}', 'start_seconds': (i - 1) * 40,
                  'end_seconds': i * 40, 'event': f'Hanh dong rieng {i}',
                  'cause': f'Nguyen nhan rieng {i}', 'consequence': f'Ket qua rieng {i}'} for i in range(1, 4)]}]
        spine = {'boundary_1_seconds': 40, 'boundary_2_seconds': 80,
                 'parts': [{'part': i, 'summary': f'Phan {i}'} for i in range(1, 4)],
                 'selected_event_clusters': [{'cluster_id': f'C{i}', 'event_ids': [f'EV{i}'],
                                               'sentence_count': 2} for i in range(1, 4)]}
        def generate(prompt, label):
            if '2/5:' in label:
                return json.dumps(spine), 'model'
            if '3/5:' in label:
                return json.dumps({'editorial_anchors': {}, 'event_importance': {}}), 'model'
            cid = label.split()[-1]
            i = int(cid[1:])
            return json.dumps({'cluster_id': cid, 'event_plans': [
                {'source_event_ids': [f'EV{i}'], 'focuses': [f'Nguyen nhan rieng {i}', f'Ket qua rieng {i}']}]}), 'model'
        normalize = lambda text: normalize_story_plan_from_evidence(text, chunks, 120, 120,
                                                                   require_editorial_contract=True)
        text, _ = run_two_stage_planning(chunks, 120, 120, 'plan', generate, json.loads,
                                         normalize, defer_visual_selection=True)
        plan = normalize(text)
        events = [e for part in plan['events_by_part'] for e in part]
        self.assertEqual(len(events), 6)
        self.assertTrue(all(e['visual_selection_pending'] and e['visual_sources'] == [] for e in events))
        responses = []
        for part in range(1, 4):
            context = build_part_story_context(plan, part)
            prompt = build_single_part_video_prompt(part, 40, (part - 1) * 40, part * 40, 40,
                                                    story_context=context)
            self.assertIn('chua chon shot', prompt)
            required = context['required_events']
            response = coerce_single_part_response(json.dumps({'scenes': [
                {'event_id': e['event_id'], 'tts': 'Loi ke tu su kien da quan sat.'} for e in required]}),
                part, 40, 40, locked_event_by_id={e['event_id']: e for e in required})
            responses.append(response)
        draft = merge_single_part_responses(responses, plan['boundaries'], 120)
        self.assertEqual([s['scene_id'] for p in draft['parts'] for s in p['scenes']],
                         [f'S{i:04d}' for i in range(1, 7)])
        self.assertEqual([s['planning_source_event_ids'] for p in draft['parts'] for s in p['scenes']],
                         [['EV1'], ['EV1'], ['EV2'], ['EV2'], ['EV3'], ['EV3']])
        for event in chunks[0]['events']:
            event['visual_sources'] = [{'start_seconds': event['start_seconds'] + 5,
                                         'end_seconds': event['start_seconds'] + 15,
                                         'event': event['event'], 'verified_visual_shot': True}]
        def select(cards):
            return {'matches': [{'scene_id': c['scene_id'], 'supported': True,
                                'links': [{'shot_id': c['candidates'][0]['shot_id'], 'trigger_text': ''}]}
                                for c in cards]}
        matched = select_sentence_shots(draft, chunks, select)
        self.assertTrue(validate_render_visual_grounding(payload_to_timeline(matched), 120))


if __name__ == '__main__':
    unittest.main()
