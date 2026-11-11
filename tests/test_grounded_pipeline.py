import copy
import json
import unittest
import os
import tempfile
import threading

from grounded_pipeline import (shot_index, bind_observations, grounded_detail,
    deduplicate_observations, supplement_observations, lock_plan_visuals, audit_narrations,
    parse_observations, observation_cache_path, save_observation_cache, load_observation_cache,
    canonical_shot_id, observation_prompt, recover_observations, apply_local_visual_fallbacks)
from recap_pipeline import (normalize_chunk_evidence, normalize_story_plan_from_evidence,
    build_part_story_context, coerce_single_part_response, merge_single_part_responses,
    clamp_payload_narrations, payload_to_timeline, single_part_missing_requirements)
from story_planning import run_two_stage_planning
from event_storyboard import validate_render_visual_grounding


class GroundedPipelineTests(unittest.TestCase):
    def test_uncertain_and_dialogue_claims_do_not_become_visual_facts(self):
        raw = {'events': [{'event': 'At the door', 'content_kind': 'story', 'observations': [
            {'text': 'Stands at the door', 'shot_ids': ['SHOT_000001'], 'evidence_kind': 'visual', 'status': 'confirmed'},
            {'text': 'Opens the door', 'shot_ids': ['SHOT_000001'], 'evidence_kind': 'visual', 'status': 'uncertain'},
            {'text': 'Says he will leave', 'shot_ids': ['SHOT_000001'], 'evidence_kind': 'dialogue', 'status': 'confirmed'}]}]}
        bound = bind_observations(raw, shot_index(6), 0, 6, 1)
        facts = bound['events'][0]['grounded_facts']
        self.assertEqual([f['text'] for f in facts], ['Stands at the door'])
        self.assertEqual(facts[0]['status'], 'confirmed')
        self.assertEqual(len(bound['review_candidates']), 1)
        self.assertEqual(len(bound['dialogue_context']), 1)
        self.assertNotIn('Opens the door', bound['events'][0]['event'])

    def test_shared_shot_never_merges_facts_from_other_situations(self):
        chunks = self.observations(6)
        a, b = chunks[0]['events'][:2]
        original = b['grounded_facts'][0]['text']
        b['grounded_facts'][0]['shot_ids'] = a['grounded_facts'][0]['shot_ids'][:]
        merged, _ = deduplicate_observations(chunks)
        self.assertEqual(merged[0]['events'][1]['grounded_facts'][0]['text'], original)
        self.assertNotIn(original, merged[0]['events'][0]['grounded_facts'][0]['text'])

    def test_parallel_video_checks_retry_only_missing_and_keep_all_checkpoints(self):
        scenes = []
        for i in range(7):
            scene = self.fallback_scene(f'S{i + 1:04d}')
            scene['observed_facts'] = [{'text': scene['tts']}]
            scenes.append(scene)
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        calls, cache = [], {}
        def audit(cards):
            ids = [c['scene_id'] for c in cards]
            with lock:
                calls.append(ids)
                first_pass = len(calls) <= 2
            if first_pass:
                barrier.wait(timeout=5)
            return {'checks': [{'scene_id': sid, 'faithful': True}
                               for sid in ids if not (first_pass and sid == 'S0002')]}
        def save(state):
            cache.clear()
            cache.update(state)
        payload = {'parts': [{'scenes': scenes}]}
        result = audit_narrations(payload, audit, save=save, visual_evidence=True,
                                  source_identity='film', max_attempts=2, max_workers=2)
        self.assertEqual(sorted(calls[:2]), [['S0001', 'S0002', 'S0003', 'S0004', 'S0005', 'S0006'], ['S0007']])
        self.assertEqual(calls[2:], [['S0002']])
        self.assertEqual(len(cache), 7)
        self.assertTrue(all(s['grounding_audit_passed'] for s in result['parts'][0]['scenes']))
        result['parts'][0]['scenes'][0]['tts'] = 'Câu đã thay đổi sau kiểm chứng.'
        final = audit_narrations(result, lambda cards: self.fail('Final check must not call API'),
                                 checkpoint=cache, visual_evidence=True,
                                 source_identity='film', max_attempts=0)
        self.assertFalse(final['parts'][0]['scenes'][0]['grounding_audit_passed'])
        self.assertTrue(final['parts'][0]['scenes'][0]['grounding_audit_pending'])
        self.assertTrue(all(s['grounding_audit_passed'] for s in final['parts'][0]['scenes'][1:]))

    def test_normalized_three_part_merge_reuses_video_verdicts_without_network(self):
        plan, _ = self.plan()
        cache, responses = {}, []
        def save(state):
            cache.clear()
            cache.update(state)
        for part in range(1, 4):
            events = build_part_story_context(plan, part)['required_events']
            raw = json.dumps({'scenes': [{'event_id': e['event_id'],
                'tts': ' '.join(f['text'] for f in e['observed_facts']),
                'visual_links': [{'visual_id': 'V001', 'trigger_text': ''}]} for e in events]})
            payload = clamp_payload_narrations(json.loads(coerce_single_part_response(
                raw, part, plan['boundaries'][part] - plan['boundaries'][part - 1], 1040,
                locked_event_by_id={e['event_id']: e for e in events})))
            checked = audit_narrations(payload, lambda cards: {'checks': [
                {'scene_id': c['scene_id'], 'faithful': True} for c in cards]},
                checkpoint=cache, save=save, visual_evidence=True, source_identity='film',
                visual_source_offset=plan['boundaries'][part - 1], max_attempts=2)
            responses.append(json.dumps(checked))
        merged = clamp_payload_narrations(merge_single_part_responses(responses, plan['boundaries'], 3120))
        final = audit_narrations(merged, lambda cards: self.fail('Unchanged merge must not need video'),
            checkpoint=cache, visual_evidence=True, source_identity='film', max_attempts=0)
        scenes = [s for p in final['parts'] for s in p['scenes']]
        self.assertEqual(len(scenes), 156)
        self.assertTrue(all(s['grounding_audit_passed'] for s in scenes))

    def test_lightweight_planning_uses_one_optional_call_and_keeps_all_facts(self):
        chunks = self.observations(18)
        calls, saved = [], []
        def generate(prompt, label):
            calls.append((prompt, label))
            self.assertNotIn('shot_ids', prompt)
            self.assertNotIn('visual_sources', prompt)
            return json.dumps({'event_importance': {}, 'parts': [
                {'part': i, 'summary': f'Part {i}'} for i in range(1, 4)]}), 'brief-model'
        def normalize(text):
            return normalize_story_plan_from_evidence(text, chunks, 108, 360, require_editorial_contract=True)
        text, _ = run_two_stage_planning(chunks, 108, 360, 'test', generate, json.loads, normalize,
            lightweight=True, defer_visual_selection=True, save=saved.append)
        plan = lock_plan_visuals(normalize(text), chunks)
        slots = [s for part in plan['events_by_part'] for s in part]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(slots), 18)
        self.assertTrue(all(s['observed_facts'] and s['visual_sources'] for s in slots))
        def no_more_calls(prompt, label):
            self.fail('Completed planning checkpoint must not call Gemini again')
        run_two_stage_planning(chunks, 108, 360, 'test', no_more_calls, json.loads, normalize,
            lightweight=True, defer_visual_selection=True, checkpoint=saved[-1])

    def test_optional_editorial_failure_does_not_block_grounded_plan(self):
        chunks = self.observations(18)
        for response in ('transport', '[]', '{"event_importance":{"UNKNOWN":"critical"}}'):
            calls = []
            def generate(prompt, label):
                calls.append(label)
                if response == 'transport':
                    raise RuntimeError('503 unavailable')
                return response, 'model'
            def normalize(text):
                return normalize_story_plan_from_evidence(text, chunks, 108, 360, require_editorial_contract=True)
            text, _ = run_two_stage_planning(chunks, 108, 360, 'test', generate, json.loads, normalize,
                lightweight=True, defer_visual_selection=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(sum(len(p) for p in lock_plan_visuals(normalize(text), chunks)['events_by_part']), 18)

    def test_video_audit_cache_includes_shots_and_reuses_absolute_coordinates(self):
        scene = self.fallback_scene()
        scene['observed_facts'] = [{'text': scene['tts']}]
        scene['locked_visual_shots'] = [dict(scene['visual_sources'][0], start='00:06', end='00:12')]
        payload = {'parts': [{'scenes': [scene]}]}
        cache, calls = {}, []
        def save(state):
            cache.clear()
            cache.update(state)
        def audit(cards):
            calls.append(cards)
            if len(calls) > 1:
                self.assertTrue(cards[0]['locked_visual_shots'])
            return {'checks': [{'scene_id': 'S0001', 'faithful': True}]}
        audit_narrations(payload, audit, save=save)  # Old text-only verdict cannot authorize video.
        audit_narrations(payload, audit, cache, save, visual_evidence=True,
                         source_identity='source', visual_source_offset=30)
        self.assertEqual(len(calls), 2)
        absolute = copy.deepcopy(payload)
        shot = absolute['parts'][0]['scenes'][0]['locked_visual_shots'][0]
        shot.update(start_seconds=36, end_seconds=42, start='00:36', end='00:42')
        audit_narrations(absolute, audit, cache, save, visual_evidence=True, source_identity='source')
        self.assertEqual(len(calls), 2)
        shot.update(start_seconds=42, end_seconds=48, start='00:42', end='00:48')
        audit_narrations(absolute, audit, cache, save, visual_evidence=True, source_identity='source')
        self.assertEqual(len(calls), 3)

    def fallback_scene(self, sid='S0001'):
        return {'scene_id': sid, 'tts': 'Nobita cam dien thoai',
            'grounded_plan_locked': True, 'grounding_audit_passed': False,
            'visual_alignment_errors': ['unsupported_narration:unconfirmed_action'],
            'start': '00:00:06.00', 'end': '00:00:18.00',
            'visual_sources': [
                {'shot_id': 'SHOT_000002', 'start_seconds': 6, 'end_seconds': 12,
                 'description': 'Doraemon mo cua'},
                {'shot_id': 'SHOT_000003', 'start_seconds': 12, 'end_seconds': 18,
                 'description': 'Nobita cam dien thoai'}]}

    def test_local_visual_fallback_selects_similar_observed_shot_and_exports(self):
        scene, reports = self.fallback_scene(), []
        payload = {'parts': [{'part': 1, 'scenes': [scene]}]}
        result = apply_local_visual_fallbacks(payload, reports.append)
        selected = result['parts'][0]['scenes'][0]
        self.assertEqual(selected['start'], '00:00:12.00')
        self.assertEqual(reports[0]['shot_ids'], ['SHOT_000003'])
        self.assertEqual(reports[0]['tts'], scene['tts'])
        self.assertEqual(reports[0]['status'], 'approximate_unverified')
        self.assertFalse(selected['grounding_audit_passed'])
        self.assertEqual(selected['locked_visual_shots'][0]['visual_match_tier'], 'approximate_unverified')
        self.assertFalse(selected['visual_alignment_errors'])
        self.assertEqual(scene['visual_alignment_errors'], ['unsupported_narration:unconfirmed_action'])
        timeline = payload_to_timeline(result)
        self.assertTrue(validate_render_visual_grounding(timeline, 30))
        self.assertIn(scene['tts'], timeline)

    def test_local_fallback_does_not_use_neighbor_owned_or_invalid_sources(self):
        scene = self.fallback_scene()
        neighbor = {'scene_id': 'S0002', 'tts': 'Next', 'start': '00:12', 'end': '00:18',
                    'grounded_plan_locked': True, 'grounding_audit_passed': True,
                    'locked_visual_shots': [{'start': '00:12', 'end': '00:18'}]}
        result = apply_local_visual_fallbacks({'parts': [{'scenes': [scene, neighbor]}]})
        chosen = result['parts'][0]['scenes'][0]
        self.assertEqual(chosen['visual_fallback']['shot_ids'], ['SHOT_000002'])
        self.assertEqual(result['parts'][0]['scenes'][1], neighbor)
        scene['visual_sources'] = [{'shot_id': 'SHOT_000002', 'start_seconds': 6,
                                   'end_seconds': 100, 'description': scene['tts']}]
        result = apply_local_visual_fallbacks({'parts': [{'scenes': [scene, neighbor]}]})
        self.assertNotIn('visual_fallback', result['parts'][0]['scenes'][0])
        self.assertTrue(result['parts'][0]['scenes'][0]['visual_alignment_errors'])

    def test_pending_audit_can_continue_with_explicit_unverified_fallback(self):
        scene = self.fallback_scene()
        scene['observed_facts'] = [{'text': 'Nobita cam dien thoai'}]
        pending = audit_narrations({'parts': [{'scenes': [scene]}]}, lambda cards: {'checks': []})
        result = apply_local_visual_fallbacks(pending)
        selected = result['parts'][0]['scenes'][0]
        self.assertTrue(selected['grounding_audit_pending'])
        self.assertFalse(selected['grounding_audit_passed'])
        self.assertEqual(selected['visual_fallback']['status'], 'approximate_unverified')
        self.assertTrue(validate_render_visual_grounding(payload_to_timeline(result), 30))

    def test_fallback_survives_three_part_merge_and_final_audit(self):
        responses = []
        for i in range(3):
            scene = self.fallback_scene(f'S{i + 1:04d}')
            scene['observed_facts'] = [{'text': scene['tts']}]
            for source in scene['visual_sources']:
                source['start'] = f"00:{int(source['start_seconds']):02d}"
                source['end'] = f"00:{int(source['end_seconds']):02d}"
            payload = {'parts': [{'part': i + 1, 'scenes': [scene]}]}
            responses.append(json.dumps(apply_local_visual_fallbacks(payload)))
        merged = clamp_payload_narrations(merge_single_part_responses(responses, [0, 30, 60, 90], 60))
        checked = audit_narrations(merged, lambda cards: {'checks': []})
        reports = []
        final = apply_local_visual_fallbacks(checked, reports.append)
        self.assertEqual(len(reports), 3)
        self.assertEqual([r['shot_ids'] for r in reports], [['SHOT_000003']] * 3)
        self.assertEqual([r['start'] for r in reports], ['00:12.00', '00:42.00', '01:12.00'])
        self.assertTrue(validate_render_visual_grounding(payload_to_timeline(final), 90))

    def test_clip_prompt_uses_global_shot_range(self):
        prompt = observation_prompt(4, 1438.8, 1948.4, 20)
        self.assertIn('SHOT_000240 den SHOT_000325', prompt)
        self.assertIn('KHONG danh so lai tu 1', prompt)
        self.assertNotIn('SHOT_000001', prompt)
        self.assertIn('SHOT_000001 den SHOT_000001', observation_prompt(1, 0, 6, 1))

    def test_uncertain_actions_remain_explicit_without_replaying_the_whole_clip(self):
        document = {'events': [{'event': 'Open door', 'content_kind': 'story',
            'observations': [{'text': 'Open door', 'shot_ids': ['SHOT_000240']}]}],
            'review_candidates': [{'text': 'Pick up phone', 'reason': 'shot label unclear'}]}
        calls = []
        def regenerate(prompt):
            calls.append(prompt)
            return json.dumps(document)
        result = recover_observations(json.dumps(document), shot_index(2000),
                                      1438.8, 1948.4, 4, regenerate)
        self.assertEqual(len(calls), 0)
        self.assertEqual(result['review_candidates'], document['review_candidates'])
        self.assertEqual(len(result['events']), 1)

    def test_outside_shots_trigger_video_recovery_without_offset_guessing(self):
        def document(ids):
            return json.dumps({'events': [{'event': 'Open door', 'content_kind': 'story',
                'observations': [{'text': 'Open door', 'shot_ids': ids}]}]})
        calls, saved = [], []
        def regenerate(prompt):
            calls.append(prompt)
            return document(['SHOT_000240'])
        bound = recover_observations(document(['SHOT_000001', 'SHOT_000240']),
            shot_index(2000), 1438.8, 1948.4, 4, regenerate,
            save=lambda text, attempt: saved.append((text, attempt)))
        self.assertEqual(len(calls), 1)
        self.assertIn('shot_outside_attached_clip', calls[0])
        self.assertEqual(bound['events'][0]['grounded_facts'][0]['shot_ids'], ['SHOT_000240'])
        self.assertEqual(len(saved), 2)

    def test_observation_recovery_is_bounded_and_handles_bad_json(self):
        calls = []
        def regenerate(prompt):
            calls.append(prompt)
            return 'bad JSON'
        with self.assertRaisesRegex(ValueError, 'GROUNDED_OBSERVATIONS_INVALID: doan 4'):
            recover_observations('bad JSON', shot_index(2000), 1438.8, 1948.4, 4, regenerate)
        self.assertEqual(len(calls), 2)

    def test_valid_observations_do_not_call_gemini_again(self):
        raw = json.dumps({'events': [{'event': 'Open door', 'content_kind': 'story',
            'observations': [{'text': 'Open door', 'shot_ids': ['SHOT_000240']}]}]})
        def regenerate(prompt):
            self.fail('Valid evidence must not be regenerated')
        self.assertTrue(recover_observations(raw, shot_index(2000), 1438.8, 1948.4,
                                             4, regenerate)['events'])

    def test_shot_padding_normalization_preserves_the_exact_numeric_identity(self):
        catalog = shot_index(1200)
        for value in ('SHOT_0000162', 'SHOT_000162', 'SHOT_162', ' shot_00162 '):
            self.assertEqual(canonical_shot_id(value, catalog), 'SHOT_000162')
        for value in ('SHOT_000000', 'SHOT_999999', 'SHOT_00016O', 'SHOT_-162',
                      'SHOT_162.0', 'SHOT_1e2', 'CLIP_162', 'prefix SHOT_162', 162, None):
            self.assertIsNone(canonical_shot_id(value, catalog))

    def test_invalid_group_ownership_is_repaired_without_regenerating_the_whole_plan(self):
        from story_planning import repair_spine_ownership, validate_spine, balance_spine_budget
        chunks = self.observations(18)
        events = chunks[0]['events']
        for event in events:
            event['sentence_first'] = True
        first = events[0]['evidence_id']
        bad = {'selected_event_clusters': [
            {'cluster_id': 'KEEP', 'event_ids': [first], 'sentence_count': 3},
            {'cluster_id': 'C0014', 'event_ids': [first, 'EV999_9999'], 'sentence_count': 3}]}
        repaired, diagnostics = repair_spine_ownership(bad, events, 108)
        repaired.update(sentence_budget_mode='exact_target', grounding_required=True)
        balanced = balance_spine_budget(repaired, events, 18)
        validate_spine(balanced, events, 18)
        self.assertEqual(balanced['selected_event_clusters'][0]['cluster_id'], 'KEEP')
        self.assertEqual([eid for c in balanced['selected_event_clusters'] for eid in c['event_ids']],
                         [e['evidence_id'] for e in events])
        self.assertEqual({d['reason'] for d in diagnostics}, {'duplicate', 'unknown', 'restored_missing_event'})

    def test_malformed_group_json_and_optional_roles_continue_with_grounded_backend_plan(self):
        chunks = self.observations(18)
        for event in chunks[0]['events']:
            event['sentence_first'] = True
        saved, calls = [], []
        def generate(prompt, label):
            calls.append(label)
            if '2/5:' in label:
                return 'bad JSON', 'offline'
            return json.dumps({'editorial_anchors': 'bad'}), 'offline'
        def normalize(text):
            return normalize_story_plan_from_evidence(text, chunks, 108, 360, require_editorial_contract=True)
        text, _ = run_two_stage_planning(chunks, 108, 360, 'test', generate, json.loads,
            normalize, defer_visual_selection=True, save=saved.append)
        locked = lock_plan_visuals(normalize(text), chunks)
        self.assertEqual(sum(len(p) for p in locked['events_by_part']), 18)
        self.assertEqual(sum('2/5:' in c for c in calls), 1)
        self.assertEqual(saved[-1]['grouping_raw_responses'][0]['text'], 'bad JSON')
        self.assertTrue(all(not ids for ids in saved[-1]['backbone']['editorial_anchors'].values()))

    def test_interleaved_picture_windows_rebuild_slots_using_observed_illustrations(self):
        chunks = self.observations(6)
        events = chunks[0]['events']
        first, second, third = events[0]['grounded_facts']
        # Source order of the two narrative slots conflicts; neither fact nor
        # image may be fabricated to make the old ordering pass.
        slots = [{'scene_id': 'S0001', 'planned_fact_ids': [first['fact_id'], third['fact_id']]},
                 {'scene_id': 'S0002', 'planned_fact_ids': [second['fact_id']]}]
        plan = lock_plan_visuals({'events_by_part': [slots], 'boundaries': [0,36]}, chunks)
        self.assertTrue(plan['visual_order_repaired'])
        rows = plan['events_by_part'][0]
        self.assertEqual(len(rows), 2)
        self.assertEqual({fid for s in rows for fid in s['planned_fact_ids']},
                         {first['fact_id'], second['fact_id'], third['fact_id']})
        self.assertLessEqual(rows[0]['end_seconds'], rows[1]['start_seconds'])

    def test_padded_aliases_share_one_source_and_still_obey_clip_bounds(self):
        payload = {'events': [{'event': 'Cô mở cửa', 'content_kind': 'story', 'observations': [
            {'text': 'Cô mở cửa.', 'shot_ids': ['SHOT_0000162', 'SHOT_000162']}]}]}
        catalog = shot_index(1200)
        bound = bind_observations(payload, catalog, 960, 978, 3)
        event = bound['events'][0]
        self.assertEqual(event['grounded_facts'][0]['shot_ids'], ['SHOT_000162'])
        self.assertEqual(len(event['visual_shots']), 1)
        self.assertEqual(event['visual_shots'][0]['start_seconds'], 6)
        self.assertEqual(bound['_shot_id_normalizations'], {'SHOT_0000162': 'SHOT_000162'})
        normalized = normalize_chunk_evidence(json.dumps(bound), 960, 18, chunk_number=3)
        self.assertEqual(normalized['events'][0]['visual_sources'][0]['start_seconds'], 966)
        outside = bind_observations(payload, catalog, 0, 6, 3)
        self.assertEqual(outside['events'], [])
        self.assertEqual(outside['_grounding_rejections'][0]['reason'], 'shot_outside_attached_clip')
        self.assertEqual(payload['events'][0]['observations'][0]['shot_ids'][0], 'SHOT_0000162')

    def test_array_response_keeps_all_events_instead_of_largest_inner_object(self):
        events = [{'local_event_id': f'L{i}', 'event': f'Sự kiện {i}', 'content_kind': 'story',
                   'observations': [{'text': f'Hành động riêng {i}',
                                     'shot_ids': [f'SHOT_{i+1:06d}']}]} for i in range(14)]
        for text in (json.dumps(events), '```json\n' + json.dumps(events) + '\n```',
                     'Phản hồi:\n' + json.dumps(events)):
            parsed = parse_observations(text)
            self.assertEqual(parsed['events'], events)
            bound = bind_observations(parsed, shot_index(84), 0, 84, 3)
            normalized = normalize_chunk_evidence(json.dumps(bound), 0, 84, chunk_number=3)
            self.assertEqual(len(normalized['events']), 14)

    def test_object_response_preserves_characters_and_end_state(self):
        payload = {'characters': [{'id': 'C1', 'name': 'Nobita'}],
                   'events': [], 'end_state': {'location': 'Nhà'}}
        self.assertEqual(parse_observations(json.dumps(payload)), payload)
        event = {'event': 'Mở cửa', 'observations': []}
        self.assertEqual(parse_observations(json.dumps([event]))['events'], [event])

    def test_parser_does_not_mine_valid_looking_events_from_invalid_envelopes(self):
        for payload in ({'events': 'bad', 'nested': {'events': []}},
                        [{'event': 'Mở cửa', 'observations': []}, 'invalid'],
                        {'event': 'Mở cửa', 'observations': []}):
            with self.assertRaisesRegex(ValueError, 'OBSERVATION_SCHEMA_INVALID'):
                parse_observations(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'OBSERVATION_SCHEMA_INVALID'):
            parse_observations('not JSON')

    def test_single_item_document_wrapper_preserves_every_event_and_metadata(self):
        events = [{'local_event_id': f'L{i}', 'event': f'Sự kiện {i}', 'content_kind': 'story',
                   'observations': [{'text': f'Hành động riêng {i}',
                                     'shot_ids': [f'SHOT_{i+1:06d}']}]} for i in range(19)]
        payload = {'characters': [{'id': 'C1', 'name': 'Nobita'}], 'events': events,
                   'review_candidates': [{'reason': 'uncertain'}], 'end_state': {'location': 'Nhà'}}
        for text in (json.dumps([payload]), '```json\n' + json.dumps([payload]) + '\n```'):
            parsed = parse_observations(text)
            self.assertEqual(parsed, payload)
            bound = bind_observations(parsed, shot_index(114), 0, 114, 4)
            normalized = normalize_chunk_evidence(json.dumps(bound), 0, 114, chunk_number=4)
            self.assertEqual(len(normalized['events']), 19)
            self.assertEqual(sum(len(e['grounded_facts']) for e in normalized['events']), 19)

    def test_document_wrapper_does_not_bypass_schema_or_shot_validation(self):
        for payload in ([{'events': 'bad', 'nested': {'events': []}}],
                        [{'events': []}, {'events': []}]):
            with self.assertRaisesRegex(ValueError, 'OBSERVATION_SCHEMA_INVALID'):
                parse_observations(json.dumps(payload))
        payload = [{'events': [{'event': 'Mở cửa', 'content_kind': 'story', 'observations': [
            {'text': 'Cô mở cửa.', 'shot_ids': ['SHOT_999999']}]}]}]
        bound = bind_observations(parse_observations(json.dumps(payload)), shot_index(6), 0, 6, 1)
        self.assertEqual(bound['events'], [])
        self.assertEqual(bound['_grounding_rejections'][0]['reason'], 'unknown_shot_id')

    def test_observation_cache_requires_same_file_budget_contract_and_subtitles(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'source.mp4')
            with open(source, 'wb') as handle:
                handle.write(b'original video')
            path = observation_cache_path(directory, source, 3120, 12)
            state = {'primary_grounding_responses': {'3': '[{"event":"Mở cửa","observations":[]}]'},
                     'full_video_outline': {'beats': []}, 'full_video_outline_complete': True}
            save_observation_cache(path, state)
            self.assertEqual(load_observation_cache(path), state)
            for args in ((3000, 12, ''), (3120, 13, ''), (3120, 12, 'new subtitles')):
                self.assertEqual(load_observation_cache(observation_cache_path(directory, source, *args)), {})
            with open(source, 'ab') as handle:
                handle.write(b'changed')
            self.assertEqual(load_observation_cache(observation_cache_path(directory, source, 3120, 12)), {})

    def observations(self, count=156):
        catalog = shot_index(count * 6)
        raw = {'events': []}
        for i in range(count // 3):
            raw['events'].append({'local_event_id': f'L{i}', 'content_kind': 'story',
                'event': f'Nhóm sự kiện {i}', 'observations': [
                    {'text': [f'Người đàn ông {i} mở cửa căn nhà.',
                              f'Cô gái {i} chạy nhanh xuống bậc thềm.',
                              f'Chiếc xe {i} lao qua cầu rồi bốc cháy.'][j],
                     'shot_ids': [f'SHOT_{i*3+j+1:06d}']} for j in range(3)]})
        bound = bind_observations(raw, catalog, 0, count * 6, 1)
        return [normalize_chunk_evidence(json.dumps(bound), 0, count * 6, chunk_number=1)]

    def plan(self, target_words=3120):
        chunks = self.observations()
        for event in chunks[0]['events']:
            event['sentence_first'] = True
        spine = {'boundary_1_seconds': 312, 'boundary_2_seconds': 624,
            'parts': [{'part': p, 'summary': f'Phần {p}'} for p in range(1, 4)],
            'selected_event_clusters': [{'cluster_id': f'C{i}', 'event_ids': [e['evidence_id']],
                'sentence_count': 3} for i, e in enumerate(chunks[0]['events'])]}
        calls = []
        def generate(prompt, label):
            calls.append(label)
            if '2/5:' in label:
                return json.dumps(spine), 'model'
            if '3/5:' in label:
                return json.dumps({'editorial_anchors': {}, 'event_importance': {}}), 'model'
            self.fail('Grounded detail must not ask AI to invent sentence ideas: ' + label)
        def normalize(text):
            return normalize_story_plan_from_evidence(text, chunks, 936, target_words,
                                                      require_editorial_contract=True)
        text, _ = run_two_stage_planning(chunks, 936, target_words, 'plan', generate,
                                        json.loads, normalize, defer_visual_selection=True)
        self.assertEqual(len(calls), 2)
        return lock_plan_visuals(normalize(text), chunks), chunks

    def test_unknown_ids_nonstory_and_guessed_times_are_rejected(self):
        raw = {'events': [
            {'content_kind': 'story', 'event': 'Logo Toho', 'observations': [
                {'text': 'Logo hiện lên.', 'shot_ids': ['SHOT_000001']}]},
            {'content_kind': 'story', 'event': 'Hành động', 'observations': [
                {'text': 'Cô mở cánh cửa.', 'shot_ids': ['SHOT_999999'], 'start_seconds': 0},
                {'text': 'Cô bước vào phòng.', 'shot_ids': ['SHOT_000002'], 'start_seconds': 999}]},
        ]}
        bound = bind_observations(raw, shot_index(12), 0, 12, 1)
        self.assertEqual(len(bound['events']), 1)
        self.assertEqual(bound['events'][0]['visual_shots'][0]['start_seconds'], 6)
        self.assertEqual(len(bound['events'][0]['grounded_facts']), 1)
        self.assertEqual(len(bound['_grounding_rejections']), 2)
        outside = bind_observations(raw, shot_index(12), 0, 6, 1)
        self.assertEqual(outside['events'], [])

    def test_clip_offset_preserves_proof_and_global_source_times(self):
        bound = bind_observations({'events': [{'content_kind': 'story', 'event': 'Cô chạy',
            'observations': [{'text': 'Cô chạy ra cửa.', 'shot_ids': ['SHOT_000003']}]}]},
            shot_index(24), 10, 24, 2)
        normalized = normalize_chunk_evidence(json.dumps(bound), 10, 14, chunk_number=2)
        event = normalized['events'][0]
        self.assertEqual(event['visual_sources'][0]['start_seconds'], 12)
        self.assertEqual(event['visual_sources'][0]['shot_id'], 'SHOT_000003')
        self.assertEqual(event['grounded_facts'][0]['source_event_id'], event['evidence_id'])

    def test_exact_156_slots_are_proven_before_prose_without_detail_requests(self):
        plan, chunks = self.plan()
        slots = [s for p in plan['events_by_part'] for s in p]
        self.assertEqual(len(slots), 156)
        self.assertEqual(len({f for s in slots for f in s['planned_fact_ids']}), 156)
        self.assertTrue(all(s['grounded_plan_locked'] and s['visual_sources'] and s['observed_facts'] for s in slots))
        self.assertTrue(all(not s['visual_selection_pending'] for s in slots))
        self.assertEqual(lock_plan_visuals(plan, chunks), plan)

    def test_three_part_writer_merge_and_render_keep_all_proofs(self):
        plan, _ = self.plan()
        responses = []
        for part in range(1, 4):
            context = build_part_story_context(plan, part)
            required = context['required_events']
            text = json.dumps({'scenes': [{'event_id': e['event_id'],
                'tts': e['observed_facts'][0]['text'],
                'visual_links': [{'visual_id': 'V001', 'trigger_text': ''}]} for e in required]})
            result = coerce_single_part_response(text, part, plan['boundaries'][part] - plan['boundaries'][part-1],
                1040, locked_event_by_id={e['event_id']: e for e in required})
            audited = audit_narrations(json.loads(result), lambda cards: {'checks': [
                {'scene_id': c['scene_id'], 'faithful': True} for c in cards]})
            responses.append(json.dumps(audited))
        merged = clamp_payload_narrations(merge_single_part_responses(responses, plan['boundaries'], 3120))
        scenes = [s for p in merged['parts'] for s in p['scenes']]
        self.assertEqual(len(scenes), 156)
        self.assertTrue(all(s['grounded_plan_locked'] and s['grounding_audit_passed'] and s['locked_visual_shots'] for s in scenes))
        timeline = payload_to_timeline(merged)
        self.assertEqual(len(timeline.splitlines()), 156)
        self.assertTrue(validate_render_visual_grounding(timeline, 936))
        self.assertNotIn('Cần hình', timeline)

    def test_audit_rejects_new_action_and_only_rechecks_changed_text(self):
        payload = {'parts': [{'scenes': [{'scene_id': 'S0001', 'tts': 'Cô rút súng.',
            'grounded_plan_locked': True, 'observed_facts': [{'text': 'Cô mở cửa.'}]}]}]}
        cache, calls = {}, []
        def save(state):
            cache.clear()
            cache.update(state)
        def audit(cards):
            calls.append(cards)
            return {'checks': [{'scene_id': c['scene_id'], 'faithful': c['tts'] == 'Cô mở cửa.',
                               'reason': 'new_action'} for c in cards]}
        failed = audit_narrations(payload, audit, cache, save)
        self.assertFalse(failed['parts'][0]['scenes'][0]['grounding_audit_passed'])
        audit_narrations(failed, audit, cache, save)
        self.assertEqual(len(calls), 1)
        failed['parts'][0]['scenes'][0]['tts'] = 'Cô mở cửa.'
        fixed = audit_narrations(failed, audit, cache, save)
        self.assertTrue(fixed['parts'][0]['scenes'][0]['grounding_audit_passed'])
        self.assertEqual(fixed['parts'][0]['scenes'][0]['visual_alignment_errors'], [])
        self.assertEqual(len(calls), 2)

    def test_compressed_plan_keeps_outcomes_and_multiple_whole_images_per_sentence(self):
        plan, chunks = self.plan(2080)
        responses = []
        for part in range(1, 4):
            required = build_part_story_context(plan, part)['required_events']
            raw = {'scenes': [{'event_id': e['event_id'],
                'tts': ' '.join(f['text'] for f in e['observed_facts']),
                'visual_links': [{'visual_id': f'V{i+1:03d}',
                    'trigger_text': ' '.join(f['text'].split()[:2]) if i else ''}
                    for i, f in enumerate(e['observed_facts'])]} for e in required]}
            responses.append(coerce_single_part_response(json.dumps(raw), part,
                plan['boundaries'][part] - plan['boundaries'][part-1], 700,
                locked_event_by_id={e['event_id']: e for e in required}))
        merged = clamp_payload_narrations(merge_single_part_responses(responses, plan['boundaries'], 2080))
        scenes = [s for p in merged['parts'] for s in p['scenes']]
        self.assertEqual(len(scenes), 104)
        self.assertEqual(sum(len(s['observed_facts']) for s in scenes), 156)
        self.assertEqual(sum(len(s['locked_visual_shots']) for s in scenes), 156)
        self.assertTrue(all(not s['visual_alignment_errors'] for s in scenes))
        for scene in scenes:
            for shot in scene['locked_visual_shots']:
                self.assertEqual(shot['end_seconds'] - shot['start_seconds'], 6)
        self.assertTrue(validate_render_visual_grounding(payload_to_timeline(merged), 936))

    def test_missing_or_duplicate_audit_verdict_cannot_pass(self):
        payload = {'parts': [{'scenes': [{'scene_id': 'S0001', 'tts': 'Cô mở cửa.',
            'grounded_plan_locked': True, 'observed_facts': []}]}]}
        for checks in ([], [{'scene_id': 'S0001', 'faithful': True}] * 2):
            result = audit_narrations(payload, lambda cards: {'checks': checks})
            self.assertFalse(result['parts'][0]['scenes'][0]['grounding_audit_passed'])

    def test_incomplete_audit_retries_only_missing_scenes(self):
        payload = {'parts': [{'scenes': [
            {'scene_id': sid, 'tts': sid, 'grounded_plan_locked': True,
             'observed_facts': []} for sid in ('S0001', 'S0002', 'S0003')]}]}
        calls, cache = [], {}
        def audit(cards):
            calls.append([c['scene_id'] for c in cards])
            if len(calls) == 1:
                return {'checks': [{'scene_id': 'S0001', 'faithful': True},
                                   {'scene_id': 'S0002', 'faithful': False, 'reason': 'new_action'}]}
            return {'checks': [{'scene_id': c['scene_id'], 'faithful': True} for c in cards]}
        def save(state):
            cache.clear()
            cache.update(state)
        result = audit_narrations(payload, audit, save=save)
        self.assertEqual(calls, [['S0001', 'S0002', 'S0003'], ['S0003']])
        scenes = result['parts'][0]['scenes']
        self.assertFalse(any(s['grounding_audit_pending'] for s in scenes))
        self.assertEqual(scenes[1]['visual_alignment_errors'], ['unsupported_narration:new_action'])
        self.assertEqual(len(cache), 3)

    def test_missing_verdicts_are_not_cached_and_old_missing_cache_is_rechecked(self):
        payload = {'parts': [{'scenes': [{'scene_id': 'S0001', 'tts': 'Open door',
            'grounded_plan_locked': True, 'observed_facts': []}]}]}
        cache, calls = {}, []
        def save(state):
            cache.clear()
            cache.update(state)
        def missing(cards):
            calls.append(cards)
            return {'checks': []}
        result = audit_narrations(payload, missing, save=save)
        self.assertEqual(len(calls), 3)
        self.assertEqual(cache, {})
        scene = result['parts'][0]['scenes'][0]
        self.assertTrue(scene['grounding_audit_pending'])
        self.assertFalse(scene['grounding_audit_passed'])
        self.assertEqual(scene['visual_alignment_errors'], ['narration_audit_pending:missing_evidence_verdict'])
        import hashlib
        signature = hashlib.sha256(json.dumps(['Open door', []], sort_keys=True,
                                               ensure_ascii=False).encode()).hexdigest()
        cache[signature] = {'faithful': False, 'reason': 'missing_evidence_verdict'}
        fixed = audit_narrations(result, lambda cards: {'checks': [
            {'scene_id': 'S0001', 'faithful': True}]}, checkpoint=cache, save=save)
        self.assertTrue(fixed['parts'][0]['scenes'][0]['grounding_audit_passed'])
        self.assertEqual(fixed['parts'][0]['scenes'][0]['visual_alignment_errors'], [])

    def test_invalid_audit_json_and_non_boolean_verdict_are_retried(self):
        payload = {'parts': [{'scenes': [{'scene_id': 'S0001', 'tts': 'Open door',
            'grounded_plan_locked': True, 'observed_facts': []}]}]}
        calls = []
        def audit(cards):
            calls.append(cards)
            if len(calls) == 1:
                raise ValueError('invalid JSON')
            return {'checks': [{'scene_id': 'S0001', 'faithful': 'true' if len(calls) == 2 else True}]}
        result = audit_narrations(payload, audit)
        self.assertEqual(len(calls), 3)
        self.assertTrue(result['parts'][0]['scenes'][0]['grounding_audit_passed'])

    def test_shortage_is_found_before_writing_and_no_fact_is_invented(self):
        events = self.observations(6)[0]['events']
        cluster = {'cluster_id': 'C1', 'event_ids': [e['evidence_id'] for e in events], 'sentence_count': 7}
        with self.assertRaisesRegex(ValueError, 'PLAN_GROUNDED_FACTS_SHORTAGE'):
            grounded_detail(cluster, events)

    def test_shared_shot_facts_are_merged_without_fabricating_cut_positions(self):
        chunks = self.observations(6)
        event = chunks[0]['events'][0]
        event['grounded_facts'][1]['shot_ids'] = event['grounded_facts'][0]['shot_ids'][:]
        merged, removed = deduplicate_observations(chunks)
        self.assertEqual(removed, 1)
        self.assertEqual(len(merged[0]['events'][0]['grounded_facts']), 2)
        self.assertIn('Cô gái', merged[0]['events'][0]['grounded_facts'][0]['text'])
        self.assertEqual(deduplicate_observations(merged)[0], merged)
        # Re-applying a cached constituent must not duplicate it after resume.
        merged[0]['events'].append(copy.deepcopy(chunks[0]['events'][0]))
        resumed, _ = deduplicate_observations(merged)
        self.assertEqual(sum(len(e['grounded_facts']) for e in resumed[0]['events']), 5)

    def test_single_event_can_have_several_sentences_with_whole_source_windows(self):
        chunks = self.observations(6)
        event = chunks[0]['events'][0]
        detail = grounded_detail({'cluster_id': 'C1', 'event_ids': [event['evidence_id']], 'sentence_count': 3}, [event])
        slots = [{'scene_id': f'S{i+1:04d}', 'planned_fact_ids': row['fact_ids']} for i, row in enumerate(detail['sentences'])]
        plan = lock_plan_visuals({'events_by_part': [slots], 'boundaries': [0, 36]}, chunks)
        self.assertEqual([(s['start_seconds'], s['end_seconds']) for s in plan['events_by_part'][0]],
                         [(0, 6), (6, 12), (12, 18)])

    def test_multi_window_fact_keeps_all_observed_source_windows(self):
        chunks = self.observations(6)
        event = chunks[0]['events'][0]
        fact = event['grounded_facts'][0]
        fact['shot_ids'] = ['SHOT_000001', 'SHOT_000002']
        plan = lock_plan_visuals({'events_by_part': [[{'scene_id': 'S0001',
            'planned_fact_ids': [fact['fact_id']]}]], 'boundaries': [0, 36]}, chunks)
        source = plan['events_by_part'][0][0]['visual_sources'][0]
        self.assertEqual((source['start_seconds'], source['end_seconds']), (0, 12))
        self.assertEqual(source['shot_ids'], fact['shot_ids'])

    def test_audited_unsupported_prose_reaches_existing_targeted_repair_gate(self):
        plan, _ = self.plan()
        event = build_part_story_context(plan, 1)['required_events'][0]
        raw = json.dumps({'scenes': [{'event_id': event['event_id'], 'tts': 'Cô rút súng bắn người đàn ông.',
                                      'visual_links': [{'visual_id': 'V001', 'trigger_text': ''}]}]})
        payload = json.loads(coerce_single_part_response(raw, 1, 312, 1040,
                               locked_event_by_id={event['event_id']: event}))
        payload = audit_narrations(payload, lambda cards: {'checks': [
            {'scene_id': c['scene_id'], 'faithful': False, 'reason': 'invented gun'} for c in cards]})
        requirements = single_part_missing_requirements(json.dumps(payload), 312, 1040)
        self.assertTrue(any('unsupported_narration' in str(row) for row in requirements['invalid_timestamp_scenes']))

    def test_empty_supplements_do_not_rescan_same_gap_and_resume_is_cached(self):
        chunks = [{'events': [], 'start_seconds': 0, 'end_seconds': 600}]
        calls, saved = [], []
        def extract(start, end, deficit, attempt):
            calls.append((start, end))
            return []
        result, count = supplement_observations(chunks, 600, 20, extract, save=saved.append)
        self.assertEqual(count, 2)
        self.assertNotEqual(calls[0], calls[1])
        supplement_observations(chunks, 600, 20, lambda *args: self.fail('Repeated completed request'), checkpoint=saved[-1])


if __name__ == '__main__':
    unittest.main()
