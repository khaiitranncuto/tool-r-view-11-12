import copy
import json
import unittest

from situation_recap import (bind_character_registry, character_registry,
                             build_situation_plan, registry_audit_payload)
from grounded_pipeline import lock_plan_visuals
from recap_pipeline import (build_part_story_context, build_single_part_video_prompt,
                            coerce_single_part_response, single_part_missing_requirements,
                            normalize_global_evidence_audit, apply_verified_evidence_results)
from tts_engine import split_text_into_sentences


class SituationRecapTests(unittest.TestCase):
    def test_active_global_plan_does_not_partition_the_source_into_three_parts(self):
        chunks = self.chunks()
        legacy = build_situation_plan(chunks, {}, 72, 300)
        global_plan = build_situation_plan(chunks, {}, 72, 300, single_timeline=True)
        self.assertEqual(global_plan['boundaries'], [0, 72])
        self.assertEqual(len(global_plan['events_by_part']), 1)
        self.assertEqual(global_plan['events_by_part'][0],
                         [e for group in legacy['events_by_part'] for e in group])
        locked = lock_plan_visuals(global_plan, chunks)
        self.assertEqual(locked['boundaries'], [0, 72])
        self.assertEqual(len(locked['events_by_part']), 1)

    def test_unknown_identity_can_gain_a_real_name_without_new_id(self):
        registry = self.registry()
        registry[0]['name'] = ''
        registry[0]['neutral_alias'] = 'cậu bé đeo kính'
        payload = self.character_payload()
        payload['characters'][0].update(name='Kukuru', name_evidence='Tên được gọi trong lời thoại')
        bound = bind_character_registry(payload, registry)
        person = bound['characters'][0]
        self.assertEqual(person['canonical_id'], 'P01')
        self.assertEqual(person['name'], 'Kukuru')
        saved = character_registry({'characters': registry}, [{'characters': [person]}])
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['name'], 'Kukuru')

    def test_mentioned_character_is_allowed_without_being_visible(self):
        chunks = self.chunks()
        plan = build_situation_plan(chunks, {}, 72, 300)
        plan['characters'] = [{'id': 'P01', 'name': 'Nobita'}, {'id': 'P07', 'name': 'Giga Zombie'}]
        event = plan['events_by_part'][0][0]
        event['characters'] = ['P01']
        event['spoken_text'] = 'Chúng ta cần tìm Giga Zombie.'
        context = build_part_story_context(plan, 1)
        row = context['required_events'][0]
        self.assertIn('Giga Zombie', row['mentioned_character_names'])
        self.assertIn('Giga Zombie', row['allowed_character_names'])
        self.assertNotIn('Giga Zombie', row['visible_character_names'])

    def test_lookahead_shot_belongs_to_next_core_before_situation_grouping(self):
        chunks = self.chunks()
        original = chunks[0]['events']
        chunks = [dict(start_seconds=0, end_seconds=42, primary_end_seconds=24,
                       events=copy.deepcopy(original[:3])),
                  dict(start_seconds=24, end_seconds=72, primary_end_seconds=72,
                       events=copy.deepcopy(original[2:]))]
        chunks[1]['events'][0]['situation_id'] = 'OTHER_LOCAL_ID'
        plan = lock_plan_visuals(build_situation_plan(chunks, {}, 72, 300), chunks)
        selected = [s['shot_id'] for part in plan['events_by_part'] for e in part
                    for s in e['visual_sources']]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), 12)

    def registry(self):
        return [{'id': 'P01', 'name': 'Nobita', 'confidence': 1,
                 'name_evidence': 'Được gọi tên trong lời thoại',
                 'stable_features': ['đeo kính tròn', 'tóc đen'],
                 'visual_signature': 'Kính tròn, tóc đen', 'aliases': []}]

    def character_payload(self):
        return {'characters': [{'id': 'C1', 'canonical_id': 'P01', 'name': 'Nobita',
                'match_confidence': .99, 'alias': 'cậu bé đeo kính',
                'identity_evidence': {'matched_features': ['đeo kính tròn', 'tóc đen'],
                                      'shot_ids': ['SHOT_000001']}}],
                'events': [{'event': 'Nobita chạy về nhà', 'characters': ['C1'],
                            'grounded_facts': [{'text': 'Nobita chạy về nhà',
                                                'shot_ids': ['SHOT_000001']}]}]}

    def test_registry_match_requires_features_and_observed_shot(self):
        payload = self.character_payload()
        matched = bind_character_registry(payload, self.registry())
        self.assertEqual(matched['characters'][0]['canonical_id'], 'P01')
        self.assertEqual(matched['characters'][0]['name'], 'Nobita')
        for kind in ('bad_name', 'bad_shot', 'bad_features', 'low_confidence'):
            candidate = copy.deepcopy(payload)
            c = candidate['characters'][0]
            if kind == 'bad_name':
                c['name'] = 'Giga Zombie'
                candidate['events'][0]['event'] = 'Giga Zombie chạy về nhà'
            elif kind == 'bad_shot':
                c['identity_evidence']['shot_ids'] = ['SHOT_999999']
            elif kind == 'bad_features':
                c['identity_evidence']['matched_features'] = ['áo đỏ', 'đội mũ']
            else:
                c['match_confidence'] = .5
            result = bind_character_registry(candidate, self.registry())
            self.assertEqual(result['characters'][0]['canonical_id'], '' if kind == 'bad_name' else 'P01')
            self.assertEqual(result['characters'][0]['identity_match_status'], 'pending')
            self.assertNotEqual(result['characters'][0]['identity_status'], 'confirmed')
            self.assertEqual(registry_audit_payload(self.registry(), [{'characters': result['characters']}])['identity_links'], [])
            self.assertNotIn(c['name'], result['events'][0]['event'])

    def test_unknown_ids_are_append_only_and_overview_needs_name_evidence(self):
        unknown = self.character_payload()
        unknown['characters'][0].update(canonical_id='', name='', id='C2')
        first = bind_character_registry(unknown, self.registry())['characters'][0]
        first['id'] = 'CH001_C2'
        registry = character_registry({'characters': self.registry()}, [{'characters': [first]}])
        second = bind_character_registry(unknown, registry)['characters'][0]
        self.assertEqual(first['canonical_id'], second['canonical_id'])
        uncertain = self.registry()
        uncertain[0]['name_evidence'] = ''
        self.assertEqual(character_registry({'characters': uncertain})[0]['name'], '')

    def chunks(self):
        events = []
        for i in range(6):
            start = i * 12.0
            facts = [{'fact_id': f'F{i}_{j}', 'text': 'Cậu bé săn cá rồi mang về làng',
                      'shot_ids': [f'SHOT_{i * 2 + j + 1:06d}']} for j in range(2)]
            events.append({'evidence_id': f'EV{i}', 'event': 'Cậu bé săn cá',
                           'situation_id': f'SIT{i // 2}',
                           'start_seconds': start, 'end_seconds': start + 12,
                           'grounded_facts': facts,
                           'visual_sources': [{'shot_id': f['shot_ids'][0],
                               'start_seconds': start + j * 6, 'end_seconds': start + (j + 1) * 6,
                               'fact_ids': [f['fact_id']]} for j, f in enumerate(facts)]})
        return [{'events': events}]

    def test_situation_groups_keep_proof_without_sentence_quota(self):
        chunks = self.chunks()
        plan = lock_plan_visuals(build_situation_plan(chunks, {}, 72, 5620), chunks)
        self.assertEqual(plan['scene_contract']['scene_count'], 3)
        self.assertEqual(sum(len(e['observed_facts']) for p in plan['events_by_part'] for e in p), 12)
        context = build_part_story_context(plan, 1)
        event = context['required_events'][0]
        self.assertEqual(len(event['visual_sources']), 4)
        prompt = build_single_part_video_prompt(1, 24, 0, 24, 100, story_context=context)
        self.assertIn('KHONG phai mot cau hay mot shot', prompt)
        text = ('Cậu bé mang cá về làng sau chuyến đi săn. '
                'Nhưng khi trở lại, cậu phát hiện nơi ở đã bị tàn phá, '
                'chỉ còn những đống đổ nát trước mắt.')
        # A paragraph may use a representative subset of the preselected group.
        response = json.dumps({'scenes': [{'event_id': event['event_id'], 'tts': text,
                              'visual_links': [{'visual_id': 'V001', 'trigger_text': ''}]}]})
        payload_text = coerce_single_part_response(response, 1, 24, 100,
                           locked_event_by_id={event['event_id']: event})
        scene = json.loads(payload_text)['parts'][0]['scenes'][0]
        self.assertEqual(scene['tts'], text)
        self.assertTrue(scene['situation_narration'])
        self.assertEqual(scene['visual_alignment_errors'], [])
        report = single_part_missing_requirements(payload_text, 24, 100,
                                                  require_tts_word_range=True)
        self.assertEqual(report['invalid_tts_scenes'], [])
        self.assertEqual(len(split_text_into_sentences('[TIME=00:00-00:06] ' + text)), 1)

    def test_registry_ids_survive_existing_global_identity_pipeline(self):
        local = bind_character_registry(self.character_payload(), self.registry())['characters'][0]
        local.update(id='CH001_C1', proposed_name='Nobita')
        chunks = [{'characters': [local], 'events': []}]
        audit = normalize_global_evidence_audit(
            json.dumps(registry_audit_payload(self.registry(), chunks)), chunks, 60)
        result = apply_verified_evidence_results(chunks, audit)
        self.assertEqual(result[0]['characters'][0]['canonical_id'], 'P01')
        self.assertEqual(result[0]['characters'][0]['name'], 'Nobita')

    def test_one_stable_trait_plus_compatible_clothes_keeps_real_names(self):
        cases = [
            ('P03', 'Shizuka', ['tóc tết', 'mắt to'], 'tóc tết hai bên, váy hồng', ['tóc tết', 'váy hồng']),
            ('P04', 'Gian', ['to lớn', 'tóc ngắn'], 'thân hình to lớn, áo cam', ['to lớn', 'áo cam']),
            ('P05', 'Suneo', ['tóc nhọn', 'mỏ nhọn'], 'tóc nhọn, áo tím', ['tóc nhọn', 'áo tím']),
        ]
        for cid, name, stable, signature, matched in cases:
            registry = [{'id': cid, 'name': name, 'confidence': 1, 'name_evidence': 'lời thoại',
                         'stable_features': stable, 'visual_signature': signature}]
            payload = self.character_payload()
            payload['characters'][0].update(canonical_id=cid, name=name,
                identity_evidence={'matched_features': matched, 'shot_ids': ['SHOT_000099']})
            result = bind_character_registry(payload, registry, identity_shot_ids={'SHOT_000001', 'SHOT_000099'})
            self.assertEqual(result['characters'][0]['canonical_id'], cid)
            self.assertEqual(result['characters'][0]['name'], name)
            self.assertEqual(result['characters'][0]['identity_match_status'], 'matched')

    def test_known_name_recovers_id_and_pending_view_never_creates_new_person(self):
        payload = self.character_payload()
        payload['characters'][0]['canonical_id'] = ''
        result = bind_character_registry(payload, self.registry())
        self.assertEqual(result['characters'][0]['canonical_id'], 'P01')
        payload['characters'][0].update(canonical_id='P01', match_confidence=.4)
        pending = bind_character_registry(payload, self.registry())
        self.assertEqual(pending['characters'][0]['canonical_id'], 'P01')
        self.assertEqual(pending['characters'][0]['name'], 'Nobita')
        self.assertEqual(len(character_registry({'characters': self.registry()},
                                              [{'characters': pending['characters']}])), 1)

    def test_interleaved_storylines_are_split_before_locking_not_repaired_after(self):
        chunks = self.chunks()
        a, b = copy.deepcopy(chunks[0]['events'][:2])
        # A appears first and returns after B, like Shizuka/piano -> Doraemon -> Shizuka/hamster.
        a['grounded_facts'] += copy.deepcopy(chunks[0]['events'][2]['grounded_facts'])
        a['visual_sources'] += copy.deepcopy(chunks[0]['events'][2]['visual_sources'])
        a['end_seconds'] = 36
        a['situation_id'], b['situation_id'] = 'PIANO', 'FLOOD'
        a['characters'], b['characters'] = ['P03'], ['P01', 'P02']
        source = [{'events': [a, b]}]
        plan = lock_plan_visuals(build_situation_plan(source, {}, 36, 300), source)
        events = [e for part in plan['events_by_part'] for e in part]
        self.assertEqual([e['start_seconds'] for e in events], [0, 12, 24])
        self.assertEqual([e['characters'] for e in events], [['P03'], ['P01', 'P02'], ['P03']])
        self.assertFalse(plan.get('visual_order_repaired', False))

    def test_overview_cannot_remove_observed_story_even_before_opening(self):
        chunks = self.chunks()
        outline = {'non_story_ranges': [{'start_seconds': 60, 'end_seconds': 72,
                                        'kind': 'ending_montage', 'confidence': .99}]}
        plan = build_situation_plan(chunks, outline, 72, 300)
        events = [e for part in plan['events_by_part'] for e in part]
        self.assertEqual(sum(len(e['planned_fact_ids']) for e in events), 12)
        self.assertEqual(events[-1]['end_seconds'], 72)

    def test_ambiguous_shot_does_not_join_two_storylines(self):
        chunks = self.chunks()
        a, b = chunks[0]['events'][:2]
        b['situation_id'] = 'OTHER'
        b['grounded_facts'][0]['shot_ids'] = a['grounded_facts'][0]['shot_ids'][:]
        plan = build_situation_plan(chunks, {}, 72, 300)
        self.assertEqual(plan['ambiguous_source_shots'], [{'chunk_index': 0, 'shot_id': 'SHOT_000001'}])
        self.assertTrue(all('SHOT_000001' not in f['shot_ids']
                            for part in plan['events_by_part'] for e in part for f in e['observed_facts']))

    def test_situation_dialogue_and_source_action_survive_writer_context(self):
        from transcript_pipeline import attach_transcript_to_evidence
        from narrative_contract import chapter_prompt
        chunks = self.chunks()
        chunks, _ = attach_transcript_to_evidence(chunks, [
            {'start_seconds': 1, 'end_seconds': 3, 'text': 'We need food before leaving.'},
            {'start_seconds': 50, 'end_seconds': 52, 'text': 'The next village is safe.'}])
        plan = lock_plan_visuals(build_situation_plan(chunks, {}, 72, 300), chunks)
        context = build_part_story_context(plan, 1)
        event = context['required_events'][0]
        prompt = chapter_prompt([event], context)
        self.assertIn('We need food before leaving.', prompt)
        self.assertNotIn('The next village is safe.', event['spoken_text'])
        self.assertEqual(event['visual_sources'][0]['source_start_seconds'], 0)
        self.assertIn('fact_ids', event['visual_sources'][0])
        self.assertEqual(event['visual_sources'][0]['event'], event['observed_facts'][0]['text'])


if __name__ == '__main__':
    unittest.main()
