import copy
import json
import unittest
from unittest.mock import Mock

from evidence_writer import (build_sentence_plan, bind_planned_scenes, writing_prompt,
                             writing_schema, write_batch, paragraph_card)
from narrative_contract import allocate_sentences, sentence_texts, sentences
from recap_pipeline import (coerce_single_part_response, clamp_payload_narrations,
                            merge_single_part_responses, apply_targeted_narration_repairs)
from source_grounded_writer import is_source_grounded


def event(index=1, quota=3):
    start = (index - 1) * 12
    return {'event_id': f'E{index:04d}', 'scene_id': f'S{index:04d}', 'sentence_quota': quota,
        'grounded_plan_locked': True, 'situation_narration': True,
        'source_link_contract_version': 1,
        'observed_facts': [{'fact_id': f'F{index}',
            'text': 'Nobita mở cửa, bước vào phòng rồi đặt cặp lên bàn.',
            'shot_ids': [f'SHOT_{index:06d}']}],
        'visual_sources': [{'shot_id': f'SHOT_{index:06d}', 'fact_ids': [f'F{index}'],
            'start_seconds': start, 'end_seconds': start + 6,
            'start': f'00:{start:02d}', 'end': f'00:{start + 6:02d}'}]}


def planned(events):
    plan = {'events_by_part': [events, [], []], 'scene_contract': {},
            'sentence_contract': {'target_count': sum(e['sentence_quota'] for e in events)}}
    return build_sentence_plan(plan)['events_by_part'][0]


def response(events):
    text = 'Nobita mở cánh cửa. Sau đó, cậu bước vào phòng. Cuối cùng, cậu đặt cặp lên bàn.'
    return {'scenes': [{'event_id': e['event_id'], 'paragraph': text, 'sentence_evidence': [
        {'fact_ids': [f'F{int(e["event_id"][1:])}'], 'evidence_kind': 'visual'}
        for _ in range(3)]} for e in events]}


class ParagraphNarrationTests(unittest.TestCase):
    def order_case(self):
        e = event(quota=4)
        e['observed_facts'] = [dict(fact_id=f'F{i}', text=text, shot_ids=[f'SHOT_{i:06d}'],
            evidence_kind='visual', spoken_text='transcript context') for i, text in enumerate(
                ['Doraemon bay trên mây, cầm gậy phát sáng.', 'Doraemon bắn tia năng lượng.', 'Nhóm bạn quan sát.'], 1)]
        e['visual_sources'] = [dict(shot_id=f'SHOT_{i:06d}', fact_ids=[f'F{i}'],
            start_seconds=(i - 1) * 6, end_seconds=i * 6) for i in range(1, 4)]
        raw = {'scenes': [dict(event_id='E0001', paragraph=
            'Doraemon bay trên mây. Cậu bắn tia năng lượng. Nhóm bạn quan sát. Doraemon tiếp tục bay.',
            sentence_evidence=[dict(fact_ids=[f'F{i}'], evidence_kind='visual') for i in [1, 2, 3, 1]])]}
        return planned([e]), raw

    def test_order_error_identifies_sentence_fact_and_source_windows(self):
        events, raw = self.order_case()
        with self.assertRaisesRegex(ValueError, r'ACTION_ORDER_INVALID:E0001:sentence=4;.*F1.*previous_source_start=12'):
            bind_planned_scenes(raw, events)
        card = paragraph_card(events[0])
        self.assertEqual([f['source_order'] for f in card['facts']], [1, 2, 3])
        self.assertEqual([f['evidence_kind'] for f in card['facts']], ['visual'] * 3)
        resumed = copy.deepcopy(events[0])
        resumed['observed_facts'].reverse()
        for i, source in enumerate(resumed['visual_sources']):
            source['start'] = f'00:{i * 6:02d}'
            source.pop('start_seconds')
        self.assertEqual([f['fact_id'] for f in paragraph_card(resumed)['facts']], ['F1', 'F2', 'F3'])

    def test_e0048_visual_facts_cannot_be_relabelled_using_nearby_whisper(self):
        events, raw = self.order_case()
        raw['scenes'][0]['sentence_evidence'][-1]['evidence_kind'] = 'dialogue'
        with self.assertRaisesRegex(ValueError, 'EVIDENCE_KIND_MISMATCH:E0001:sentence=4'):
            bind_planned_scenes(raw, events)
        def generate(missing, prompt):
            self.assertIn('Whisper is context only', prompt)
            fixed = copy.deepcopy(raw)
            fixed['scenes'][0]['paragraph'] = (
                'Doraemon bay trên mây. Cậu cầm gậy phát sáng. Cậu bắn tia năng lượng. Nhóm bạn quan sát.')
            fixed['scenes'][0]['sentence_evidence'] = [dict(fact_ids=[f'F{i}'], evidence_kind='visual')
                for i in [1, 1, 2, 3]]
            return json.dumps(fixed)
        result = write_batch(events, {}, generate, candidates=[raw])
        self.assertEqual(len(result[0]['sentence_proofs']), 4)
        self.assertTrue(is_source_grounded(result[0]))

    def test_supported_indirect_dialogue_can_use_nearby_reaction_without_replaying_action(self):
        events, raw = self.order_case()
        e = events[0]
        e['sentence_quota'] = 3
        e['observed_facts'][1].update(evidence_kind='dialogue',
            text='Họ cần tìm nơi trú ẩn.', spoken_text='Chúng ta cần tìm nơi trú ẩn.')
        raw['scenes'][0].update(paragraph=
            'Doraemon bay trên mây. Nhóm bạn quan sát. Họ cần tìm nơi trú ẩn.', sentence_evidence=[
                dict(fact_ids=['F1'], evidence_kind='visual'),
                dict(fact_ids=['F3'], evidence_kind='visual'),
                dict(fact_ids=['F2'], evidence_kind='dialogue')])
        bound = bind_planned_scenes(raw, events)[0]
        self.assertEqual([p['visual_id'] for p in bound['sentence_proofs']], ['V001', 'V003', 'V003'])
        self.assertTrue(is_source_grounded(bound))
        # A distant scene cannot be borrowed as a nearby reaction illustration.
        e['visual_sources'][2].update(start_seconds=60, end_seconds=66)
        with self.assertRaisesRegex(ValueError, 'ACTION_ORDER_INVALID:E0001:sentence=3'):
            bind_planned_scenes(raw, events)

    def test_whisper_is_context_only_and_direct_quotes_require_selective_repair(self):
        e = event(quota=1)
        e['spoken_text'] = 'Lâu được giảm động ngân thiết kỷ của ta là...'
        events = planned([e])
        card = paragraph_card(events[0])
        self.assertIn('transcript_context_only', card)
        self.assertNotIn('spoken_text', card['facts'][0])
        prompt = writing_prompt(events, {})
        self.assertIn('KHONG CHEP HOI THOAI', prompt)
        self.assertIn('CHI GIUP HIEU COT TRUYEN', prompt)
        bad = {'scenes': [{'event_id': 'E0001', 'paragraph': 'Nobita nói “Đi đâu!” khi mở cửa.',
            'sentence_evidence': [{'fact_ids': ['F1'], 'evidence_kind': 'visual'}]}]}
        original = copy.deepcopy(bad)
        def generate(missing, prompt):
            self.assertIn('RECAP_DIRECT_DIALOGUE', prompt)
            return json.dumps({'scenes': [{'event_id': 'E0001', 'paragraph': 'Nobita mở cửa rồi bước vào phòng.',
                'sentence_evidence': [{'fact_ids': ['F1'], 'evidence_kind': 'visual'}]}]})
        rows = write_batch(events, {}, generate, candidates=[bad])
        self.assertEqual(rows[0]['tts'], 'Nobita mở cửa rồi bước vào phòng.')
        self.assertEqual(bad, original)
        self.assertTrue(is_source_grounded(rows[0]))

    def test_seconds_only_resumed_sources_bind_using_same_catalog(self):
        events = planned([event()])
        for source in events[0]['visual_sources']:
            source.pop('start')
            source.pop('end')
        original = copy.deepcopy(events)
        rows = bind_planned_scenes(response(events), events)
        self.assertEqual(len(rows[0]['sentence_proofs']), 3)
        self.assertTrue(is_source_grounded(rows[0]))
        self.assertEqual(events, original)

    def test_schema_avoids_branches_and_backend_locks_event_quotas(self):
        events = planned([event(1, quota=1), event(2, quota=3)])
        schema = writing_schema(events)
        self.assertNotIn('anyOf', json.dumps(schema))
        shape = schema['properties']['scenes']['items']['properties']
        self.assertEqual(shape['event_id']['enum'], ['E0001', 'E0002'])
        proof = shape['sentence_evidence']
        self.assertEqual((proof['minItems'], proof['maxItems']), (1, 3))
        # Provider bounds are shared; event-specific counts stay mandatory locally.
        bad = response(events)
        bad['scenes'][0]['paragraph'] = 'Nobita mở cửa.'
        with self.assertRaisesRegex(ValueError, 'expected=1:actual=3'):
            bind_planned_scenes(bad, events)

    def test_one_sentence_two_evidence_kinds_repairs_only_failed_event(self):
        events = planned([event(1, quota=1), event(2, quota=3)])
        good = response(events)
        good['scenes'][0].update(paragraph='Nobita mở cửa.',
            sentence_evidence=[{'fact_ids': ['F1'], 'evidence_kind': 'visual'}])
        bad = copy.deepcopy(good)
        bad['scenes'][0]['sentence_evidence'].append({'fact_ids': ['F1'], 'evidence_kind': 'dialogue'})
        original = copy.deepcopy(bad)
        def generate(missing, prompt):
            self.assertEqual([e['event_id'] for e in missing], ['E0001'])
            self.assertIn('expected=1:actual=2', prompt)
            self.assertIn('one evidence item per sentence', prompt)
            return json.dumps({'scenes': [good['scenes'][0]]})
        rows = write_batch(events, {}, generate, candidates=[bad])
        self.assertEqual([len(r['sentence_proofs']) for r in rows], [1, 3])
        self.assertTrue(all(is_source_grounded(r) for r in rows))
        self.assertEqual(bad, original)

    def test_one_compound_fact_can_support_several_distinct_sentences(self):
        events = planned([event()])
        original = copy.deepcopy(events[0]['visual_sources'])
        rows = bind_planned_scenes(response(events), events)
        self.assertEqual(len(sentences(rows[0]['tts'])), 3)
        self.assertEqual([p['fact_ids'] for p in rows[0]['sentence_proofs']], [['F1']] * 3)
        self.assertTrue(is_source_grounded(rows[0]))
        self.assertEqual(events[0]['visual_sources'], original)
        self.assertEqual(len(rows[0]['locked_visual_shots']), 1)

    def test_exact_quota_is_allocated_to_groups_not_capped_by_fact_count_or_seconds(self):
        events = [event(i) for i in (1, 2, 3)]
        plan = {'events_by_part': [[e] for e in events], 'scene_contract': {}, 'situation_plan_version': 4}
        result = build_sentence_plan(allocate_sentences(plan, 12, 240))
        rows = [e for p in result['events_by_part'] for e in p]
        self.assertEqual(sum(e['sentence_quota'] for e in rows), 12)
        self.assertTrue(all(e['sentence_quota'] > len(e['observed_facts']) for e in rows))
        self.assertEqual([e['visual_sources'] for e in rows], [e['visual_sources'] for e in events])
        self.assertTrue(all('focus' not in s for e in rows for s in e['sentence_slots']))

    def test_all_evidence_reaches_writer_even_when_only_one_sentence_is_requested(self):
        e = event(quota=1)
        e['observed_facts'] += [{'fact_id': 'F_OTHER', 'text': 'Cửa phòng đóng lại.', 'shot_ids': ['SHOT_000001']}]
        events = planned([e])
        card = paragraph_card(events[0])
        self.assertEqual(len(card['facts']), 2)
        self.assertEqual(card['sentence_quota'], 1)
        prompt = writing_prompt(events, {'previous_part_tail': [{'tts': 'Đoạn trước.'}],
                                         'next_batch_context': [{'event': 'Chuyện tiếp theo.'}]})
        self.assertIn('MOT DOAN KE LIEN MACH', prompt)
        self.assertIn('sentence_evidence', prompt)
        self.assertIn('DA KE:', prompt)
        self.assertNotIn('E0001_C001', prompt)
        schema = writing_schema(events)
        self.assertIn('paragraph', schema['properties']['scenes']['items']['required'])

    def test_paragraph_split_preserves_punctuation_and_decimal_numbers(self):
        text = 'Cậu được 10 điểm. Nước sâu 3.5 mét! Cậu lùi lại?'
        self.assertEqual(sentence_texts(text), ['Cậu được 10 điểm.', 'Nước sâu 3.5 mét!', 'Cậu lùi lại?'])
        self.assertEqual(' '.join(sentence_texts(text)), text)

    def test_wrong_count_duplicate_prose_and_unsupported_fact_ids_are_rejected(self):
        events = planned([event()])
        for defect in ('short', 'duplicate', 'foreign', 'missing_proof'):
            raw = response(events)
            row = raw['scenes'][0]
            if defect == 'short':
                row['paragraph'] = 'Nobita mở cửa.'
            elif defect == 'duplicate':
                row['paragraph'] = 'Nobita mở cửa. Nobita mở cửa. Nobita mở cửa.'
            elif defect == 'foreign':
                row['sentence_evidence'][0]['fact_ids'] = ['FOREIGN']
            else:
                row['sentence_evidence'].pop()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                bind_planned_scenes(raw, events)

    def test_only_bad_paragraph_is_rewritten_and_valid_paragraph_survives_resume(self):
        events = planned([event(1), event(2)])
        calls, checkpoints = [], []
        def generate(requested, prompt):
            calls.append([e['event_id'] for e in requested])
            raw = response(requested)
            if len(calls) == 1:
                raw['scenes'][1]['paragraph'] = 'Nobita mở cửa.'
            return json.dumps(raw, ensure_ascii=False)
        rows = write_batch(events, {}, generate, save=lambda state: checkpoints.append(copy.deepcopy(state)))
        self.assertEqual(calls, [['E0001', 'E0002'], ['E0002']])
        self.assertEqual(rows[0]['tts'], response(events)['scenes'][0]['paragraph'])
        cached = write_batch(events, {}, Mock(side_effect=AssertionError('no repeat model')),
                             candidates=checkpoints[-1])
        self.assertEqual(cached, rows)

    def test_paragraph_proofs_survive_coercion_targeted_repair_merge_and_normalization(self):
        responses = []
        for part in (1, 2, 3):
            events = planned([event()])
            rows = bind_planned_scenes(response(events), events)
            text = coerce_single_part_response(json.dumps({'scenes': rows}), part, 12, 60,
                locked_event_by_id={'E0001': events[0]})
            normalized = clamp_payload_narrations(json.loads(text))
            self.assertEqual(normalized['parts'][0]['scenes'][0]['paragraph_plan_version'], 2)
            self.assertTrue(is_source_grounded(normalized['parts'][0]['scenes'][0]))
            responses.append(json.dumps(normalized))
        final = clamp_payload_narrations(merge_single_part_responses(responses, [0, 12, 24, 36], 180))
        self.assertEqual(sum(len(sentences(s['tts'])) for p in final['parts'] for s in p['scenes']), 9)
        self.assertTrue(all(is_source_grounded(s) for p in final['parts'] for s in p['scenes']))


if __name__ == '__main__':
    unittest.main()
