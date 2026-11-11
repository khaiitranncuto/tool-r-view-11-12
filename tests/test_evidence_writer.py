import copy
import json
import unittest
import ast
import os
from pathlib import Path
import tempfile
import threading
import uuid
import time
from unittest.mock import Mock

from evidence_writer import (sentence_slots, build_sentence_plan, bind_planned_scenes,
    write_batch, writing_prompt, writing_batches, writing_schema, audit_planned_payload, rejected_cached_events)
from narrative_contract import allocate_sentences, sentences
from source_grounded_writer import is_source_grounded
from recap_pipeline import (coerce_single_part_response, clamp_payload_narrations,
                            merge_single_part_responses, apply_targeted_narration_repairs)


def event(index=1, quota=2):
    base = (index - 1) * 18
    result = {'event_id': f'E{index:04d}', 'scene_id': f'S{index:04d}',
        'sentence_quota': quota, 'grounded_plan_locked': True,
        'situation_narration': True, 'source_link_contract_version': 1,
        'evidence_sentence_plan_version': 1,
        'observed_facts': [{'fact_id': f'F{index}_{i}', 'text': f'Nobita thực hiện hành động {i}.',
            'shot_ids': [f'SHOT_{index * 10 + i:06d}']} for i in range(3)],
        'visual_sources': [{'shot_id': f'SHOT_{index * 10 + i:06d}', 'fact_ids': [f'F{index}_{i}'],
            'start': f'00:{base + i * 6:02d}', 'end': f'00:{base + i * 6 + 6:02d}',
            'start_seconds': base + i * 6, 'end_seconds': base + i * 6 + 6} for i in range(3)]}
    result['sentence_slots'] = sentence_slots(result)
    return result


def response(events):
    return {'scenes': [{'event_id': e['event_id'], 'sentences': [
        {'slot_id': s['slot_id'], 'text': s['focus']} for s in e['sentence_slots']]} for e in events]}


def payload(events):
    rows = bind_planned_scenes(response(events), events)
    return json.loads(coerce_single_part_response(json.dumps({'scenes': rows}), 1, 60, 120,
        locked_event_by_id={e['event_id']: e for e in events}))


class EvidenceWriterTests(unittest.TestCase):
    def test_targeted_repair_keeps_other_event_and_relocks_the_same_sentence_plan(self):
        events = [event(1), event(2)]
        original = payload(events)
        repaired = response([events[1]])
        for unit in repaired['scenes'][0]['sentences']:
            unit['text'] = 'Tại đây, ' + unit['text']
        rows = bind_planned_scenes(repaired, [events[1]])
        locked = {e['event_id']: e for e in events}
        changed = apply_targeted_narration_repairs(original, json.dumps({'scenes': rows}), ['E0002'], locked)
        changed = json.loads(coerce_single_part_response(json.dumps(changed), 1, 60, 120,
            locked_event_by_id=locked))
        changed = clamp_payload_narrations(changed)
        self.assertEqual(changed['parts'][0]['scenes'][0], original['parts'][0]['scenes'][0])
        self.assertTrue(all(is_source_grounded(s) for s in changed['parts'][0]['scenes']))
        self.assertEqual(sum(len(sentences(s['tts'])) for s in changed['parts'][0]['scenes']), 4)

    def test_factual_rejection_invalidates_only_that_cached_event(self):
        events = [event(1), event(2)]
        rows = bind_planned_scenes(response(events), events)
        cache = {}
        audit_planned_payload(payload(events), lambda cards: {'checks': [
            {'scene_id': c['scene_id'], 'faithful': c['scene_id'] == 'S0001',
             'reason': '' if c['scene_id'] == 'S0001' else 'Unsupported action'} for c in cards]},
            Mock(side_effect=AssertionError('no visual uncertainty')), save_text=lambda state: cache.update(state))
        self.assertEqual(rejected_cached_events(rows, cache), {'E0002'})

    def test_actual_application_audits_planned_prose_as_text_and_reuses_cache(self):
        from google.genai import types
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'audit_grounded_payload')
        factory = ast.parse('def factory():\n    pass1_calls = 0\n').body[0]
        factory.body.extend([node, ast.Return(value=ast.Name(id=node.name, ctx=ast.Load()))])
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        data, state = payload([event()]), {}
        generate = Mock(return_value=json.dumps({'checks': [{'scene_id': 'S0001', 'faithful': True, 'reason': ''}]}))
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'source.mp4'
            video.write_bytes(b'offline identity only')
            env = dict(copy=copy, os=os, json=json, uuid=uuid, time=time, task_lock=threading.Lock(),
                task_id='offline', TASKS={'offline': state}, grounded_audit_cache={},
                video_path=str(video), writing_config=types.GenerateContentConfig(), client=object(),
                selected_model='offline-model', progress_base=42, update_prog=lambda *a: None,
                generate_gemini_text_with_recovery=generate, record_gemini_usage=lambda *a: None,
                gemini_quota_pacer=None, append_task_log=lambda *a: None,
                update_task_fields=lambda task_id, **fields: state.update(fields))
            exec(compile(module, '<actual application audit>', 'exec'), env)
            audit = env['factory']()
            checked = audit(data)
            self.assertTrue(checked['parts'][0]['scenes'][0]['grounding_audit_passed'])
            self.assertEqual(checked['parts'][0]['scenes'][0]['grounding_audit_method'], 'text')
            self.assertEqual(len(generate.call_args.args[2]), 1)
            self.assertIsInstance(generate.call_args.args[2][0], str)
            audit(checked, cache_only=True)
            self.assertEqual(generate.call_count, 1)

    def test_short_substring_and_long_character_name_do_not_require_content_rewrite(self):
        e = event(quota=3)
        texts = ['Nobita vào phòng.', 'Mẹ Nobita vào phòng.', 'Nobita vào phòng rồi ngồi xuống.']
        for fact, text in zip(e['observed_facts'], texts):
            fact['text'] = text
        e['sentence_slots'] = sentence_slots(e)
        rows = bind_planned_scenes(response([e]), [e])
        self.assertTrue(is_source_grounded(rows[0]))
        self.assertEqual(len(rows[0]['visual_links']), 3)
        self.assertEqual(len(rows[0]['visual_beats']), 2)

    def test_slots_are_distinct_ordered_propositions_with_backend_owned_proof(self):
        e = event()
        slots = e['sentence_slots']
        self.assertEqual(len(slots), 2)
        self.assertEqual([s['fact_ids'] for s in slots], [['F1_0'], ['F1_2']])
        rows = bind_planned_scenes(response([e]), [e])
        self.assertEqual(len(sentences(rows[0]['tts'])), 2)
        self.assertTrue(is_source_grounded(rows[0]))
        self.assertEqual([p['slot_id'] for p in rows[0]['sentence_proofs']], [s['slot_id'] for s in slots])

    def test_uncertain_duplicate_or_unproved_facts_cannot_fill_quota(self):
        e = event(quota=3)
        for change in ('uncertain', 'duplicate', 'missing_shot'):
            bad = copy.deepcopy(e)
            if change == 'uncertain':
                bad['observed_facts'][1]['status'] = 'uncertain'
            elif change == 'duplicate':
                bad['observed_facts'][1]['text'] = bad['observed_facts'][0]['text']
            else:
                bad['observed_facts'][1]['shot_ids'] = ['FOREIGN']
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'SHORTAGE'):
                sentence_slots(bad)

    def test_plan_preserves_exact_count_without_extra_model_call(self):
        events = [event(i) for i in (1, 2, 3)]
        plan = {'events_by_part': [[e] for e in events], 'scene_contract': {}, 'situation_plan_version': 4}
        planned = build_sentence_plan(allocate_sentences(plan, 8, 160))
        self.assertEqual(sum(len(e['sentence_slots']) for p in planned['events_by_part'] for e in p), 8)
        expanded = build_sentence_plan(allocate_sentences(plan, 10, 200))
        self.assertEqual(sum(e['sentence_quota'] for p in expanded['events_by_part'] for e in p), 10)

    def test_prompt_has_one_compact_contract_and_does_not_request_media_or_free_proof_ids(self):
        e = event()
        prompt = writing_prompt([e], {})
        self.assertIn('KE HOACH:', prompt)
        self.assertNotIn('MANIFEST:', prompt)
        self.assertNotIn('XEM VIDEO', prompt)
        self.assertEqual(prompt.count(e['sentence_slots'][0]['focus']), 1)
        self.assertEqual(writing_schema([e])['required'], ['scenes'])

    def test_wrong_slot_count_order_and_multiple_sentences_are_rejected(self):
        e = event()
        for change in ('missing', 'order', 'extra_sentence', 'invented_id'):
            bad = response([e])
            units = bad['scenes'][0]['sentences']
            if change == 'missing':
                units.pop()
            elif change == 'order':
                units.reverse()
            elif change == 'extra_sentence':
                units[0]['text'] += ' Cậu đi tiếp.'
            else:
                units[0]['slot_id'] = 'FOREIGN'
            with self.subTest(change=change), self.assertRaises(ValueError):
                bind_planned_scenes(bad, [e])

    def test_repair_requests_only_invalid_event_and_keeps_good_prose(self):
        events = [event(1), event(2)]
        calls, drafts = [], []
        def generate(requested, prompt):
            calls.append([e['event_id'] for e in requested])
            data = response(requested)
            if len(calls) == 1:
                data['scenes'][1]['sentences'].pop()
            return json.dumps(data)
        rows = write_batch(events, {}, generate, save=lambda x: drafts.append(copy.deepcopy(x)))
        self.assertEqual(calls, [['E0001', 'E0002'], ['E0002']])
        self.assertTrue(all(is_source_grounded(r) for r in rows))
        cached = write_batch(events, {}, Mock(side_effect=AssertionError('must reuse drafts')), candidates=drafts[-1])
        self.assertEqual(cached, rows)

    def test_serialization_and_merge_preserve_slots_and_exact_sentence_proofs(self):
        e = event()
        rows = bind_planned_scenes(response([e]), [e])
        raw = coerce_single_part_response(json.dumps({'scenes': rows}), 1, 18, 40,
            locked_event_by_id={e['event_id']: e})
        normalized = clamp_payload_narrations(json.loads(raw))
        self.assertTrue(is_source_grounded(normalized['parts'][0]['scenes'][0]))
        responses = [coerce_single_part_response(json.dumps({'scenes': rows}), i, 18, 40,
            locked_event_by_id={e['event_id']: e}) for i in (1, 2, 3)]
        final = clamp_payload_narrations(merge_single_part_responses(responses, [0, 18, 36, 54], 120))
        self.assertTrue(all(is_source_grounded(s) for p in final['parts'] for s in p['scenes']))
        changed = copy.deepcopy(normalized)
        changed['parts'][0]['scenes'][0]['sentence_slots'][0]['focus'] = 'Different action'
        with self.assertRaisesRegex(ValueError, 'LOCK_CHANGED'):
            clamp_payload_narrations(changed)

    def test_text_audit_only_opens_video_for_explicit_visual_uncertainty_and_reuses_verdicts(self):
        data = payload([event(1), event(2)])
        calls, text_cache, video_cache = [], {}, {}
        def check_text(cards):
            calls.append(('text', [c['scene_id'] for c in cards]))
            return {'checks': [{'scene_id': c['scene_id'], 'faithful': c['scene_id'] != 'S0002',
                'reason': 'visual_review_required: ambiguous actor' if c['scene_id'] == 'S0002' else ''} for c in cards]}
        def check_video(cards):
            calls.append(('video', [c['scene_id'] for c in cards]))
            return {'checks': [{'scene_id': c['scene_id'], 'faithful': True, 'reason': ''} for c in cards]}
        options = dict(text_cache=text_cache, video_cache=video_cache,
            save_text=lambda x: text_cache.update(x), save_video=lambda x: video_cache.update(x),
            source_identity='same-film')
        checked = audit_planned_payload(data, check_text, check_video, **options)
        self.assertEqual(calls, [('text', ['S0001', 'S0002']), ('video', ['S0002'])])
        self.assertTrue(all(s['grounding_audit_passed'] for s in checked['parts'][0]['scenes']))
        audit_planned_payload(checked, Mock(side_effect=AssertionError('no repeat text')),
            Mock(side_effect=AssertionError('no repeat video')), cache_only=True, **options)

    def test_content_rejection_and_missing_verdict_do_not_trigger_video_or_pass(self):
        data = payload([event()])
        for reply in ({'checks': []}, {'checks': [{'scene_id': 'S0001', 'faithful': False,
                                                  'reason': 'Adds an unsupported action'}]}):
            video = Mock(side_effect=AssertionError('not a visual ambiguity'))
            checked = audit_planned_payload(data, lambda _: reply, video)
            scene = checked['parts'][0]['scenes'][0]
            self.assertFalse(scene['grounding_audit_passed'])
            video.assert_not_called()


if __name__ == '__main__':
    unittest.main()
