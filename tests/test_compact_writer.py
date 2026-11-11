import copy
import json
import unittest

from evidence_writer import write_batch
from flat_writer import flat_prompt, flat_to_paragraphs
import test_narration_units


class CompactWriterTests(unittest.TestCase):
    def events(self):
        first = test_narration_units.NarrationUnitTests().plan()['events_by_part'][0][0]
        second = copy.deepcopy(first)
        second.update(event_id='E0002', scene_id='S0002')
        second['sentence_slots'][0]['unit_id'] = 'E0002_U001'
        second['sentence_slots'][0]['slot_id'] = 'E0002_C001'
        return [first, second]

    def response(self, events):
        return json.dumps({'sentences': [dict(event_id=e['event_id'], unit_id=u['unit_id'],
            text='Kukuru nhảy ra khỏi tủ. Cậu lao về phía nhóm bạn.')
            for e in events for u in e['sentence_slots']]})

    def test_prompt_sends_fact_text_once_and_preserves_every_unit_assignment(self):
        events = self.events()[:1]
        event = events[0]
        event['sentence_slots'].append(dict(event['sentence_slots'][0], unit_id='E0001_U002'))
        event['sentence_quota'] = 2
        prompt = flat_prompt(events, {})
        packet = json.loads(prompt.split('\n')[-1])
        for fact in event['observed_facts']:
            self.assertEqual(prompt.count(fact['text']), 1)
        self.assertEqual([u['unit_id'] for u in packet['events'][0]['units']],
                         [u['unit_id'] for u in event['sentence_slots']])
        self.assertEqual(packet['events'][0]['units'][1]['fact_ids'], ['F1', 'F2'])
        self.assertNotIn('focus', packet['events'][0]['units'][0])
        self.assertIn('NGÔN NGỮ BẮT BUỘC', prompt)
        self.assertNotIn('???', prompt)

    def test_503_splits_between_situations_with_narration_context(self):
        events = self.events()
        calls, splits, saved = [], [], []
        def generate(group, prompt):
            calls.append(([e['event_id'] for e in group], json.loads(prompt.split('\n')[-1])))
            if len(group) > 1:
                raise RuntimeError('503')
            return self.response(group)
        rows = write_batch(events, {}, generate, prompt_builder=flat_prompt,
            response_adapter=flat_to_paragraphs, capacity_error=lambda exc: str(exc) == '503',
            on_split=lambda p, c: splits.append(c), save=lambda drafts: saved.append(copy.deepcopy(drafts)))
        self.assertEqual([c[0] for c in calls], [['E0001', 'E0002'], ['E0001'], ['E0002']])
        self.assertEqual(calls[2][1]['before'], [rows[0]['tts']])
        self.assertEqual([r['event_id'] for r in rows], ['E0001', 'E0002'])
        self.assertEqual(len(splits), 1)
        self.assertEqual(len(saved[-1][0]['scenes']), 2)

    def test_later_failure_does_not_lose_completed_child(self):
        events = self.events()
        saved = []
        def generate(group, prompt):
            if len(group) > 1 or group[0]['event_id'] == 'E0002':
                raise RuntimeError('503')
            return self.response(group)
        with self.assertRaisesRegex(RuntimeError, '503'):
            write_batch(events, {}, generate, prompt_builder=flat_prompt,
                response_adapter=flat_to_paragraphs, capacity_error=lambda exc: str(exc) == '503',
                save=lambda drafts: saved.append(copy.deepcopy(drafts)))
        calls = []
        def resume(group, prompt):
            calls.extend(e['event_id'] for e in group)
            return self.response(group)
        rows = write_batch(events, {}, resume, candidates=saved[-1], prompt_builder=flat_prompt,
                           response_adapter=flat_to_paragraphs)
        self.assertEqual(calls, ['E0002'])
        self.assertEqual(len(rows), 2)

    def test_quota_error_does_not_split(self):
        calls = []
        def generate(group, prompt):
            calls.append(1)
            raise RuntimeError('429')
        with self.assertRaisesRegex(RuntimeError, '429'):
            write_batch(self.events(), {}, generate, prompt_builder=flat_prompt,
                response_adapter=flat_to_paragraphs, capacity_error=lambda exc: str(exc) == '503')
        self.assertEqual(len(calls), 1)
