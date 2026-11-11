import copy
import json
import unittest
from unittest.mock import patch

from evidence_writer import write_batch, writing_schema
from flat_writer import flat_prompt, flat_schema, flat_to_paragraphs, writer_format
from source_grounded_writer import is_source_grounded
from test_paragraph_narration import event, planned


class FlatWriterTests(unittest.TestCase):
    def data(self):
        events = planned([event(1, 3), event(2, 1)])
        payload = {'sentences': [dict(event_id='E0001', text=text, fact_ids=['F1']) for text in [
            'Nobita mở cửa.', 'Cậu bước vào phòng.', 'Sau đó, cậu đặt cặp lên bàn.']] + [
            dict(event_id='E0002', text='Nobita mở cửa phòng.', fact_ids=['F2'])]}
        return events, payload

    def test_minimal_schema_keeps_original_schema_available(self):
        events, _ = self.data()
        schema = json.dumps(flat_schema(events))
        for key in ['enum', 'anyOf', 'minItems', 'maxItems', 'sentence_evidence', 'evidence_kind']:
            self.assertNotIn(key, schema)
        self.assertIn('sentence_evidence', json.dumps(writing_schema(events)))
        self.assertLess(len(schema), len(json.dumps(writing_schema(events))))

    def test_flat_output_binds_with_original_proofs_and_resumes_without_generation(self):
        events, payload = self.data()
        original = copy.deepcopy(payload)
        saved = []
        def generate(missing, prompt):
            self.assertIn('MOT DOAN KE LIEN MACH', prompt)
            self.assertIn('sau do tra moi cau thanh mot hang', prompt)
            return json.dumps(payload)
        rows = write_batch(events, {}, generate, prompt_builder=flat_prompt,
            response_adapter=flat_to_paragraphs, save=lambda drafts: saved.extend(drafts))
        self.assertEqual([len(r['sentence_proofs']) for r in rows], [3, 1])
        self.assertTrue(all(is_source_grounded(r) for r in rows))
        def forbidden(*_):
            self.fail('Validated cached rows should be reused')
        again = write_batch(events, {}, forbidden, candidates=saved,
            prompt_builder=flat_prompt, response_adapter=flat_to_paragraphs)
        self.assertEqual(rows, again)
        self.assertEqual(payload, original)

    def test_bad_fact_repairs_only_bad_event(self):
        events, good = self.data()
        bad = copy.deepcopy(good)
        bad['sentences'][-1]['fact_ids'] = ['F1']
        calls = []
        def generate(missing, prompt):
            calls.append([e['event_id'] for e in missing])
            return json.dumps(bad if len(calls) == 1 else {'sentences': [good['sentences'][-1]]})
        rows = write_batch(events, {}, generate, prompt_builder=flat_prompt, response_adapter=flat_to_paragraphs)
        self.assertEqual(calls, [['E0001', 'E0002'], ['E0002']])
        self.assertEqual(len(rows), 2)

    def test_double_sentence_and_interleaved_event_are_not_silently_repaired(self):
        events, payload = self.data()
        payload['sentences'][0]['text'] = 'Nobita mở cửa. Cậu đi vào.'
        mapped = flat_to_paragraphs(payload, events)
        self.assertIn('ONE_SENTENCE_REQUIRED', mapped['scenes'][0]['format_error'])
        _, payload = self.data()
        payload['sentences'].insert(1, payload['sentences'].pop())
        mapped = flat_to_paragraphs(payload, events)
        self.assertIn('INTERLEAVED_EVENT', mapped['scenes'][0]['format_error'])

    def test_unknown_ids_and_non_object_payload_rejected(self):
        events, payload = self.data()
        payload['sentences'][0]['event_id'] = 'E9999'
        for raw in [payload, [], {'sentences': [None]}]:
            with self.assertRaises(ValueError):
                flat_to_paragraphs(raw, events)

    def test_mixed_evidence_is_not_arbitrarily_relabelled(self):
        events, payload = self.data()
        events[0]['observed_facts'].append(dict(fact_id='FD', text='Cần tìm nơi trú ẩn.',
            shot_ids=['SHOT_000001'], evidence_kind='dialogue', spoken_text='Cần tìm nơi trú ẩn.'))
        payload['sentences'][0]['fact_ids'] = ['F1', 'FD']
        mapped = flat_to_paragraphs(payload, events)
        self.assertIn('MIXED_EVIDENCE', mapped['scenes'][0]['format_error'])

    def test_legacy_switch_does_not_modify_original_schema(self):
        events, _ = self.data()
        original = writing_schema(events)
        with patch.dict('os.environ', {'RECAP_WRITER_FORMAT': 'legacy'}):
            self.assertEqual(writer_format(), 'legacy')
            self.assertEqual(writing_schema(events), original)
        with patch.dict('os.environ', {'RECAP_WRITER_FORMAT': 'flat'}):
            self.assertEqual(writer_format(), 'flat')
        with patch.dict('os.environ', {'RECAP_WRITER_FORMAT': 'typo'}):
            with self.assertRaises(ValueError):
                writer_format()
