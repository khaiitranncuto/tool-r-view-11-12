import copy
import json
import threading
import unittest
from unittest.mock import patch, Mock

from source_fact_repair import reviewed_event, coherent_repair_context, review_prompt


class SourceFactRepairTests(unittest.TestCase):
    def event(self):
        return dict(event_id='E0085', scene_id='S0085', sentence_quota=1,
            event='Bố Nobita về nhà.', narration_focus='Bố Nobita về nhà.',
            observed_facts=[dict(fact_id='F1', text='Bố Nobita về nhà.',
                raw_text='Bố Nobita về nhà.', shot_ids=['SHOT_000001'], actor_ids=['C1'])],
            sentence_slots=[dict(unit_id='E0085_U001', fact_ids=['F1'],
                shot_ids=['SHOT_000001'], visual_ids=['V001'], focus='Bố Nobita về nhà.')],
            visual_sources=[dict(visual_id='V001', shot_ids=['SHOT_000001'],
                fact_ids=['F1'], start_seconds=10, end_seconds=16, description='Bố Nobita về nhà.')])

    def test_correction_updates_focus_without_changing_ids_quota_or_ranges(self):
        event = self.event()
        original = copy.deepcopy(event)
        result = reviewed_event(event, {'facts': [dict(fact_id='F1', status='confirmed',
                                                      text='Bố Nobita rời khỏi nhà.')]})
        self.assertEqual(event, original)
        self.assertEqual(result['sentence_quota'], 1)
        self.assertEqual(result['observed_facts'][0]['text'], 'Bố Nobita rời khỏi nhà.')
        self.assertEqual(result['observed_facts'][0]['original_observation_text'], 'Bố Nobita về nhà.')
        self.assertEqual(result['sentence_slots'][0]['focus'], 'Bố Nobita rời khỏi nhà.')
        for key in ('unit_id', 'fact_ids', 'shot_ids', 'visual_ids'):
            self.assertEqual(result['sentence_slots'][0][key], original['sentence_slots'][0][key])
        for key in ('start_seconds', 'end_seconds', 'shot_ids', 'visual_id', 'fact_ids'):
            self.assertEqual(result['visual_sources'][0][key], original['visual_sources'][0][key])

    def test_unclear_missing_unknown_and_english_review_cannot_replace_source(self):
        for response in ({'facts': []}, {'facts': [dict(fact_id='OTHER', status='confirmed', text='Bố đi.')]},
                         {'facts': [dict(fact_id='F1', status='unclear', text='Không rõ.')]},
                         {'facts': [dict(fact_id='F1', status='confirmed',
                           text='The father is leaving the house and walking down the street.')]}):
            event = self.event()
            original = copy.deepcopy(event)
            with self.assertRaises(ValueError):
                reviewed_event(event, response)
            self.assertEqual(event, original)

    def test_repair_uses_neighbors_without_rewriting_them(self):
        scenes = [dict(scene_id=f'S{i}', tts=text) for i, text in enumerate(
            ['Đoạn đầu.', 'Mẹ đứng chờ.', 'Bố về nhà.', 'Doraemon bước tới.', 'Đoạn cuối.'])]
        original = copy.deepcopy(scenes)
        context = coherent_repair_context(scenes, 'S2', {'custom_prompt': 'Kể ngắn gọn.'})
        self.assertEqual(context['previous_part_tail'][-1]['tts'], 'Mẹ đứng chờ.')
        self.assertEqual(context['next_batch_context'][0]['event'], 'Doraemon bước tới.')
        self.assertIn('không viết lại chúng', context['custom_prompt'])
        self.assertIn('không bịa quan hệ nhân quả', context['custom_prompt'])
        self.assertEqual(scenes, original)
        self.assertEqual(coherent_repair_context(scenes, 'S0', {})['previous_part_tail'], [])
        self.assertEqual(coherent_repair_context(scenes, 'S4', {})['next_batch_context'], [])

    def test_review_requires_independent_observation(self):
        prompt = review_prompt(self.event(), [])
        self.assertIn('Không mặc định fact cũ', prompt)
        self.assertIn('status=unclear', prompt)

    def test_app_reviews_rewrites_and_rechecks_only_rejected_scene(self):
        from test_visual_audit_diagnostics import VisualAuditDiagnosticTests
        reviewer = Mock(return_value=json.dumps({'facts': [dict(
            fact_id='F1', status='confirmed', text='Bố Nobita rời khỏi nhà.')]}))
        audit, payload, logs = VisualAuditDiagnosticTests().app_auditor(reviewer)
        event = self.event()
        bad = dict(event, tts='Bố Nobita về nhà.', evidence_sentence_plan_version=1,
                   grounding_audit_method='video', grounding_audit_pending=False,
                   grounding_audit_passed=False, visual_alignment_errors=['unsupported_narration:sai hướng'])
        good = dict(bad, scene_id='S0084', event_id='E0084', tts='Mẹ đứng chờ.',
                    grounding_audit_passed=True, visual_alignment_errors=[])
        payload = {'parts': [{'part': 1, 'scenes': [good, bad]}]}
        repaired = dict(bad, tts='Bố Nobita rời khỏi nhà.',
                        grounding_audit_passed=True, visual_alignment_errors=[])
        env = audit.__globals__
        fields = []
        env.update(required_events=[event], locked_event_by_id={'E0085': event},
            story_plan={'events_by_part': [[event]]}, story_context={}, style='', custom_prompt='',
            source_fact_review_attempted=set(), task_lock=threading.Lock(), TASKS={},
            update_task_fields=lambda *a, **kw: fields.append(kw))
        env['audit_narrations'] = Mock(return_value={'parts': [{'scenes': [repaired]}]})
        with patch('evidence_writer.audit_planned_payload', side_effect=[
                copy.deepcopy(payload), {'parts': [{'scenes': [repaired]}]}]), \
                patch('evidence_writer.write_batch', return_value=[repaired]) as writer:
            result = audit(payload)
        self.assertEqual(reviewer.call_count, 1, str(logs))
        self.assertEqual(result['parts'][0]['scenes'][0], good)
        self.assertEqual(result['parts'][0]['scenes'][1]['tts'], 'Bố Nobita rời khỏi nhà.')
        self.assertEqual(event['observed_facts'][0]['text'], 'Bố Nobita rời khỏi nhà.')
        self.assertEqual(writer.call_args.args[1]['previous_part_tail'][-1]['tts'], 'Mẹ đứng chờ.')
        self.assertEqual(env['audit_narrations'].call_count, 1)
        self.assertIn('S0085', env['source_fact_review_attempted'])
        self.assertTrue(any('grounded_plan_checkpoint' in item for item in fields))
        with patch('evidence_writer.audit_planned_payload', return_value=copy.deepcopy(payload)), \
                patch('evidence_writer.write_batch') as repeated_writer:
            audit(payload)
        self.assertEqual(reviewer.call_count, 1)
        repeated_writer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
