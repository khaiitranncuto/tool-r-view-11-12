import ast
import copy
import json
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace
import unittest
import uuid

from grounded_pipeline import audit_narrations
from visual_audit_diagnostics import parse_audit_response, inspect_audit_response, summarize_audit_scenes


class VisualAuditDiagnosticTests(unittest.TestCase):
    def test_parser_preserves_all_verdicts_in_direct_list_and_checks_envelope(self):
        rows = [{'scene_id': f'S{i:04d}', 'faithful': i != 3,
                 'reason': 'wrong person' if i == 3 else ''} for i in range(1, 7)]
        for document in (rows, {'checks': rows}, [{'checks': rows}],
                         [{'checks': rows[:3]}, {'checks': rows[3:]}]):
            for raw in (json.dumps(document), '```json\n' + json.dumps(document) + '\n```',
                        'Audit result:\n' + json.dumps(document)):
                parsed = parse_audit_response(raw)
                self.assertEqual(parsed['checks'], rows)
                report, accepted = inspect_audit_response(parsed, [r['scene_id'] for r in rows])
                self.assertTrue(report['complete'])
                self.assertEqual(len(accepted), 6)
                self.assertEqual(report['rejected'], [{'scene_id': 'S0003', 'reason': 'wrong person'}])

    def test_parser_does_not_hide_duplicates_invent_ids_or_coerce_boolean(self):
        raw = '[{"scene_id":"S1","faithful":true},{"scene_id":"S1","faithful":false,' \
              '"reason":"wrong"},{"scene_id":"S2","faithful":"true"},' \
              '{"event_id":"S3","faithful":true},{"scene_id":"OTHER","faithful":true}]'
        report, accepted = inspect_audit_response(parse_audit_response(raw), ['S1', 'S2', 'S3'])
        self.assertEqual(accepted, {})
        self.assertEqual(report['duplicate_ids'], ['S1'])
        self.assertEqual(report['unexpected_ids'], ['OTHER'])
        self.assertEqual([r['reason'] for r in report['unresolved']],
                         ['duplicate_scene_id', 'faithful_not_boolean', 'scene_id_not_returned'])

    def test_parser_rejects_truncated_outer_documents_and_mixed_wrappers(self):
        row = {'scene_id': 'S1', 'faithful': True}
        for raw in ('[' + json.dumps(row) + ',', '{"checks":[' + json.dumps(row), '',
                    json.dumps(row) + json.dumps(row),
                    json.dumps([{'checks': [row]}, row])):
            with self.assertRaisesRegex(ValueError, 'AUDIT_JSON_INVALID'):
                parse_audit_response(raw)
        self.assertEqual(parse_audit_response(json.dumps(row)), {'checks': [row]})

    def test_diagnostics_distinguish_rejected_missing_and_invalid_verdicts(self):
        report, accepted = inspect_audit_response({'checks': [
            {'scene_id': 'S1', 'faithful': True},
            {'scene_id': 'S2', 'faithful': False, 'reason': 'wrong person'},
            {'scene_id': 'S3', 'faithful': 'true'},
            {'scene_id': 'S4', 'faithful': False},
            {'scene_id': 'S5', 'faithful': True}, {'scene_id': 'S5', 'faithful': True},
            {'scene_id': 'OTHER', 'faithful': True}, {'event_id': 'S6', 'faithful': True},
        ]}, ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
        self.assertEqual(set(accepted), {'S1', 'S2'})
        self.assertEqual(report['passed_ids'], ['S1'])
        self.assertEqual(report['rejected'], [{'scene_id': 'S2', 'reason': 'wrong person'}])
        self.assertEqual({r['scene_id']: r['reason'] for r in report['unresolved']}, {
            'S3': 'faithful_not_boolean', 'S4': 'rejection_reason_missing',
            'S5': 'duplicate_scene_id', 'S6': 'scene_id_not_returned'})
        self.assertEqual(report['unexpected_ids'], ['OTHER'])
        self.assertEqual(len(report['invalid_rows']), 1)

    def test_missing_schema_is_not_reported_as_content_rejection(self):
        for response in ([], {}, {'checks': 'invalid'}):
            report, accepted = inspect_audit_response(response, ['S1'])
            self.assertEqual(accepted, {})
            self.assertEqual(report['schema_error'], 'checks_missing_or_not_list')
            self.assertEqual(report['rejected'], [])

    def test_diagnostic_callback_preserves_retry_cache_and_explains_pending(self):
        payload = {'parts': [{'part': 1, 'scenes': [
            {'scene_id': sid, 'tts': 'Mở cửa ' + sid, 'grounded_plan_locked': True,
             'observed_facts': []} for sid in ('S1', 'S2', 'S3')]}]}
        events, cache, calls = [], {}, []
        def check(cards):
            calls.append([c['scene_id'] for c in cards])
            return {'checks': [{'scene_id': 'S1', 'faithful': True},
                               {'scene_id': 'S2', 'faithful': False, 'reason': 'wrong action'}]}
        def save(state):
            cache.clear()
            cache.update(state)
        result = audit_narrations(payload, check, save=save, max_attempts=2,
            diagnostic=lambda event, data: events.append((event, data)))
        self.assertEqual(calls, [['S1', 'S2', 'S3'], ['S3']])
        self.assertEqual(len(cache), 2)
        summary = summarize_audit_scenes(result)
        self.assertEqual(summary['passed_ids'], ['S1'])
        self.assertEqual(summary['rejected_ids'], ['S2'])
        self.assertEqual(summary['pending_ids'], ['S3'])
        retry = [data for event, data in events if event == 'AUDIT_PENDING_AFTER_ATTEMPT']
        self.assertEqual([r['will_retry'] for r in retry], [True, False])
        self.assertEqual(events[-1], ('AUDIT_SUMMARY', summary))

    def app_auditor(self, generate):
        # Run the actual nested app callback without a key, server, or upload.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'audit_grounded_payload')
        factory = ast.parse('def factory():\n    pass1_calls = 0\n').body[0]
        factory.body.extend([node, ast.Return(ast.Name(id=node.name, ctx=ast.Load()))])
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        events = []
        fake_os = SimpleNamespace(stat=lambda path: SimpleNamespace(st_size=123, st_mtime_ns=456),
                                  path=SimpleNamespace(realpath=lambda path: path))
        env = dict(uuid=uuid, time=time, json=json, copy=copy, traceback=traceback, os=fake_os,
            selected_model='offline-model', api_key='test-secret', video_path='source.mp4',
            append_task_log=lambda task, level, message, details='': events.append(
                (level, message, json.loads(details) if details.startswith('{') else details)),
            task_id='offline', update_task_fields=lambda *args, **kwargs: None,
            visual_batches=lambda cards, **kwargs: [cards], visual_call_lock=threading.Lock(),
            ensure_visual_pack=lambda *args: (object(), [{'source_start': 6, 'source_end': 12}], 'pack.mp4'),
            generate_gemini_text_with_recovery=generate, visual_client=None, writing_config=None,
            record_gemini_usage=lambda *args: None, gemini_quota_pacer=None,
            estimate_gemini_video_input_tokens=lambda *args: 100, update_prog=lambda *args: None,
            parse_json_object=json.loads, grounded_audit_cache={}, audit_narrations=audit_narrations,
            is_gemini_auth_error=lambda exc: False, is_daily_project_gemini_quota_error=lambda exc: False,
            is_gemini_transport_timeout=lambda exc: 'timed out' in str(exc),
            is_gemini_capacity_503_error=lambda exc: False,
            is_per_minute_gemini_quota_error=lambda exc: False,
            gemini_error_chain_text=str)
        exec(compile(module, '<app visual audit>', 'exec'), env)
        payload = {'parts': [{'part': 1, 'scenes': [{'scene_id': 'S0001', 'tts': 'Mở cửa.',
            'grounded_plan_locked': True, 'observed_facts': [{'text': 'Mở cửa.'}],
            'locked_visual_shots': [{'shot_id': 'SHOT_000002', 'start': '00:06', 'end': '00:12'}]}]}]}
        return env['factory'](), payload, events

    def test_app_logs_raw_reply_and_schema_error_without_rejecting_content(self):
        raw = '{"results":[{"scene_id":"S0001","faithful":true}]}'
        audit, payload, events = self.app_auditor(lambda *args, **kwargs: raw)
        result = audit(payload)
        self.assertTrue(result['parts'][0]['scenes'][0]['grounding_audit_pending'])
        raw_logs = [data for _, _, data in events if isinstance(data, dict)
                    and data.get('event') == 'AUDIT_RESPONSE_RAW']
        self.assertEqual(len(raw_logs), 2)
        self.assertEqual(raw_logs[0]['raw_response'], raw)
        self.assertEqual(raw_logs[0]['expected_ids'], ['S0001'])
        reports = [data for _, _, data in events if isinstance(data, dict)
                   and data.get('event') == 'AUDIT_REQUEST_RESULT']
        self.assertEqual(reports[0]['schema_error'], 'checks_missing_or_not_list')

    def test_app_accepts_direct_array_without_retry_and_reuses_verified_cache(self):
        calls = []
        def generate(*args, **kwargs):
            calls.append(True)
            return '[{"scene_id":"S0001","faithful":true,"reason":""}]'
        audit, payload, events = self.app_auditor(generate)
        result = audit(payload)
        self.assertTrue(result['parts'][0]['scenes'][0]['grounding_audit_passed'])
        self.assertFalse(result['parts'][0]['scenes'][0]['grounding_audit_pending'])
        final = audit(result, cache_only=True)
        self.assertTrue(final['parts'][0]['scenes'][0]['grounding_audit_passed'])
        self.assertEqual(len(calls), 1)

    def test_app_keeps_rejection_from_direct_array(self):
        audit, payload, events = self.app_auditor(lambda *args, **kwargs:
            '[{"scene_id":"S0001","faithful":false,"reason":"wrong person"}]')
        result = audit(payload)
        scene = result['parts'][0]['scenes'][0]
        self.assertFalse(scene['grounding_audit_passed'])
        self.assertFalse(scene['grounding_audit_pending'])
        self.assertIn('unsupported_narration:wrong person', scene['visual_alignment_errors'])

    def test_app_logs_timeout_and_redacts_key_in_error_and_traceback(self):
        def generate(*args, **kwargs):
            raise RuntimeError('read operation timed out: test-secret')
        audit, payload, events = self.app_auditor(generate)
        result = audit(payload)
        self.assertTrue(result['parts'][0]['scenes'][0]['grounding_audit_pending'])
        errors = [data for _, _, data in events if isinstance(data, dict)
                  and data.get('event') == 'AUDIT_REQUEST_ERROR']
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0]['category'], 'TIMEOUT')
        self.assertEqual(errors[0]['phase'], 'generate')
        self.assertNotIn('test-secret', json.dumps(errors))
        self.assertIn('[REDACTED_API_KEY]', errors[0]['error'])

    def test_app_logs_final_cache_miss_without_extra_api_call(self):
        calls = []
        def generate(*args, **kwargs):
            calls.append(True)
            return '{"checks":[{"scene_id":"S0001","faithful":true}]}'
        audit, payload, events = self.app_auditor(generate)
        checked = audit(payload)
        changed = copy.deepcopy(checked)
        changed['parts'][0]['scenes'][0]['tts'] = 'Đóng cửa.'
        final = audit(changed, cache_only=True)
        self.assertEqual(len(calls), 1)
        self.assertTrue(final['parts'][0]['scenes'][0]['grounding_audit_pending'])
        transitions = [data for _, _, data in events if isinstance(data, dict)
                       and data.get('event') == 'AUDIT_STATUS_TRANSITION']
        self.assertEqual(transitions[0]['before_passed'], True)
        self.assertEqual(transitions[0]['after_status'], 'pending')


if __name__ == '__main__':
    unittest.main()
