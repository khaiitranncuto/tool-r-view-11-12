import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from app import generate_gemini_text_with_recovery
from request_diagnostics import RequestTrace


class Fake503(Exception):
    code = 503

    def __str__(self):
        return '503 UNAVAILABLE high demand secret-api-key'


class RequestDiagnosticTests(unittest.TestCase):
    def test_actual_helper_records_all_attempts_waits_and_stop(self):
        rows = []
        models = SimpleNamespace(generate_content=Mock(side_effect=Fake503()))
        with self.assertRaises(Fake503):
            generate_gemini_text_with_recovery(SimpleNamespace(models=models), 'model',
                ['private prompt secret-api-key'], None, 'writer part 1 batch 1',
                sleep_fn=lambda _: None, capacity_max_retries=2, diagnostic_callback=rows.append)
        self.assertEqual(models.generate_content.call_count, 3)
        failures = [r for r in rows if r['event'] == 'failure']
        self.assertEqual([r['attempt'] for r in failures], [1, 2, 3])
        self.assertTrue(all(r['http_code'] == 503 and r['sent'] for r in failures))
        self.assertEqual([r['wait_seconds'] for r in rows if r['event'] == 'retry_wait'], [180, 180])
        self.assertEqual(rows[-1]['reason'], 'attempt_limit')
        self.assertNotIn('secret-api-key', json.dumps(rows))
        self.assertNotIn('private prompt', json.dumps(rows))

    def test_stream_success_reports_first_chunk_and_inherits_task_logger(self):
        rows = []
        def usage(*_):
            pass
        usage.diagnostic_callback = rows.append
        models = SimpleNamespace(generate_content_stream=Mock(return_value=iter([
            SimpleNamespace(text='private result', usage_metadata=None)])))
        text = generate_gemini_text_with_recovery(SimpleNamespace(models=models), 'model', ['prompt'],
            None, 'writer', stream=True, usage_callback=usage)
        self.assertEqual(text, 'private result')
        self.assertEqual([r['event'] for r in rows], ['prepare', 'send', 'first_chunk', 'success'])
        self.assertNotIn('private result', json.dumps(rows))

    def test_pacing_failure_is_distinguished_from_provider_request(self):
        rows = []
        pacer = SimpleNamespace(reserve_request=Mock(side_effect=ValueError('local pacing failed')))
        models = SimpleNamespace(generate_content=Mock())
        with self.assertRaises(ValueError):
            generate_gemini_text_with_recovery(SimpleNamespace(models=models), 'model', [],
                None, 'writer', quota_pacer=pacer, diagnostic_callback=rows.append)
        models.generate_content.assert_not_called()
        failure = next(r for r in rows if r['event'] == 'failure')
        self.assertFalse(failure['sent'])
        self.assertIsNone(failure['http_code'])

    def test_http_504_and_transport_timeout_have_separate_categories(self):
        rows = []
        trace = RequestTrace(rows.append, 'writer', 'model', [], None)
        trace.begin()
        trace.send()
        error = RuntimeError()
        error.code = 504
        trace.failure(error)
        trace.failure(TimeoutError())
        categories = [r['category'] for r in rows if r['event'] == 'failure']
        self.assertEqual(categories, ['provider_gateway_timeout', 'transport_timeout'])

    def test_log_callback_failure_cannot_break_successful_generation(self):
        models = SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(text='ok', usage_metadata=None)))
        result = generate_gemini_text_with_recovery(SimpleNamespace(models=models), 'model', [], None,
            'writer', diagnostic_callback=Mock(side_effect=OSError('disk full')))
        self.assertEqual(result, 'ok')

    def test_all_populated_chapter_checkpoint_writes_preserve_identity_version(self):
        tree = ast.parse((Path(__file__).parents[1] / 'app.py').read_text(encoding='utf-8'))
        populated = []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'chapter_evidence_checkpoint':
                if isinstance(node.value, ast.Dict) and node.value.keys:
                    keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                    populated.append(keys)
                    self.assertIn('character_registry_version', keys)
        self.assertGreaterEqual(len(populated), 2)
