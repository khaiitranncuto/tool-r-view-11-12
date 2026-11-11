import ast
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from recap_visual_media import visual_batches
from visual_audit_diagnostics import parse_audit_response, inspect_audit_response


class AdaptiveVisualAuditTests(unittest.TestCase):
    def test_real_callback_splits_only_failed_group_and_retains_success(self):
        tree = ast.parse(Path('app.py').read_text(encoding='utf-8-sig'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'check' and any(isinstance(x, ast.Name)
                    and x.id == 'pending_batches' for x in ast.walk(n)))
        for index, statement in enumerate(node.body):
            if isinstance(statement, ast.Nonlocal):
                node.body[index] = ast.copy_location(ast.Global(names=statement.names), statement)
        cards = [{'scene_id': f'S{i}'} for i in range(4)]
        calls, events = [], []

        def generate(*args, **kwargs):
            ids = json.loads(args[2][1].split('CAC CAU: ')[1])
            calls.append([c['scene_id'] for c in ids])
            if len(ids) > 1:
                raise RuntimeError('503')
            return json.dumps({'checks': [dict(ids[0], faithful=True)]})

        env = dict(pass1_calls=0, uuid=__import__('uuid'), time=__import__('time'),
                   json=json, traceback=__import__('traceback'), source_offset=0,
                   task_id='test', api_key='', selected_model='test', writing_config=None,
                   visual_client=None, record_gemini_usage=None, gemini_quota_pacer=None,
                   visual_call_lock=__import__('threading').Lock(),
                   visual_batches=lambda cards, **kw: [cards[:3], cards[3:]],
                   ensure_visual_pack=lambda *a: ('media', [], 'test.mp4'),
                   generate_gemini_text_with_recovery=generate,
                   parse_audit_response=parse_audit_response,
                   inspect_audit_response=inspect_audit_response,
                   diagnostic=lambda event, data: events.append((event, data)),
                   estimate_gemini_video_input_tokens=lambda *a: 0,
                   update_prog=Mock(), append_task_log=Mock(),
                   gemini_error_chain_text=str)
        for name in ('auth', 'daily_project_gemini_quota', 'gemini_transport_timeout',
                     'per_minute_gemini_quota'):
            env['is_' + name + '_error'] = lambda exc: False
        env['is_gemini_auth_error'] = lambda exc: False
        env['is_gemini_transport_timeout'] = lambda exc: False
        env['is_gemini_capacity_503_error'] = lambda exc: str(exc) == '503'
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'app.py', 'exec'), env)
        result = env['check'](cards)
        self.assertEqual([r['scene_id'] for r in result['checks']], ['S0', 'S1', 'S2', 'S3'])
        self.assertEqual(calls, [['S0', 'S1', 'S2'], ['S0'], ['S1', 'S2'], ['S1'], ['S2'], ['S3']])
        self.assertEqual(sum(event == 'AUDIT_BATCH_SPLIT' for event, _ in events), 2)

    def test_text_limit_keeps_all_evidence_even_for_oversized_single_card(self):
        cards = [dict(scene_id=f'S{i}', text='x' * 1000, visual_sources=[
            dict(start_seconds=0, end_seconds=6, shot_id='SHOT_000001')]) for i in range(3)]
        batches = list(visual_batches(cards, max_chars=1500))
        self.assertEqual([len(b) for b in batches], [1, 1, 1])
        self.assertEqual([c for batch in batches for c in batch], cards)


if __name__ == '__main__':
    unittest.main()
