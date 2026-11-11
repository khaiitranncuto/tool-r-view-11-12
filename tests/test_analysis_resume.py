import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis_resume import request_fingerprint, recovery_fields, remember_analysis, load_recovery


class AnalysisResumeTests(unittest.TestCase):
    def test_http_retry_copies_completed_groups_into_new_project_only_for_same_settings(self):
        import app as application
        import task_store
        created = []
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'film.mp4'
            video.write_bytes(b'offline-video')
            logs = Path(directory) / 'logs'
            logs.mkdir()
            with patch.object(application, 'WORKSPACE_DIR', directory), \
                 patch.object(application, 'LOG_DIR', str(logs)), \
                 patch.object(task_store, 'JOB_DIR', str(Path(directory) / '.jobs')), \
                 patch('app.validate_gemini_api_key', return_value=None), patch('app.threading.Thread'):
                client = application.app.test_client()
                body = {'video_name': 'film.mp4', 'api_key': 'offline-key', 'groq_api_key': 'offline-groq',
                        'review_minutes': 20, 'project_id': 'new-ui-project'}
                try:
                    first = client.post('/api/analyze-video', json=body)
                    self.assertEqual(first.status_code, 200)
                    first_id = first.get_json()['task_id']
                    created.append(first_id)
                    application.update_task_fields(first_id, status='failed',
                        evidence_writing_batches={'saved': ['good group']},
                        story_plan_checkpoint={'plan': 'saved plan'})
                    second = client.post('/api/analyze-video', json=dict(body, project_id='another-ui-project')).get_json()
                    created.append(second['task_id'])
                    self.assertTrue(second['resumed'])
                    state = application.TASKS[second['task_id']]
                    self.assertEqual(state['project_id'], 'another-ui-project')
                    self.assertEqual(state['status'], 'running')
                    self.assertEqual(state['evidence_writing_batches'], {'saved': ['good group']})
                    third = client.post('/api/analyze-video', json=dict(body, review_minutes=10)).get_json()
                    created.append(third['task_id'])
                    self.assertFalse(third['resumed'])
                    self.assertNotIn('evidence_writing_batches', application.TASKS[third['task_id']])
                finally:
                    for task_id in created:
                        application.TASKS.pop(task_id, None)
                        application._task_log_checkpoint_at.pop(task_id, None)
                    application.PROJECTS.pop('new-ui-project', None)
                    application.PROJECTS.pop('another-ui-project', None)

    def test_recovery_requires_same_source_settings_and_failed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'film.mp4'
            video.write_bytes(b'video')
            fingerprint = request_fingerprint(video, {'fps': 2, 'sentences': 281})
            state = {'status': 'failed', 'analysis_request_fingerprint': fingerprint,
                'project_id': 'old', 'result': '[error]', 'api_key': 'never-copy',
                'story_plan_checkpoint': {'plan': 'same'},
                'evidence_writing_batches': {'batch': ['complete']},
                'evidence_text_audit': {'signature': {'faithful': True}},
                'primary_grounding_responses': {'1': 'observations'}}
            fields = recovery_fields(state, fingerprint)
            self.assertEqual(fields['evidence_writing_batches'], {'batch': ['complete']})
            self.assertNotIn('project_id', fields)
            self.assertNotIn('status', fields)
            self.assertNotIn('result', fields)
            self.assertNotIn('api_key', fields)
            remember_analysis(directory, fingerprint, 'previous-id')
            tid, recovered = load_recovery(directory, fingerprint, lambda _: state)
            self.assertEqual(tid, 'previous-id')
            self.assertEqual(recovered, fields)
            for status in ('running', 'success'):
                self.assertEqual(recovery_fields(dict(state, status=status), fingerprint), {})
            self.assertNotEqual(fingerprint, request_fingerprint(video, {'fps': 2, 'sentences': 280}))
            video.write_bytes(b'different video')
            changed = request_fingerprint(video, {'fps': 2, 'sentences': 281})
            self.assertEqual(recovery_fields(state, changed), {})

    def test_damaged_or_missing_registry_is_not_a_failed_request(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_recovery(directory, 'key', lambda _: None), (None, {}))
            (Path(directory) / 'key.json').write_text('{broken', encoding='utf-8')
            self.assertEqual(load_recovery(directory, 'key', lambda _: None), (None, {}))


if __name__ == '__main__':
    unittest.main()
