import os
import tempfile
import unittest
from unittest import mock

import task_store


class PersistentTaskStoreTests(unittest.TestCase):
    def test_state_and_payload_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(task_store, "JOB_DIR", temp_dir):
            task_id = "11111111-1111-1111-1111-111111111111"
            task_store.save_task_state(task_id, {"status": "running", "progress": 25})
            task_store.save_task_payload(task_id, {"sentences": ["Cau mot"]})

            state = task_store.load_task_state(task_id)
            payload = task_store.load_task_payload(task_id)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["progress"], 25)
            self.assertEqual(payload["sentences"], ["Cau mot"])
            self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(temp_dir)))

    def test_invalid_task_id_cannot_escape_job_directory(self):
        with self.assertRaises(ValueError):
            task_store.task_state_path("../outside")

    def test_failed_serialization_keeps_previous_checkpoint_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(task_store, 'JOB_DIR', directory):
            task_id = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
            task_store.save_task_state(task_id, {'status': 'running'})
            with self.assertRaises(TypeError):
                task_store.save_task_state(task_id, {'invalid': object()})
            self.assertEqual(task_store.load_task_state(task_id)['status'], 'running')
            self.assertFalse(any(name.endswith('.tmp') for name in os.listdir(directory)))

    def test_failed_replace_keeps_previous_checkpoint_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(task_store, 'JOB_DIR', directory):
            task_id = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
            task_store.save_task_state(task_id, {'status': 'running'})
            with mock.patch('task_store.os.replace', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    task_store.save_task_state(task_id, {'status': 'success'})
            self.assertEqual(task_store.load_task_state(task_id)['status'], 'running')
            self.assertFalse(any(name.endswith('.tmp') for name in os.listdir(directory)))

    def test_worker_startup_failure_is_persisted_before_heartbeat(self):
        import render_worker
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(task_store, 'JOB_DIR', directory), \
             mock.patch.object(render_worker, 'JOB_DIR', directory), \
             mock.patch('render_worker._run_render', side_effect=ValueError('Invalid render payload')):
            task_id = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
            task_store.save_task_state(task_id, {'status': 'queued', 'worker_pid': os.getpid()})
            with self.assertRaisesRegex(ValueError, 'Invalid render payload'):
                render_worker.main(task_id)
            state = task_store.load_task_state(task_id)
            self.assertEqual(state['status'], 'failed')
            self.assertIn('Invalid render payload', state['message'])


if __name__ == "__main__":
    unittest.main()
