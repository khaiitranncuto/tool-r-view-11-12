"""Copy compatible failed-analysis checkpoints into a new UI project."""
import copy
import hashlib
import json
import os


WORKFLOW_VERSION = 1


def request_fingerprint(video_path, options):
    stat = os.stat(video_path)
    identity = [WORKFLOW_VERSION, os.path.realpath(video_path), stat.st_size, stat.st_mtime_ns, options]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def recovery_fields(state, fingerprint):
    if (not isinstance(state, dict) or state.get('status') != 'failed'
            or state.get('analysis_request_fingerprint') != fingerprint):
        return {}
    names = {'primary_grounding_responses', 'observation_contract_version',
             'full_video_outline', 'full_video_outline_complete', 'grounding_recovery_responses',
             'grounded_gap_responses', 'evidence_writing_batches', 'evidence_writing_candidates',
             'evidence_text_audit', 'grounded_narration_audit'}
    return copy.deepcopy({k: v for k, v in state.items() if k in names or k.endswith('_checkpoint')})


def load_recovery(directory, fingerprint, load_state):
    path = os.path.join(directory, fingerprint + '.json')
    try:
        with open(path, encoding='utf-8') as handle:
            record = json.load(handle)
        task_id = record['task_id']
        fields = recovery_fields(load_state(task_id), fingerprint)
        return (task_id, fields) if fields else (None, {})
    except (OSError, ValueError, TypeError, KeyError):
        return None, {}


def remember_analysis(directory, fingerprint, task_id):
    from task_store import _atomic_write_json
    _atomic_write_json(os.path.join(directory, fingerprint + '.json'), {'task_id': task_id})
