"""Explain visual-audit replies without guessing IDs or changing verdicts."""

import json
import re


def parse_audit_response(text):
    """Read the whole audit document, retaining every verdict and its ID.

    Unlike the general object parser, never select an individual verdict from
    an array or rescue an inner object from a truncated outer document.
    """
    source = str(text or '').lstrip('\ufeff').strip()
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', source, re.I)
    if fenced:
        source = fenced.group(1).strip()
    if not source.startswith(('{', '[')):
        first = re.search(r'[\[{]', source)
        source = source[first.start():] if first else ''
    try:
        value, end = json.JSONDecoder(strict=False).raw_decode(source)
    except json.JSONDecodeError as exc:
        raise ValueError('AUDIT_JSON_INVALID: ' + str(exc)) from exc
    if source[end:].lstrip().startswith(('{', '[')):
        raise ValueError('AUDIT_JSON_INVALID: multiple JSON documents')

    def normalize(document):
        if isinstance(document, list):
            containers = [isinstance(row, dict) and 'checks' in row
                          and 'scene_id' not in row for row in document]
            if any(containers):
                if not all(containers):
                    raise ValueError('AUDIT_JSON_INVALID: mixed verdicts and containers')
                checks = []
                for row in document:
                    if not isinstance(row['checks'], list):
                        raise ValueError('AUDIT_JSON_INVALID: checks container must contain a list')
                    checks.extend(row['checks'])
                return {'checks': checks}
            return {'checks': document}
        if isinstance(document, dict):
            if 'checks' not in document and 'scene_id' in document and 'faithful' in document:
                return {'checks': [document]}
            return document
        raise ValueError('AUDIT_JSON_INVALID: expected object or list')

    return normalize(value)


def inspect_audit_response(response, expected_ids):
    expected_ids = list(expected_ids)
    rows = response.get('checks') if isinstance(response, dict) else None
    report = {
        'response_type': type(response).__name__, 'checks_type': type(rows).__name__,
        'expected_ids': expected_ids, 'received_ids': [], 'duplicate_ids': [],
        'unexpected_ids': [], 'invalid_rows': [], 'unresolved': [],
        'passed_ids': [], 'rejected': [], 'schema_error': None,
    }
    if not isinstance(rows, list):
        report['schema_error'] = 'checks_missing_or_not_list'
    lookup, duplicates = {}, set()
    for index, row in enumerate(rows if isinstance(rows, list) else [], 1):
        if not isinstance(row, dict) or not isinstance(row.get('scene_id'), str):
            report['invalid_rows'].append({'row': index, 'reason': 'scene_id_missing_or_not_string',
                                           'value': row})
            continue
        sid = row['scene_id']
        report['received_ids'].append(sid)
        if sid in lookup:
            duplicates.add(sid)
        if sid not in expected_ids and sid not in report['unexpected_ids']:
            report['unexpected_ids'].append(sid)
        lookup[sid] = row
    report['duplicate_ids'] = sorted(duplicates)
    accepted = {}
    for sid in expected_ids:
        row = lookup.get(sid)
        if sid in duplicates:
            reason = 'duplicate_scene_id'
        elif row is None:
            reason = report['schema_error'] or 'scene_id_not_returned'
        elif type(row.get('faithful')) is not bool:
            reason = 'faithful_not_boolean'
        elif row['faithful'] is False and not str(row.get('reason') or '').strip():
            reason = 'rejection_reason_missing'
        else:
            accepted[sid] = {'faithful': row['faithful'], 'reason': str(row.get('reason') or '')}
            if row['faithful']:
                report['passed_ids'].append(sid)
            else:
                report['rejected'].append({'scene_id': sid, 'reason': accepted[sid]['reason']})
            continue
        report['unresolved'].append({'scene_id': sid, 'reason': reason})
    report['complete'] = len(accepted) == len(expected_ids)
    return report, accepted


def summarize_audit_scenes(payload):
    """Keep missing replies, rejected content, and approximate images separate."""
    scenes = [s for p in payload.get('parts') or [] for s in p.get('scenes') or []
              if s.get('grounded_plan_locked')]
    return {
        'total': len(scenes),
        'passed_ids': [s['scene_id'] for s in scenes if s.get('grounding_audit_passed')],
        'pending_ids': [s['scene_id'] for s in scenes if s.get('grounding_audit_pending')],
        'rejected_ids': [s['scene_id'] for s in scenes
                         if not s.get('grounding_audit_passed') and not s.get('grounding_audit_pending')],
        'fallback_ids': [s['scene_id'] for s in scenes if s.get('visual_fallback')],
    }
