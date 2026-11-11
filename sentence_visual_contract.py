"""Export gate for exact sentence targets and per-sentence source coverage."""
import math
import os

from narrative_contract import sentence_texts
from source_grounded_writer import is_source_grounded


def inspect_sentence_visual_contract(payload, target, duration=None, *, require_audit=True,
                                     source_path=None, source_catalog=None):
    issues, sentences_report = [], []
    if source_path is not None and not os.path.isfile(source_path):
        issues.append({'code': 'SOURCE_VIDEO_MISSING'})
    scenes = [s for p in payload.get('parts') or [] for s in p.get('scenes') or []]
    total = 0
    for scene in scenes:
        sid = scene.get('scene_id') or scene.get('event_id')
        proofs = scene.get('sentence_proofs') or []
        unit_mode = bool(scene.get('narration_unit_version'))
        texts = [p.get('text', '') for p in proofs] if unit_mode else sentence_texts(scene.get('tts') or '')
        if unit_mode:
            if ' '.join(texts) != scene.get('tts'):
                issues.append(dict(code='NARRATION_UNIT_TEXT_MISMATCH', scene_id=sid))
            expected_ids = [u.get('unit_id') for u in scene.get('sentence_slots') or []]
            if not expected_ids or [p.get('unit_id') for p in proofs] != expected_ids:
                issues.append(dict(code='NARRATION_UNIT_IDS_INVALID', scene_id=sid))
        total += len(texts)
        if len(texts) != scene.get('sentence_quota', len(texts)) or len(proofs) != len(texts):
            issues.append(dict(code='SENTENCE_PROOF_COUNT_MISMATCH', scene_id=sid))
        if not is_source_grounded(scene):
            issues.append(dict(code='SOURCE_SIGNATURE_INVALID', scene_id=sid))
        if scene.get('visual_fallback') or scene.get('visual_selection_pending') or scene.get('visual_alignment_errors'):
            issues.append(dict(code='UNVERIFIED_VISUAL', scene_id=sid))
        if require_audit and (scene.get('grounding_audit_passed') is not True
                              or scene.get('grounding_audit_pending') or not scene.get('grounded_plan_locked')):
            issues.append(dict(code='CONTENT_AUDIT_NOT_PASSED', scene_id=sid))
        facts = {f.get('fact_id'): f for f in scene.get('observed_facts') or []}
        visuals = {v.get('visual_id'): v for v in scene.get('locked_visual_shots') or []}
        previous = -1.
        for index, text in enumerate(texts):
            errors = []
            proof = proofs[index] if index < len(proofs) and isinstance(proofs[index], dict) else {}
            visual = visuals.get(proof.get('visual_id'))
            if unit_mode:
                selected_ids = proof.get('visual_ids') or []
                selected = [visuals.get(vid) for vid in selected_ids]
                if not selected or any(v is None for v in selected):
                    errors.append('SENTENCE_VISUAL_MISSING')
                else:
                    available = {shot for v in selected for shot in v.get('shot_ids') or []}
                    if not set(proof.get('shot_ids') or []) <= available:
                        errors.append('NARRATION_UNIT_SHOTS_MISSING')
                    for v in selected:
                        try:
                            a, b = float(v['start_seconds']), float(v['end_seconds'])
                            if not math.isfinite(a + b) or not 0 <= a < b or (duration is not None and b > duration + .05):
                                raise ValueError()
                            if source_catalog is not None:
                                owned = [source_catalog[s] for s in v['shot_ids']]
                                if a < min(s['start_seconds'] for s in owned) - .05 or b > max(s['end_seconds'] for s in owned) + .05:
                                    raise ValueError()
                        except (ValueError, TypeError, KeyError):
                            errors.append('SOURCE_WINDOW_INVALID')
            ids = proof.get('fact_ids')
            if proof.get('text') != text:
                errors.append('SENTENCE_PROOF_TEXT_MISMATCH')
            if not isinstance(ids, list) or not ids or any(not isinstance(fid, str) or fid not in facts for fid in ids):
                errors.append('SENTENCE_FACT_INVALID')
                ids = []
            elif any(facts[fid].get('status', 'confirmed') != 'confirmed' for fid in ids):
                errors.append('UNCONFIRMED_FACT')
            if not visual:
                errors.append('SENTENCE_VISUAL_MISSING')
            else:
                start = end = None
                try:
                    start, end = float(visual['start_seconds']), float(visual['end_seconds'])
                    if not math.isfinite(start + end) or not 0 <= start < end or (duration is not None and end > duration + .05):
                        raise ValueError()
                    if start < previous:
                        errors.append('SOURCE_ORDER_INVALID')
                    previous = start
                except (KeyError, TypeError, ValueError, OverflowError):
                    errors.append('SOURCE_WINDOW_INVALID')
                shots = set(visual.get('shot_ids') or ([visual['shot_id']] if visual.get('shot_id') else []))
                if not shots:
                    errors.append('SOURCE_SHOT_ID_MISSING')
                if source_catalog is not None:
                    if any(shot not in source_catalog for shot in shots):
                        errors.append('SOURCE_SHOT_UNKNOWN')
                    elif shots and start is not None and end is not None and math.isfinite(start + end):
                        try:
                            windows = sorted((float(source_catalog[shot]['start_seconds']),
                                              float(source_catalog[shot]['end_seconds'])) for shot in shots)
                            cursor = start
                            for a, b in windows:
                                if a > cursor + .05:
                                    break
                                if b > cursor:
                                    cursor = b
                            if start < windows[0][0] - .05 or cursor < end - .05:
                                errors.append('SOURCE_WINDOW_DOES_NOT_MATCH_SHOT_IDS')
                        except (KeyError, TypeError, ValueError):
                            errors.append('SOURCE_WINDOW_DOES_NOT_MATCH_SHOT_IDS')
                cited = {shot for fid in ids for shot in facts[fid].get('shot_ids') or []}
                kind = proof.get('evidence_kind')
                if kind == 'visual' and not shots & cited:
                    errors.append('SENTENCE_VISUAL_UNRELATED')
                if kind == 'visual' and any(facts[fid].get('evidence_kind', 'visual') != 'visual' for fid in ids):
                    errors.append('EVIDENCE_KIND_INVALID')
                if kind == 'dialogue':
                    if not any(facts[fid].get('evidence_kind') == 'dialogue' and facts[fid].get('spoken_text') for fid in ids):
                        errors.append('DIALOGUE_CONTEXT_IS_NOT_VISUAL_EVIDENCE')
                elif kind != 'visual':
                    errors.append('EVIDENCE_KIND_INVALID')
            record = dict(scene_id=sid, sentence=index + 1, visual_id=proof.get('visual_id'),
                          fact_ids=ids, passed=not errors, issues=errors)
            sentences_report.append(record)
            issues.extend(dict(code=code, scene_id=sid, sentence=index + 1) for code in errors)
    if total != int(target):
        issues.append(dict(code='EXACT_SENTENCE_TARGET_MISMATCH', expected=int(target), actual=total))
    return dict(passed=not issues, target=int(target), actual=total,
                linked_sentence_count=sum(r['passed'] for r in sentences_report),
                sentences=sentences_report, issues=issues)


def enforce_sentence_visual_contract(payload, target, duration=None, **options):
    report = inspect_sentence_visual_contract(payload, target, duration, **options)
    if not report['passed']:
        first = report['issues'][0]
        raise ValueError('SENTENCE_VISUAL_EXPORT_BLOCKED:' + str(first)
                         + f'; target={report["target"]}; actual={report["actual"]}')
    return report


def preflight_visual_plan(plan, duration):
    """Check the source inventory before asking the model to write any prose."""
    count = 0
    for group in plan.get('events_by_part') or []:
        for event in group:
            quota = int(event.get('sentence_quota') or 0)
            count += quota
            sources = event.get('visual_sources') or []
            if quota < 1 or not sources or not event.get('observed_facts'):
                raise ValueError('VISUAL_PLAN_NOT_READY:' + event['event_id'])
            if event.get('narration_unit_version'):
                slots = event.get('sentence_slots') or []
                if len(slots) != quota or len({u.get('unit_id') for u in slots}) != quota:
                    raise ValueError('NARRATION_UNIT_PLAN_COUNT_INVALID:' + event['event_id'])
                facts = {f['fact_id'] for f in event['observed_facts']}
                for slot in slots:
                    if not slot.get('shot_ids') or not slot.get('visual_ids') or not slot.get('fact_ids') or not set(slot['fact_ids']) <= facts:
                        raise ValueError('NARRATION_UNIT_PLAN_SOURCE_MISSING:' + slot['unit_id'])
            shots = set()
            for source in sources:
                start, end = float(source['start_seconds']), float(source['end_seconds'])
                if not math.isfinite(start + end) or not 0 <= start < end <= duration + .05:
                    raise ValueError('VISUAL_PLAN_SOURCE_OUTSIDE_VIDEO:' + event['event_id'])
                shots.update(source.get('shot_ids') or [source.get('shot_id')])
            for fact in event['observed_facts']:
                if not set(fact.get('shot_ids') or []) & shots:
                    raise ValueError('VISUAL_PLAN_FACT_WITHOUT_SOURCE:' + fact['fact_id'])
    target = int(plan['sentence_contract']['target_count'])
    if count != target:
        raise ValueError(f'VISUAL_PLAN_SENTENCE_TARGET_MISMATCH:{count}/{target}')
    return dict(passed=True, target=target, planned=count)
