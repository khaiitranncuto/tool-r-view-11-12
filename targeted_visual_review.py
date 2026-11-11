"""Reuse primary-pass shots; repair only missing/invalid ranges, with a task cap."""

import copy
import hashlib
import json
import math

MAX_REVIEW_CALLS = 2
MAX_CLIPS_PER_CALL = 6


def repair_missing_visuals(chunks, duration, review, checkpoint=None, save=None,
                           log=None, source_signature=None):
    from recap_pipeline import apply_targeted_visual_shot_repairs, verified_event_visual_sources
    result = copy.deepcopy(chunks)
    fingerprint = hashlib.sha256(json.dumps(
        [chunks, duration, source_signature], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    state = copy.deepcopy(checkpoint or {})
    if state.get('fingerprint') != fingerprint:
        state = {'fingerprint': fingerprint, 'completed_calls': 0, 'responses': []}

    def pending():
        rows = []
        for chunk in result:
            for event in chunk.get('events') or []:
                start, end = float(event['start_seconds']), float(event['end_seconds'])
                shots = []
                for shot in verified_event_visual_sources(event):
                    a, b = float(shot['start_seconds']), float(shot['end_seconds'])
                    if (math.isfinite(a) and math.isfinite(b) and
                            0 <= a < b <= duration + .05
                            and (start - .05 <= a < b <= end + .05 or shot.get('targeted_visual_repair'))):
                        shots.append(shot)
                event['visual_sources'] = shots
                event['visual_ranges_locked'] = bool(shots)
                if shots and any(s.get('targeted_visual_repair') for s in shots):
                    event['start_seconds'] = min(s['start_seconds'] for s in shots)
                    event['end_seconds'] = max(s['end_seconds'] for s in shots)
                if not shots:
                    rows.append(event)
        return rows

    pending()
    for saved in state['responses']:
        result, _ = apply_targeted_visual_shot_repairs(result, saved['text'], saved['windows'])
    for call in range(state['completed_calls'], MAX_REVIEW_CALLS):
        missing = pending()
        # A multi-minute summary is not a localized action. Do not send its
        # first 20 seconds twice and pretend that searches the full event.
        candidates = [e for e in missing if 0 < float(e['end_seconds']) - float(e['start_seconds']) <= 90]
        if not candidates:
            break
        # First visit unseen IDs, then allow one bounded repair of unresolved
        # IDs only if the task has room. There is no all-shots verification.
        seen = {w['evidence_id'] for saved in state['responses'] for w in saved['windows']}
        candidates.sort(key=lambda e: (e['evidence_id'] in seen, float(e['start_seconds'])))
        selected = candidates[:MAX_CLIPS_PER_CALL]
        windows = []
        for event in selected:
            start, end = float(event['start_seconds']), float(event['end_seconds'])
            midpoint = (start + end) / 2
            clip_start = max(0.0, midpoint - 30)
            clip_end = min(duration, clip_start + 60)
            windows.append({'evidence_id': event['evidence_id'],
                            'start_seconds': clip_start, 'end_seconds': clip_end})
        if log:
            log(f'Xem lai co muc tieu {call + 1}/{MAX_REVIEW_CALLS}: {len(selected)} ID thieu/sai shot; giu nguyen shot da co.')
        # The transport layer may retry a timeout once; it must not discard
        # preceding completed responses or trigger re-analysis of chapters.
        text = review(selected, windows, call + 1)
        result, repaired = apply_targeted_visual_shot_repairs(result, text, windows)
        state['responses'].append({'text': text, 'windows': windows})
        state['completed_calls'] = call + 1
        if save:
            save(copy.deepcopy(state))
        if log:
            log(f'Xem lai {call + 1}/{MAX_REVIEW_CALLS}: sua duoc {len(repaired)}/{len(selected)} ID.')
    missing = pending()
    rejected = {e['evidence_id'] for e in missing}
    # Unlocated claims remain context for understanding the story, never
    # arbitrary render ranges or fabricated extra narration capacity.
    for chunk in result:
        kept = []
        for event in chunk.get('events') or []:
            if event['evidence_id'] in rejected:
                chunk.setdefault('story_context', []).append({
                    'evidence_id': event['evidence_id'], 'summary': event.get('event', ''),
                    'status': 'unverified_visual_context',
                })
                continue
            event['depends_on'] = [key for key in event.get('depends_on') or [] if key not in rejected]
            kept.append(event)
        chunk['events'] = kept
    if log:
        log(f'Giu shot tu luot xem chinh; {len(rejected)} ID chua dinh vi chi lam context, khong phan bo cau.')
    return result, {'review_calls': state['completed_calls'], 'unlocated_ids': sorted(rejected)}
