"""Bounded video checks for event claims, independent of broad chapter analysis."""

import copy
import hashlib
import json
import math

REVIEW_VERSION = 1
BATCH_SIZE = 6


def review_visual_evidence(chunks, duration, review, checkpoint=None, save=None, log=None,
                           source_signature=None):
    """Inspect actual short clips, retry only failures, and never guess source times.

    review(windows) returns {"reviews": [{"review_id": ..., "supported": bool,
    "content_kind": "story", "observed_action": ..., "supported_facts": [...],
    "shots": [{"start_offset": ..., "end_offset": ...}]}]}.
    Offsets are LOCAL to each attached clip. A wider second look is bounded to
    60 seconds and must substantiate the SAME event, not a replacement story.
    Successful checks are durable even when transport or semantic checks fail.
    """
    output = copy.deepcopy(chunks)
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('VISUAL_REVIEW_DURATION_INVALID')
    signature = hashlib.sha256(json.dumps(
        {'version': REVIEW_VERSION, 'duration': duration, 'chunks': chunks, 'source': source_signature},
        ensure_ascii=False, sort_keys=True,
    ).encode()).hexdigest()
    state = copy.deepcopy(checkpoint or {})
    if state.get('signature') != signature:
        state = {'signature': signature, 'version': REVIEW_VERSION, 'confirmed': {}}
    confirmed = state.setdefault('confirmed', {})

    def persist():
        if save:
            save(copy.deepcopy(state))

    pending = []
    events = {}
    for chunk in output:
        for event in chunk.get('events') or []:
            eid = str(event.get('evidence_id', '')).strip().upper()
            if not eid or eid in events:
                raise ValueError('VISUAL_REVIEW_EVENT_ID_INVALID:' + eid)
            events[eid] = event
            sources = event.get('visual_sources') or [{
                'start_seconds': event.get('start_seconds'),
                'end_seconds': event.get('end_seconds'),
            }]
            for index, source in enumerate(sources):
                rid = f'{eid}:{index + 1}'
                if rid in confirmed:
                    continue
                try:
                    start = float(source['start_seconds'])
                    end = float(source['end_seconds'])
                except (KeyError, TypeError, ValueError):
                    raise ValueError('VISUAL_REVIEW_SOURCE_INVALID:' + rid) from None
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + .05):
                    raise ValueError('VISUAL_REVIEW_SOURCE_INVALID:' + rid)
                pending.append({'review_id': rid, 'evidence_id': eid,
                                'source_start': start, 'source_end': end,
                                'event': event.get('event', ''),
                                'action': source.get('event') or event.get('action') or event.get('event', ''),
                                'facts': [str(event.get(key) or '').strip() for key in
                                          ('state_before', 'cause', 'action', 'event', 'visual_detail',
                                           'change', 'result', 'consequence', 'state_after', 'outcome')
                                          if str(event.get(key) or '').strip()]})
    unresolved = []
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        for attempt in range(2):
            if not batch:
                break
            windows = []
            for item in batch:
                margin = 2.0 if attempt == 0 else 20.0
                start = max(0.0, item['source_start'] - margin)
                end = min(duration, item['source_end'] + margin, start + (20.0 if attempt == 0 else 60.0))
                windows.append(dict(item, clip_start=start, clip_end=end))
            if log:
                log(f'Kiem chung hinh that: {len(windows)} shot, luot {attempt + 1}/2; chi xem clip ngan.')
            payload = review(windows)
            rows = payload.get('reviews') if isinstance(payload, dict) else None
            lookup = {}
            duplicate_ids = set()
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                rid = row.get('review_id')
                if not isinstance(rid, str):
                    continue
                if rid in lookup:
                    duplicate_ids.add(rid)
                lookup[rid] = row
            failed = []
            for window in windows:
                rid = window['review_id']
                row = lookup.get(rid, {})
                shots = []
                if (rid not in duplicate_ids and row.get('supported') is True
                        and row.get('content_kind') == 'story'
                        and str(row.get('observed_action') or '').strip()):
                    raw_shots = row.get('shots')
                    for shot in raw_shots if isinstance(raw_shots, list) else []:
                        try:
                            start = float(shot['start_offset'])
                            end = float(shot['end_offset'])
                        except (TypeError, KeyError, ValueError):
                            continue
                        if not (math.isfinite(start) and math.isfinite(end)
                                and 0 <= start < end <= window['clip_end'] - window['clip_start'] + .01
                                and .05 < end - start <= 15.0):
                            continue
                        shots.append({'start_seconds': window['clip_start'] + start,
                                      'end_seconds': min(window['clip_end'], window['clip_start'] + end),
                                      'event': str(row['observed_action']).strip(),
                                      'verified_visual_shot': True, 'semantic_verified': True,
                                      'visual_match_tier': 'short_clip_verified'})
                if shots:
                    facts = row.get('supported_facts') or []
                    facts = facts if isinstance(facts, list) else []
                    # The verifier may only confirm supplied facts; it cannot
                    # expand capacity by inventing additional observations.
                    allowed = set(window['facts'])
                    facts = list(dict.fromkeys(f for f in facts if isinstance(f, str) and f in allowed))
                    confirmed[rid] = {'shots': shots, 'supported_facts': facts}
                else:
                    failed.append(window)
            persist()
            batch = failed
        unresolved.extend(item['review_id'] for item in batch)
    if unresolved:
        state['unresolved'] = unresolved
        persist()
        raise ValueError('VISUAL_EVIDENCE_UNCONFIRMED: ' + ', '.join(unresolved)
                         + '; khong tim thay dung hanh dong trong clip nguon sau 2 luot. '
                         'Da giu checkpoint; can phan tich lai cac su kien nay, khong dung canh thay the.')
    for eid, event in events.items():
        checks = [value for key, value in confirmed.items() if key.rsplit(':', 1)[0] == eid]
        sources = [shot for check in checks for shot in check['shots']]
        unique = {(round(s['start_seconds'], 3), round(s['end_seconds'], 3)): s for s in sources}
        event['visual_sources'] = sorted(unique.values(), key=lambda s: s['start_seconds'])
        event['visual_review_version'] = REVIEW_VERSION
        event['visual_supported_facts'] = list(dict.fromkeys(
            fact for check in checks for fact in check['supported_facts']))
        supported = set(event['visual_supported_facts'])
        for field in ('state_before', 'cause', 'action', 'visual_detail', 'change',
                      'result', 'consequence', 'state_after', 'outcome'):
            if str(event.get(field) or '').strip() not in supported:
                event[field] = ''
        event['event'] = ' '.join(dict.fromkeys(shot['event'] for shot in event['visual_sources']))
        # Event chronology now follows observed clips, rather than a stale
        # chapter guess which may place a repaired visual outside its owner.
        event['start_seconds'] = min(s['start_seconds'] for s in sources)
        event['end_seconds'] = max(s['end_seconds'] for s in sources)
    state.pop('unresolved', None)
    persist()
    return output


def visual_review_prompt(windows):
    cards = [{key: window[key] for key in ('review_id', 'event', 'action', 'facts')}
             for window in windows]
    return (
        'Kiem chung tung clip nguon ngan voi dung review_id gan truoc clip. '
        'Chi ket luan tu hinh dang xem, khong dung tri nho phim hoac loi thoai de doan hanh dong. '
        'supported=true chi khi hanh dong duoc yeu cau THUC SU xuat hien. '
        'Cung nhan vat nhung sai hanh dong/dia diem/ket qua la false. '
        'Opening, credits, title card khong phai bang chung cho hanh dong cot truyen. '
        'Neu tim thay o cho khac TRONG clip, tra shots moi; offset tinh tu DAU CLIP cua ID do, '
        'khong dung thoi gian phim goc. Moi shot 0.05-15 giay, khong lay khoang trong khong lien quan. '
        'supported_facts chi sao chep nguyen van cac facts da duoc clip chung minh; '
        'khong xac nhan cam xuc, nguyen nhan hay ket qua khi chua thay bang chung. '
        'Tra JSON {"reviews":[{"review_id":"...","supported":false,'
        '"content_kind":"story|credits|unknown","observed_action":"",'
        '"supported_facts":[],"shots":[{"start_offset":0,"end_offset":1}]}]}. '
        'Bat buoc tra du moi review_id. DU LIEU: ' + json.dumps(cards, ensure_ascii=False)
    )
