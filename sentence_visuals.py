"""Choose observed shots AFTER narration, in bounded text-only batches."""

import copy
import hashlib
import json
import re

from semantic_visuals import semantic_visual_plan, timecode

BATCH_SIZE = 12
MAX_CANDIDATES = 18
MAX_LOCATION_CALLS = 2


def build_shot_catalog(chunks):
    from recap_pipeline import verified_event_visual_sources
    catalog, lookup = [], {}
    for chunk in chunks:
        for event in chunk.get('events') or []:
            eid = str(event['evidence_id']).upper()
            for source in verified_event_visual_sources(event):
                start, end = float(source['start_seconds']), float(source['end_seconds'])
                if not float(event['start_seconds']) - .05 <= start < end <= float(event['end_seconds']) + .05:
                    continue
                key = (round(start, 3), round(end, 3))
                if key in lookup:
                    lookup[key]['event_ids'].append(eid)
                    continue
                row = {'shot_id': f'SH{len(catalog) + 1:05d}', 'start': start, 'end': end,
                       'action': str(source.get('event') or event.get('event') or ''),
                       'characters': event.get('characters') or [], 'event_ids': [eid]}
                catalog.append(row)
                lookup[key] = row
    return catalog


def _tokens(text):
    return set(re.findall(r'\w+', str(text).casefold()))


def sentence_candidates(scene, catalog, events):
    owner_ids = set(scene.get('planning_source_event_ids') or [])
    related_ids = set(owner_ids)
    for eid in owner_ids:
        event = events.get(eid, {})
        related_ids.update(event.get('depends_on') or [])
        related_ids.update(event.get('context_dependencies') or [])
    # A wrong character/action elsewhere in the film is never a valid fallback.
    rows = [r for r in catalog if related_ids.intersection(r['event_ids'])]
    tokens = _tokens(scene.get('tts', ''))
    rows.sort(key=lambda r: (bool(owner_ids.intersection(r['event_ids'])),
                             len(tokens & _tokens(r['action']))), reverse=True)
    return sorted(rows[:MAX_CANDIDATES], key=lambda r: (r['start'], r['end']))


def shot_selection_prompt(cards):
    shots = {r['shot_id']: r for card in cards for r in card['candidates']}
    # Adjacent sentences often share an event. Send its catalog once, not
    # 3-4 duplicate copies, to keep the text selection pass inexpensive.
    compact = {'shots': list(shots.values()), 'sentences': [
        {'scene_id': c['scene_id'], 'tts': c['tts'],
         'candidate_ids': [r['shot_id'] for r in c['candidates']],
         'source_constraints': {key: value['shot_id'] if value else None
                                for key, value in (c.get('source_constraints') or {}).items()}}
        for c in cards]}
    return (
        'Chon shot cho cac CAU RECAP DA VIET. Chi doc danh muc hinh da quan sat, khong xem lai video. '
        'Giu nguyen loi ke va thu tu cau. Shot phai theo thu tu thoi gian nguon cua toan bo loi ke. '
        'Tuan thu source_constraints; co the dung chung mot shot cho cac cau lien tiep cung hanh dong. '
        'Moi y hanh dong phai khop nhan vat, hanh dong, boi canh va ket qua. '
        'Cung nhan vat nhung sai hanh dong KHONG duoc coi la khop. Khong chon theo vi tri trong danh sach. '
        'Mot cau co the dung nhieu shot. Neu doi shot, trigger_text la cum 2-6 tu xuat hien dung mot lan '
        'trong cau, o dau y can doi hinh. Link dau tien trigger_text="". Chi dung shot_id trong candidates '
        'cua DUNG cau do; khong tu dat timestamp. Neu thieu hinh cho bat ky y quan trong, '
        'dat supported=false va links=[]; khong dung canh gan giong de lap cho trong. '
        'candidate_ids la danh sach shot duoc phep cho tung cau. Tra JSON {"matches":[{"scene_id":"S0001","supported":true,"links":'
        '[{"shot_id":"SH00001","trigger_text":""}],"reason":"hanh dong nao duoc minh hoa"}]}. '
        'Du moi scene_id. DU LIEU: ' + json.dumps(compact, ensure_ascii=False, separators=(',', ':'))
    )


def _validate_choice(scene, choice, candidates):
    if not isinstance(choice, dict) or choice.get('supported') is not True:
        return None
    links = choice.get('links')
    if not isinstance(links, list) or not links:
        return None
    lookup = {r['shot_id']: (i, r) for i, r in enumerate(candidates)}
    normalized, selected, seen = [], [], set()
    for link in links:
        if not isinstance(link, dict) or not isinstance(link.get('shot_id'), str) or link.get('shot_id') not in lookup:
            return None
        sid = link['shot_id']
        if sid in seen:
            return None
        seen.add(sid)
        index, shot = lookup[sid]
        normalized.append({'visual_id': f'V{index + 1:03d}', 'trigger_text': link.get('trigger_text', '')})
        selected.append({'shot_id': sid, 'trigger_text': link.get('trigger_text', '')})
    sources = [{'start': timecode(r['start']), 'end': timecode(r['end']), 'event': r['action']} for r in candidates]
    plan = semantic_visual_plan(scene['tts'], sources, normalized)
    return selected if not plan['visual_alignment_errors'] else None


def select_sentence_shots(payload, chunks, select, locate=None, checkpoint=None, save=None,
                          log=None, source_signature=None):
    """select(cards) reads text only. locate(queries, call) may reopen video.

    Persist accepted choices and any located shots, so a failed request never
    rewrites narration or reselects completed batches. Unknown picture claims
    remain pending and cannot be exported as invented TIME ranges.
    """
    result = copy.deepcopy(payload)
    scenes = [s for p in result['parts'] for s in p.get('scenes') or []]
    if len({s['scene_id'] for s in scenes}) != len(scenes):
        raise ValueError('SENTENCE_SHOT_IDS_DUPLICATE')
    events = {str(e['evidence_id']).upper(): e for c in chunks for e in c.get('events') or []}
    signature = hashlib.sha256(json.dumps([payload, chunks, source_signature],
                               ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    state = copy.deepcopy(checkpoint or {})
    if state.get('signature') != signature:
        state = {'signature': signature, 'choices': {}, 'located': [], 'location_calls': 0}
    catalog = build_shot_catalog(chunks) + state['located']

    def persist():
        if save:
            save(copy.deepcopy(state))

    def constraints(scene_id):
        ids = [s['scene_id'] for s in scenes]
        index = ids.index(scene_id)
        lookup = {r['shot_id']: r for r in catalog}
        previous = next((state['choices'][key][-1]['shot_id'] for key in reversed(ids[:index])
                         if key in state['choices']), None)
        following = next((state['choices'][key][0]['shot_id'] for key in ids[index + 1:]
                          if key in state['choices']), None)
        return {'previous': lookup.get(previous), 'next': lookup.get(following)}

    def fits_neighbors(scene_id, choice):
        bounds = constraints(scene_id)
        lookup = {r['shot_id']: r for r in catalog}
        first, last = lookup[choice[0]['shot_id']], lookup[choice[-1]['shot_id']]
        previous, following = bounds['previous'], bounds['next']
        return not ((previous and first['shot_id'] != previous['shot_id'] and first['start'] < previous['end'] - .01)
                    or (following and last['shot_id'] != following['shot_id'] and last['end'] > following['start'] + .01))

    def choose_pending():
        pending = [s for s in scenes if s['scene_id'] not in state['choices']]
        for offset in range(0, len(pending), BATCH_SIZE):
            batch = pending[offset:offset + BATCH_SIZE]
            cards = [{'scene_id': s['scene_id'], 'tts': s['tts'],
                      'candidates': sentence_candidates(s, catalog, events)} for s in batch]
            cards = [c for c in cards if c['candidates']]
            if not cards:
                continue
            for attempt in range(2):
                if not cards:
                    break
                if log:
                    log(f'Chon hinh sau loi ke: {len(cards)} cau, luot text {attempt + 1}/2; khong gui video.')
                for card in cards:
                    card['source_constraints'] = constraints(card['scene_id'])
                response = select(cards)
                matches = response.get('matches') if isinstance(response, dict) else None
                lookup, duplicates = {}, set()
                for row in matches if isinstance(matches, list) else []:
                    if not isinstance(row, dict) or not isinstance(row.get('scene_id'), str):
                        continue
                    key = row['scene_id']
                    if key in lookup:
                        duplicates.add(key)
                    lookup[key] = row
                by_id = {s['scene_id']: s for s in batch}
                retry = []
                for card in cards:
                    key = card['scene_id']
                    choice = None if key in duplicates else _validate_choice(by_id[key], lookup.get(key), card['candidates'])
                    if choice and fits_neighbors(key, choice):
                        state['choices'][key] = choice
                    elif lookup.get(key, {}).get('supported') is not False:
                        retry.append(card)  # Invalid schema/trigger: text-only local repair.
                persist()
                cards = retry

    choose_pending()
    while locate and state['location_calls'] < MAX_LOCATION_CALLS:
        pending = [s for s in scenes if s['scene_id'] not in state['choices']]
        if not pending:
            break
        queries = []
        for scene in pending:
            owners = [events[k] for k in scene.get('planning_source_event_ids') or [] if k in events]
            if owners:
                queries.append({'scene_id': scene['scene_id'], 'tts': scene['tts'],
                                'event_ids': [e['evidence_id'] for e in owners],
                                'start': min(float(e['start_seconds']) for e in owners),
                                'end': max(float(e['end_seconds']) for e in owners)})
        if not queries:
            break
        # The callback groups sentences sharing one event so its clip is sent
        # once. Returned rows must be actual located observations, not guesses.
        located = locate(queries, state['location_calls'] + 1)
        added = 0
        for row in located or []:
            try:
                start, end = float(row['start']), float(row['end'])
                owners = list(row['event_ids'])
            except (TypeError, ValueError, KeyError):
                continue
            windows = [q for q in queries if set(q['event_ids']).intersection(owners)]
            if not (owners and all(k in events for k in owners) and .05 < end - start <= 15
                    and any(q['start'] - .05 <= start < end <= q['end'] + .05 for q in windows)
                    and str(row.get('action') or '').strip()):
                continue
            shot = dict(row, start=start, end=end, shot_id=f'SHL{len(state["located"]) + 1:05d}')
            state['located'].append(shot)
            catalog.append(shot)
            added += 1
        state['location_calls'] += 1
        persist()
        if added:
            choose_pending()
    missing = [s['scene_id'] for s in scenes if s['scene_id'] not in state['choices']]
    if missing:
        state['pending_scene_ids'] = missing
        persist()
        raise ValueError('SENTENCE_SHOTS_PENDING: Da giu loi recap va ket qua chon hinh; con '
                         + str(len(missing)) + ' cau can hinh: ' + ', '.join(missing[:20]))

    lookup = {r['shot_id']: r for r in catalog}
    uses = {}
    for index, scene in enumerate(scenes):
        for link in state['choices'][scene['scene_id']]:
            uses.setdefault(link['shot_id'], []).append(index)
    source_cursor = -1.0
    for index, scene in enumerate(scenes):
        sources, links = [], []
        for link in state['choices'][scene['scene_id']]:
            shot = lookup[link['shot_id']]
            owners = uses[link['shot_id']]
            position = owners.index(index)
            # Share only observed frames when adjacent sentences select the
            # same shot. Never manufacture ranges in an unobserved gap.
            start = shot['start'] + (shot['end'] - shot['start']) * position / len(owners)
            end = shot['start'] + (shot['end'] - shot['start']) * (position + 1) / len(owners)
            if start < source_cursor - .01 or end - start <= .05:
                raise ValueError('SENTENCE_SHOT_ORDER: ' + scene['scene_id'] + '; can chon lai shot, khong tu day moc nguon.')
            sources.append({'start': timecode(start), 'end': timecode(end), 'event': shot['action']})
            links.append({'visual_id': f'V{len(sources):03d}', 'trigger_text': link['trigger_text']})
            source_cursor = end
        plan = semantic_visual_plan(scene['tts'], sources, links)
        if plan['visual_alignment_errors']:
            raise ValueError('SENTENCE_SHOT_ALIGNMENT: ' + scene['scene_id'])
        scene.update(plan)
        scene['visual_selection_pending'] = False
        scene['matched_shot_ids'] = [l['shot_id'] for l in state['choices'][scene['scene_id']]]
    return result
