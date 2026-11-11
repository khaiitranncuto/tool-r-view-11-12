"""Situation-sized narration and a film-wide, append-only identity registry."""
import copy
import math
import re


def normalize_non_story_ranges(rows, duration):
    result = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            start, end = float(row['start_seconds']), float(row['end_seconds'])
            confidence = float(row.get('confidence', 0))
        except (KeyError, TypeError, ValueError):
            continue
        if (all(math.isfinite(v) for v in (start, end, confidence))
                and 0 <= start < end <= duration and confidence >= .9
                and row.get('kind') in {'logo', 'opening', 'credits', 'outro', 'ending_montage'}):
            result.append(dict(row, start_seconds=start, end_seconds=end))
    return sorted(result, key=lambda r: r['start_seconds'])


def character_registry(outline, chunks=(), transcript_segments=None):
    from character_identity import profile, merge_profile, canonical_id, fold
    strict = transcript_segments is not None
    shots = {sid for chunk in chunks for event in chunk.get('events') or []
             for fact in event.get('grounded_facts') or [] for sid in fact.get('shot_ids') or []}
    rows = {}
    for character in (outline or {}).get('characters') or []:
        if not isinstance(character, dict):
            continue
        cid = canonical_id(character.get('id'), rows)
        if cid:
            row = profile(dict(character, id=cid), transcript_segments, None, strict)
            if cid in rows:
                if (rows[cid].get('proposed_name') and row.get('proposed_name')
                        and fold(rows[cid]['proposed_name']) != fold(row['proposed_name'])):
                    rows[cid]['identity_conflict'] = 'duplicate_global_id'
                    row['name_status'] = 'candidate'
                rows[cid] = merge_profile(rows[cid], row)
            else:
                rows[cid] = row
    for chunk in chunks:
        for character in chunk.get('characters') or []:
            cid = canonical_id(character.get('canonical_id'), rows)
            if not cid:
                continue
            row = profile(dict(character, id=cid, canonical_id=cid), transcript_segments, shots, strict)
            if cid in rows:
                rows[cid] = merge_profile(rows[cid], row)
            else:
                rows[cid] = row
    used = {}
    for cid, row in rows.items():
        key = fold(row['neutral_alias'])
        if key in used and used[key] != cid:
            row['neutral_alias'] += f' thứ {int(cid[1:])}'
        used[fold(row['neutral_alias'])] = cid
        row['display_name'] = row.get('name') or row['neutral_alias']
    return list(rows.values())


def _label(value):
    return ' '.join(re.findall(r"[^\W_]+", str(value).casefold()))


def _feature_matches(feature, reference):
    left, right = _label(feature), _label(reference)
    return bool(left and right and (left == right or (' ' + left + ' ') in (' ' + right + ' ')))


def bind_character_registry(payload, registry, identity_shot_ids=None, transcript_segments=None):
    """Resolve people by stable IDs; render only explicit tokens/name spans."""
    from character_identity import (VERSION, canonical_id, confidence, clean_list,
        profile, merge_profile, verify_name_evidence, render_ids, safe_name_mentions)
    result = copy.deepcopy(payload)
    strict = transcript_segments is not None
    known = {c['id']: c for c in character_registry({'characters': registry},
        transcript_segments=transcript_segments)}
    valid_shots = (set(identity_shot_ids) if identity_shot_ids is not None else
        {sid for e in result.get('events') or [] for f in e.get('grounded_facts') or []
         for sid in f.get('shot_ids') or []})
    next_id = max([int(k[1:]) for k in known] or [0]) + 1
    local_labels, local_canonical, legacy_names = {}, {}, {}
    def feature_score(character, features):
        stable = clean_list(character.get('stable_features'))
        references = stable + clean_list(character.get('face_features')) + clean_list(character.get('body_features'))
        references += clean_list(character.get('distinctive_features'))
        references += clean_list(character.get('appearance_variants'))
        matched_refs = set()
        for feature in features:
            owners = [r for r in references if _feature_matches(feature, r)]
            if owners:
                matched_refs.add(_label(owners[0]))
            elif _feature_matches(feature, character.get('visual_signature', '')):
                matched_refs.add('signature:' + _label(feature))
        stable_hit = any(_label(r) in matched_refs for r in stable)
        return len(matched_refs) if len(matched_refs) >= 2 and (stable_hit or not stable) else 0
    for c in result.get('characters') or []:
        if any(r.get('name_source') == 'gemini' for r in known.values()):
            c['name_source'] = 'gemini'
        local_id = str(c.get('id') or '').upper()
        proposed = str(c.get('name') or '').strip()
        evidence = c.get('identity_evidence') or {}
        evidence = evidence if isinstance(evidence, dict) else {}
        features = clean_list(evidence.get('matched_features'))
        features = [f for i, f in enumerate(features)
                    if not any(_feature_matches(f, earlier) for earlier in features[:i])]
        shot_ids = [sid for sid in clean_list(evidence.get('shot_ids')) if sid in valid_shots]
        observed = bool(shot_ids)
        score = confidence(c.get('match_confidence'))
        name_proved = verify_name_evidence(proposed, c.get('name_evidence'), transcript_segments, valid_shots)
        if not strict:
            name_proved = name_proved or bool(proposed and isinstance(c.get('name_evidence'), str) and c['name_evidence'].strip())
        requested = canonical_id(c.get('canonical_id') or c.get('id'), known)
        cid = requested if requested in known else ''
        named = [key for key, r in known.items() if proposed and _label(proposed) in
            {_label(n) for n in [r.get('name', ''), r.get('proposed_name', '')]
                + list(r.get('aliases') or []) + list(r.get('name_alias_candidates') or []) if n}]
        # Clip-local P IDs can collide with IDs allocated in earlier clips.
        if len(named) == 1:
            cid = named[0]
        visual_candidates = [key for key, r in known.items() if feature_score(r, features)]
        if not cid and len(named) == 1:
            cid = named[0]
        if not cid and len(visual_candidates) == 1:
            cid = visual_candidates[0]
        target = known.get(cid)
        conflict, ambiguous = False, False
        if target:
            names = {_label(n) for n in [target.get('name', ''), target.get('proposed_name', '')]
                     + list(target.get('aliases') or []) if n}
            conflict = bool(proposed and names and _label(proposed) not in names)
            target_score = feature_score(target, features)
            ambiguous = bool(target_score and any(feature_score(r, features) >= target_score
                for key, r in known.items() if key != cid) and not name_proved)
            gemini_named = c.get('name_source') == 'gemini' and len(named) == 1 and named[0] == cid
            accepted = observed and score >= .9 and bool(target_score or gemini_named) and not conflict and not ambiguous
            if conflict:
                # Never render a different person using the colliding ID's name.
                cid = ''
                target = profile(dict(c, id='P00', canonical_id='P00', confidence=score,
                    neutral_alias=c.get('alias') or 'người chưa xác định'),
                    transcript_segments, valid_shots, strict)
        else:
            accepted = observed and score >= .9 and len(features) >= 2
            if accepted:
                cid = f'P{next_id:02d}'
                next_id += 1
                target = profile(dict(c, id=cid, canonical_id=cid, confidence=score,
                    stable_features=features), transcript_segments, valid_shots, strict)
                if any(_label(target['neutral_alias']) == _label(r.get('neutral_alias', '')) for r in known.values()):
                    target['neutral_alias'] += f' thứ {int(cid[1:])}'
                    target['display_name'] = target.get('name') or target['neutral_alias']
            else:
                cid = ''
                target = profile(dict(c, id='P00', canonical_id='P00', name='', proposed_name=proposed,
                    neutral_alias=c.get('alias') or 'người chưa xác định'), transcript_segments, valid_shots, strict)
        c.update(canonical_id=cid, proposed_name=proposed,
                 identity_match_status='matched' if accepted else 'pending',
                 current_view_status='matched' if accepted else 'pending',
                 identity_status='confirmed' if accepted else 'unknown',
                 identity_confidence=score if accepted else 0.0,
                 character_registry_version=VERSION)
        c['name_candidates'] = [{'name': proposed, 'evidence': copy.deepcopy(c.get('name_evidence')),
            'status': 'confirmed' if name_proved else 'candidate'}] if proposed else []
        if conflict:
            c['identity_conflict'] = 'conflicting_name'
        gemini_name = c.get('name_source') == 'gemini' and bool(proposed) and score >= .9
        name_proved = name_proved or gemini_name
        if accepted and not target.get('name') and name_proved:
            target.update(name=proposed, name_status='confirmed', name_verified=not gemini_name and strict,
                          name_source=c.get('name_source', target.get('name_source', '')),
                          name_evidence=copy.deepcopy(c.get('name_evidence')), display_name=proposed)
        if accepted:
            observed_profile = dict(c, reference_shot_ids=shot_ids)
            observed_profile['appearance_variants'] = clean_list(c.get('appearance_variants'))
            if c.get('visual_signature') and c['visual_signature'] != target.get('visual_signature'):
                observed_profile['appearance_variants'].append(c['visual_signature'])
            target = merge_profile(target, observed_profile)
        else:
            target = merge_profile(target, c)
        c['name'] = target.get('name') or ''
        c['name_status'] = target.get('name_status', 'generated')
        c['name_verified'] = target.get('name_verified', False)
        c['neutral_alias'] = target['neutral_alias']
        c['alias'] = target['neutral_alias']
        c['display_name'] = (c['name'] if accepted or (conflict and gemini_name) else '') or c['neutral_alias']
        c['name_evidence'] = copy.deepcopy(target.get('name_evidence') or c.get('name_evidence'))
        for field in ('stable_features', 'face_features', 'body_features', 'distinctive_features',
                      'appearance_variants', 'reference_shot_ids', 'appearances',
                      'verified_reference_seconds', 'reference_frames', 'name_candidates'):
            c[field] = copy.deepcopy(target.get(field) or [])
        if not accepted:
            c['identity_conflict'] = ('conflicting_name' if conflict else
                'ambiguous_visual_match' if ambiguous else 'insufficient_current_view')
        else:
            c.pop('identity_conflict', None)
        if cid:
            known[cid] = dict(copy.deepcopy(target), id=cid)
        local_labels[local_id] = c['display_name']
        local_canonical[local_id] = cid if accepted else ''
        if proposed:
            legacy_names[local_id] = (proposed, c['display_name'])
    for event in result.get('events') or []:
        actors = [str(i).upper() for i in event.get('characters') or []]
        replacements = dict(legacy_names[i] for i in actors if i in legacy_names)
        event.setdefault('raw_event', event.get('event', ''))
        modern = any('actor_ids' in f for f in event.get('grounded_facts') or [])
        event['identity_tokens_rendered'] = modern
        event['event'] = render_ids(event['raw_event'] if modern else
            safe_name_mentions(event['raw_event'], replacements), local_labels, allow_bare=not modern)
        event['character_refs'] = [dict(local_id=i, character_id=local_canonical.get(i) or None,
            display_name=local_labels.get(i, 'người chưa xác định')) for i in actors]
        for fact in event.get('grounded_facts') or []:
            fact.setdefault('raw_text', fact.get('text', ''))
            modern_fact = 'actor_ids' in fact
            fact['identity_tokens_rendered'] = modern_fact
            fact['text'] = render_ids(fact['raw_text'] if modern_fact else
                safe_name_mentions(fact['raw_text'], replacements), local_labels, allow_bare=not modern_fact)
            fact_actors = [str(i).upper() for i in fact.get('actor_ids') or []]
            fact['canonical_actor_ids'] = [local_canonical[i] for i in fact_actors if local_canonical.get(i)]
    result['character_registry_version'] = VERSION
    return result


def registry_audit_payload(registry, chunks):
    canonical = [dict(copy.deepcopy(c), id=c['id']) for c in registry]
    links = []
    for chunk in chunks:
        for c in chunk.get('characters') or []:
            cid = c.get('canonical_id')
            if (cid in {r['id'] for r in canonical}
                    and c.get('identity_match_status') == 'matched'):
                links.append({'local_id': c['id'], 'canonical_id': cid,
                              'confidence': 1.0, 'basis': 'direct_transition',
                              'reason': 'Persistent visual identity registry; name remains registry-owned.'})
    return {'canonical_characters': canonical, 'identity_links': links}


def build_situation_plan(chunks, outline, duration, target_words, *, single_timeline=False):
    """One real situation owns many facts/shots; duration never creates sentence slots."""
    events = []
    used_facts = set()
    beats = (outline or {}).get('story_beats') or []
    # Resolve evidence in source order FIRST. A character's scene can return
    # after another storyline; never keep both appearances in one crossing range.
    atomic = []
    # The overview can be wrong by minutes. Exclusion requires shot IDs read
    # during the existing detailed pass; explicit story evidence wins conflicts.
    blocked = [s for c in chunks for s in c.get('confirmed_non_story_shots') or []]
    # Lookahead belongs to the next clip's core interval. Pick one observer
    # before grouping; clip-local situation IDs cannot identify cross-clip
    # duplicates, and both views must never reserve the same source frames.
    claims = {}
    for ci, chunk in enumerate(chunks):
        for source in chunk.get('events') or []:
            for shot in source.get('visual_sources') or []:
                if shot.get('shot_id'):
                    claims.setdefault(shot['shot_id'], []).append((ci, shot))
    shot_owner = {}
    for sid, views in claims.items():
        center = (min(float(s['start_seconds']) for _, s in views)
                  + max(float(s['end_seconds']) for _, s in views)) / 2
        def ownership_rank(ci):
            chunk = chunks[ci]
            start = float(chunk.get('start_seconds', 0))
            end = float(chunk.get('primary_end_seconds', chunk.get('end_seconds', duration)))
            return (not start <= center < end, max(start - center, center - end, 0), ci)
        shot_owner[sid] = min({ci for ci, _ in views}, key=ownership_rank)
    # A source window cannot prove two incompatible situations. Keep these
    # claims for diagnosis instead of joining their prose or guessing an owner.
    ambiguous = set()
    for ci, chunk in enumerate(chunks):
        owners = {}
        for source in chunk.get('events') or []:
            context = source.get('scene_context') or {}
            owner = (str(source.get('situation_id') or source.get('evidence_id') or id(source)),
                     tuple(_label(context.get(k, '')) for k in ('location', 'time_context', 'goal')))
            for fact in source.get('grounded_facts') or []:
                for sid in fact.get('shot_ids') or []:
                    if sid in owners and owners[sid] != owner:
                        ambiguous.add((ci, sid))
                    owners[sid] = owner
    for chunk_index, chunk in enumerate(chunks):
        for source in chunk.get('events') or []:
            if source.get('content_kind', 'story') != 'story':
                continue
            sources = {s['shot_id']: s for s in source.get('visual_sources') or [] if s.get('shot_id')}
            for fact in source.get('grounded_facts') or []:
                if fact.get('fact_id') in used_facts:
                    continue
                from evidence_writer import unsupported_dialogue_fact
                if unsupported_dialogue_fact(fact):
                    continue
                candidates = [sources[sid] for sid in fact.get('shot_ids') or []
                              if sid in sources and shot_owner.get(sid) == chunk_index
                              and (chunk_index, sid) not in ambiguous]
                if not candidates:
                    continue
                primary = min(candidates, key=lambda s: float(s['start_seconds']))
                used_facts.add(fact['fact_id'])
                supported_fact = copy.deepcopy(fact)
                supported_fact['shot_ids'] = [s['shot_id'] for s in candidates]
                atomic.append((float(primary['start_seconds']), float(primary['end_seconds']),
                               primary['shot_id'], source, supported_fact, primary, chunk_index))
    atomic.sort(key=lambda row: (row[0], row[1]))
    for start, end, shot_id, source, fact, primary, chunk_index in atomic:
        previous = events[-1] if events else None
        # situation IDs are clip-local. Never join separate owners solely
        # because another clip reused SIT001.
        owner = str(source.get('evidence_id') or source.get('event_id') or id(source))
        context = source.get('scene_context') or {}
        key = (chunk_index, str(source.get('situation_id') or owner),
               tuple(_label(context.get(k, '')) for k in ('location', 'time_context', 'goal')))
        same_shot = bool(previous and shot_id == previous['_last_shot'])
        same_run = bool(previous and previous['_situation_key'] == key
                        and (not previous.get('characters') or not source.get('characters')
                             or set(previous['characters']) & set(source['characters']))
                        and start - previous['end_seconds'] <= 30
                        and end - previous['start_seconds'] <= 180)
        if previous and same_run:
            previous['planned_fact_ids'].append(fact['fact_id'])
            previous['observed_facts'].append(copy.deepcopy(fact))
            if not same_shot:
                previous['visual_sources'].append(copy.deepcopy(primary))
            previous['end_seconds'] = max(previous['end_seconds'], end)
            previous['_last_shot'] = shot_id
            previous['characters'] = list(dict.fromkeys(list(previous.get('characters') or [])
                                                        + list(source.get('characters') or [])))
            previous['event'] += ' ' + fact['text']
            previous['narration_focus'] = previous['event']
            continue
        row = copy.deepcopy(source)
        row.update(planned_fact_ids=[fact['fact_id']], observed_facts=[copy.deepcopy(fact)],
                   visual_sources=[copy.deepcopy(primary)],
                   start_seconds=start, end_seconds=end, event=fact['text'],
                   narration_focus=fact['text'], situation_narration=True, source_link_contract_version=1,
                   tts_min_words=6, tts_max_words=120, tts_target_words=45,
                   _situation_key=key, _last_shot=shot_id)
        beat = next((b for b in beats if float(b.get('start_seconds', 0)) <= start
                     < float(b.get('end_seconds', 0))), {})
        row['story_beat_id'] = beat.get('beat_id', f'GB{len(events) + 1:03d}')
        row['story_beat_summary'] = beat.get('summary', '')
        events.append(row)
    for event in events:
        event.pop('_situation_key', None)
        event.pop('_last_shot', None)
        # A split situation must not inherit dialogue from its owner's other
        # appearances on the opposite side of an intervening storyline.
        transcript = []
        for fact in event['observed_facts']:
            for line in fact.get('transcript_evidence') or []:
                if line not in transcript:
                    transcript.append(copy.deepcopy(line))
        event['transcript_evidence'] = transcript
        event['spoken_text'] = ' '.join(line['text'] for line in transcript)
    events.sort(key=lambda e: e['start_seconds'])
    for i, event in enumerate(events, 1):
        event.update(event_id=f'E{i:04d}', scene_id=f'S{i:04d}')
    if len(events) < (1 if single_timeline else 3):
        raise ValueError('SITUATION_PLAN_INSUFFICIENT_EVIDENCE')
    parts = [events] if single_timeline else [events[len(events) * i // 3:len(events) * (i + 1) // 3] for i in range(3)]
    boundaries = [0.0] + [(parts[i][-1]['end_seconds'] + parts[i + 1][0]['start_seconds']) / 2
                          for i in range(len(parts) - 1)] + [float(duration)]
    counts = [len(p) for p in parts]
    return {'events_by_part': parts, 'boundaries': boundaries,
            'characters': character_registry(outline, chunks),
            'summaries': [' '.join(dict.fromkeys(e['story_beat_summary'] for e in p)) for p in parts],
            'story_beats_by_part': [[] for _ in parts], 'situation_plan_version': 4,
            'confirmed_non_story_shots': blocked,
            'ambiguous_source_shots': [{'chunk_index': ci, 'shot_id': sid} for ci, sid in sorted(ambiguous)],
            'scene_contract': {'scene_count': len(events), 'part_scene_counts': counts,
                               'target_words': target_words,
                               'part_word_targets': [round(target_words * n / len(events)) for n in counts],
                               'source_event_count': len(events), 'covered_source_event_count': len(events)}}
