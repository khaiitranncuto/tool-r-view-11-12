"""One visual writing request; backend owns evidence IDs and edit anchors."""
import copy
import hashlib
import json
import re

from narrative_contract import chapter_prompt, sentences, constrain_repair_sentences
from semantic_visuals import visual_catalog

VERSION = 1


def writing_prompt(events, context, manifest, repair=None):
    narrative = chapter_prompt(events, context).replace('CHUA GHEP HINH', 'DUNG HINH DA CUNG CAP').replace(
        'khong tim hinh,', 'khong tim hinh ngoai tinh huong,')
    # The prose-only schema conflicts with the proof objects required below.
    narrative = narrative.rsplit('\nChi tra JSON ', 1)[0]
    return (narrative
        + '\nXEM HINH DINH KEM truoc khi viet. Mo ta observed_facts co the nhan nham; '
          'chi ke dieu hinh thuc su chung minh hoac loi thoai du tin cay noi ro. '
          'Khong them nguoi vang mat, dong tac, noi tam. Nhan vat duoc nhac den '
          'khong nhat thiet xuat hien; phan biet voi nguoi dang lam hanh dong. '
          'Giu thu tu dien bien. Neu fact nhan sai thi bo y sai, viet cac y con '
          'duoc chung minh, van dung quota va khong lap y cho du cau. '
          'Moi cau chi ro fact_ids va mot visual_id minh hoa CHINH xac; '
          'evidence_kind=dialogue khi chi duoc loi thoai ho tro (dung canh noi/phan ung '
          'cung tinh huong, khong doan nguoi noi). Khong tao timestamp hay trigger_text. '
          'KHONG dung schema sentences chuoi o tren. Tra JSON '
          '{"scenes":[{"event_id":"E0001","sentences":[{"text":"Cau ke.",'
          '"fact_ids":["F1"],"visual_id":"V001","evidence_kind":"visual"}]}]}.'
        + '\nBANG CHUNG BAT BIEN: ' + json.dumps([
            {'event_id': e['event_id'], 'facts': e.get('observed_facts') or [],
             'visuals': visual_catalog(e.get('visual_sources') or []),
             'visible_character_names': e.get('visible_character_names') or [],
             'mentioned_character_names': e.get('mentioned_character_names') or []}
            for e in events], ensure_ascii=False)
        + '\nMANIFEST: ' + json.dumps(manifest, ensure_ascii=False)
        + ('\nCHI SUA NOI DUNG BI BAO SAI: ' + json.dumps(repair, ensure_ascii=False) if repair else ''))


def signature(scene):
    # Coordinates can become part-local/global during serialization. Stable
    # source IDs, fact bindings and the exact spoken text must never change.
    data = [VERSION, scene.get('tts'), scene.get('sentence_proofs'),
            scene.get('observed_facts'),
            [(s.get('shot_ids') or [s.get('shot_id')], s.get('fact_ids'))
             for s in scene.get('locked_visual_shots') or []]]
    if scene.get('evidence_sentence_plan_version'):
        data += [scene['evidence_sentence_plan_version'], scene.get('sentence_slots'), scene.get('visual_links')]
    if scene.get('narration_unit_version'):
        data += ['narration_units', scene['narration_unit_version']]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def is_source_grounded(scene):
    return (scene.get('source_grounded_writer_version') == VERSION
            and bool(scene.get('source_grounded_signature'))
            and scene['source_grounded_signature'] == signature(scene))


def bind_written_scenes(payload, events, characters=None):
    """Validate claims before freezing prose; derive literal cues locally."""
    from recap_pipeline import _narration_text_normalizer
    normalize = _narration_text_normalizer(characters or [])
    if not isinstance(payload, dict):
        raise ValueError('SOURCE_WRITER_SCHEMA_INVALID: payload must be an object')
    rows = payload.get('scenes') or []
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError('SOURCE_WRITER_SCHEMA_INVALID: scenes must be a list of objects')
    if [r.get('event_id') for r in rows] != [e['event_id'] for e in events]:
        raise ValueError('SOURCE_WRITER_IDS_INVALID')
    result = []
    for raw, event in zip(rows, events):
        unit_mode = bool(event.get('narration_unit_version'))
        units = raw.get('sentences') or []
        if not isinstance(units, list) or len(units) < event['sentence_quota']:
            raise ValueError('CHAPTER_SENTENCE_QUOTA_MISMATCH:' + event['event_id'])
        facts = {f['fact_id']: f for f in event.get('observed_facts') or []}
        visuals = {s['visual_id']: s for s in visual_catalog(event.get('visual_sources') or [])}
        texts, proofs = [], []
        previous_start = -1.0
        for index, unit in enumerate(units, 1):
            if not isinstance(unit, dict):
                raise ValueError(
                    f'SOURCE_WRITER_SCHEMA_INVALID:{event["event_id"]}:sentences[{index}] '
                    'must be an object with text, fact_ids, visual_id, evidence_kind')
            text = normalize(str(unit.get('text') or ''))
            ids = unit.get('fact_ids') or []
            visual_id = unit.get('visual_id')
            if (not isinstance(unit.get('text'), str) or not isinstance(ids, list)
                    or any(not isinstance(fid, str) for fid in ids)
                    or not isinstance(visual_id, str)):
                raise ValueError(f'SOURCE_WRITER_SCHEMA_INVALID:{event["event_id"]}:sentences[{index}]')
            visual = visuals.get(visual_id)
            kind = unit.get('evidence_kind', 'visual')
            if not re.search(r'\w', text) or (not unit_mode and (
                    len(sentences(text)) != 1 or not re.search(r'[.!?]["”’\)]*$', text))):
                raise ValueError('SOURCE_WRITER_SENTENCE_INVALID:' + event['event_id'])
            if not ids or any(fid not in facts for fid in ids) or visual is None:
                raise ValueError('SOURCE_WRITER_PROOF_INVALID:' + event['event_id'])
            if kind not in ('visual', 'dialogue'):
                raise ValueError('SOURCE_WRITER_EVIDENCE_KIND_INVALID')
            if kind == 'dialogue' and not (event.get('spoken_text') or any(facts[fid].get('spoken_text') for fid in ids)):
                raise ValueError('SOURCE_WRITER_DIALOGUE_MISSING:' + event['event_id'])
            proof_shots = {sid for fid in ids for sid in facts[fid].get('shot_ids') or []}
            if kind == 'visual' and not proof_shots.intersection(visual['shot_ids']):
                raise ValueError('SOURCE_WRITER_UNRELATED_VISUAL:' + event['event_id'])
            if visual['start_seconds'] < previous_start:
                raise ValueError('SOURCE_WRITER_ACTION_ORDER_INVALID:' + event['event_id'])
            previous_start = visual['start_seconds']
            texts.append(text)
            proofs.append(dict(text=text, fact_ids=list(ids), visual_id=unit['visual_id'], evidence_kind=kind))
            if unit_mode:
                vids = unit.get('visual_ids') or [visual_id]
                if any(v not in visuals for v in vids):
                    raise ValueError('NARRATION_UNIT_VISUAL_MISSING')
                for vid in vids:
                    if not set(visuals[vid]['shot_ids']) & proof_shots:
                        raise ValueError('NARRATION_UNIT_VISUAL_UNRELATED')
                previous_start = visuals[vids[-1]]['start_seconds']
                proofs[-1].update(unit_id=unit['unit_id'], visual_ids=vids, shot_ids=unit['shot_ids'])
        if unit_mode and len(units) != event['sentence_quota']:
            raise ValueError('NARRATION_UNIT_QUOTA_MISMATCH')
        tts = ' '.join(texts) if unit_mode else constrain_repair_sentences(' '.join(texts), event['sentence_quota'])
        link_proofs, link_texts = [], []
        for proof in proofs:
            ids = proof.get('visual_ids') or [proof['visual_id']]
            from narrative_contract import sentence_texts
            fragments = sentence_texts(proof['text'])
            if len(fragments) >= len(ids):
                for i, vid in enumerate(ids):
                    a, b = len(fragments) * i // len(ids), len(fragments) * (i + 1) // len(ids)
                    link_texts.append(' '.join(fragments[a:b]))
                    link_proofs.append(dict(proof, visual_id=vid))
                continue
            tokens = list(re.finditer(r'\S+', proof['text']))
            if len(tokens) < len(ids):
                raise ValueError('NARRATION_UNIT_TOO_SHORT_FOR_VISUALS')
            for i, vid in enumerate(ids):
                a, b = len(tokens) * i // len(ids), len(tokens) * (i + 1) // len(ids)
                link_texts.append(proof['text'][tokens[a].start():tokens[b - 1].end()])
                link_proofs.append(dict(proof, visual_id=vid))
        links = []
        for proof_index, (proof, text) in enumerate(zip(link_proofs, link_texts)):
            visual_id = proof['visual_id']
            if not links:
                links.append({'visual_id': visual_id, 'trigger_text': ''})
            elif visual_id != links[-1]['visual_id']:
                # Sentence boundaries are known before TTS. Choose a unique
                # literal anchor instead of asking the model to retype it.
                tokens = list(re.finditer(r'\S+', text))
                trigger = None
                for length in range(2, min(12, len(tokens)) + 1):
                    literal = text[:tokens[length - 1].end()]
                    # A sentence-ending full stop can distinguish a short
                    # clause from the same prefix in a later longer clause.
                    for candidate in dict.fromkeys((literal.rstrip('.,!?;:'), literal)):
                        if len(list(re.finditer(r'(?<!\w)' + re.escape(candidate) + r'(?!\w)', tts, re.I))) == 1:
                            trigger = candidate
                            break
                    if trigger:
                        break
                if trigger is None and tts == ' '.join(link_texts):
                    # A short sentence can be a substring of another sentence
                    # ("Nobita enters" / "Mother sees Nobita enters"). Include
                    # lookahead words, but keep the cut at this sentence START.
                    offset = sum(len(t) + 1 for t in link_texts[:proof_index])
                    tail = tts[offset:]
                    tail_tokens = list(re.finditer(r'\S+', tail))
                    for length in range(2, min(12, len(tail_tokens)) + 1):
                        candidate = tail[:tail_tokens[length - 1].end()]
                        if len(list(re.finditer(r'(?<!\w)' + re.escape(candidate) + r'(?!\w)', tts, re.I))) == 1:
                            trigger = candidate
                            break
                if trigger:
                    links.append({'visual_id': visual_id, 'trigger_text': trigger})
                else:
                    raise ValueError('SOURCE_WRITER_AMBIGUOUS_ANCHOR:' + event['event_id'])
        from recap_pipeline import build_backend_visual_plan
        plan = build_backend_visual_plan(tts, event.get('visual_sources') or [], links)
        if plan.get('visual_alignment_errors'):
            raise ValueError('SOURCE_WRITER_LINK_INVALID:' + str(plan['visual_alignment_errors']))
        row = dict(event_id=event['event_id'], tts=tts, visual_links=links,
                   sentence_proofs=proofs, source_grounded_writer_version=VERSION,
                   observed_facts=copy.deepcopy(event.get('observed_facts') or []))
        row.update(plan)
        if unit_mode:
            row['narration_unit_version'] = 1
        row['source_grounded_signature'] = signature(row)
        result.append(row)
    return result


def write_validated_scenes(events, candidates, generate, save=None, characters=None):
    """Reuse valid drafts or repair one batch once; never invent proof for strings."""
    from recap_visual_media import parse_writing_response

    error = None
    for candidate in reversed(candidates):
        try:
            return bind_written_scenes(candidate, events, characters=characters)
        except ValueError as exc:
            error = exc
    for _ in range(2):
        raw = generate(candidates[-1] if candidates else None, str(error) if error else None)
        try:
            candidate = parse_writing_response(raw)
            candidates.append(candidate)
            candidates[:] = candidates[-2:]
            if save:
                save(candidates)
            return bind_written_scenes(candidate, events, characters=characters)
        except ValueError as exc:
            error = exc
    raise error
