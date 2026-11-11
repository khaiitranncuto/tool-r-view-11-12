"""Budget coherent situation paragraphs; split and bind sentences after writing."""
import copy
import hashlib
import json
import re

from narrative_contract import sentences, sentence_texts
from semantic_visuals import visual_catalog, timecode, seconds
from source_grounded_writer import bind_written_scenes, signature

VERSION = 1


def has_direct_dialogue(text):
    """Narrator-only recap does not contain quoted speech or dialogue turns."""
    text = str(text or '')
    return bool(re.search(r'"[^"\n]*\w[^"\n]*"|“[^”\n]*\w[^”\n]*”|«[^»\n]*\w[^»\n]*»|‘[^’\n]*\w[^’\n]*’|(?<!\w)\x27[^\x27\n]*\w[^\x27\n]*\x27(?!\w)', text)
                or re.search(r'(?:^|\n)\s*[—–-]\s+\S', text))


def unsupported_dialogue_fact(fact):
    """Speech content cannot be established by a picture of someone talking."""
    text = str(fact.get('text') or '')
    speech = fact.get('evidence_kind') == 'dialogue' or bool(re.search(
        r'\b(?:giải thích|nói rằng|cho biết rằng|kể về|tiết lộ rằng|thông báo rằng)\b', text, re.I))
    return speech and not str(fact.get('spoken_text') or '').strip()


def rejected_cached_events(rows, audit_cache):
    """A structurally valid repair is not reusable after a factual rejection."""
    rejected = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = hashlib.sha256(json.dumps([row.get('tts'), row.get('observed_facts')],
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        verdict = audit_cache.get(key) or {}
        if verdict.get('faithful') is False:
            rejected.add(row.get('event_id'))
    return rejected


def supported_facts(event):
    """Uncertain/inferred observations cannot reserve spoken sentences."""
    result, seen = [], set()
    for fact in event.get('observed_facts') or []:
        if not isinstance(fact, dict):
            continue
        if unsupported_dialogue_fact(fact):
            continue
        if (fact.get('status', 'confirmed') != 'confirmed'
                or fact.get('evidence_kind', 'visual') not in ('visual', 'dialogue')):
            continue
        text = re.sub(r'\s+', ' ', str(fact.get('text') or '')).strip()
        key = re.sub(r'[^\w\s]', '', text.casefold()).strip()
        if not text or not fact.get('fact_id') or not fact.get('shot_ids') or key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def sentence_slots(event):
    sources = [dict(s, start=s.get('start') or timecode(float(s['start_seconds'])),
                    end=s.get('end') or timecode(float(s['end_seconds'])))
               for s in event.get('visual_sources') or []]
    visuals = visual_catalog(sources)
    candidates = []
    for fact in supported_facts(event):
        owners = [v for v in visuals if set(v['shot_ids']) & set(fact['shot_ids'])]
        if owners:
            visual = min(owners, key=lambda v: v['start_seconds'])
            candidates.append((visual['start_seconds'], fact, visual))
    candidates.sort(key=lambda row: row[0])
    quota = int(event['sentence_quota'])
    if quota < 1 or len(candidates) < quota:
        raise ValueError(f'SENTENCE_PLAN_EVIDENCE_SHORTAGE:{event["event_id"]}:{len(candidates)}/{quota}')
    # Select distinct factual ideas across the situation, including its start
    # and result. Unselected details remain context, never extra filler clauses.
    positions = ([0] if quota == 1 else
                 [round(i * (len(candidates) - 1) / (quota - 1)) for i in range(quota)])
    result = []
    for index, position in enumerate(positions, 1):
        _, fact, visual = candidates[position]
        kind = fact.get('evidence_kind', 'visual')
        if kind == 'dialogue' and not (fact.get('spoken_text') or event.get('spoken_text')):
            raise ValueError('SENTENCE_PLAN_DIALOGUE_MISSING:' + fact['fact_id'])
        result.append(dict(slot_id=f'{event["event_id"]}_C{index:03d}', focus=fact['text'],
            fact_ids=[fact['fact_id']], visual_id=visual['visual_id'], evidence_kind=kind,
            **({'spoken_text': fact.get('spoken_text') or event.get('spoken_text')}
               if kind == 'dialogue' else
               ({'spoken_text': fact['spoken_text']} if fact.get('spoken_text') else {}))))
    return result


def build_sentence_plan(plan):
    result = copy.deepcopy(plan)
    total = 0
    for part in result['events_by_part']:
        for event in part:
            if not supported_facts(event):
                raise ValueError('SENTENCE_PLAN_EVIDENCE_MISSING:' + event['event_id'])
            # Positions only: the writer organizes the COMPLETE situation as a
            # paragraph, then the backend splits and binds the finished prose.
            event['sentence_slots'] = [dict(slot_id=f'{event["event_id"]}_C{i:03d}')
                for i in range(1, int(event['sentence_quota']) + 1)]
            event['paragraph_plan_version'] = 2
            event['evidence_sentence_plan_version'] = VERSION
            total += len(event['sentence_slots'])
    expected = int(result['sentence_contract']['target_count'])
    if total != expected:
        raise ValueError(f'EXACT_SENTENCE_BUDGET_MISMATCH:{total}/{expected}')
    result['sentence_plan_version'] = 2
    return result


def paragraph_card(event):
    source_order = {}
    for source in event.get('visual_sources') or []:
        start = (float(source['start_seconds']) if source.get('start_seconds') is not None
                 else seconds(source.get('start', 0)))
        for sid in source.get('shot_ids') or [source.get('shot_id')]:
            if sid:
                source_order[sid] = min(start, source_order.get(sid, start))
    ordered = sorted(supported_facts(event), key=lambda f: min(
        (source_order.get(sid, float('inf')) for sid in f.get('shot_ids') or []), default=float('inf')))
    facts = [dict(fact_id=f['fact_id'], text=f['text'],
                  source_order=i, evidence_kind=f.get('evidence_kind', 'visual'),
                  **({'actor_ids': f['canonical_actor_ids']} if f.get('canonical_actor_ids') else {}))
             for i, f in enumerate(ordered, 1)]
    context = list(dict.fromkeys(str(f.get('spoken_text') or '').strip()
        for f in supported_facts(event) if str(f.get('spoken_text') or '').strip()))
    if not context and event.get('spoken_text'):
        context = [str(event['spoken_text'])]
    return dict(event_id=event['event_id'], sentence_quota=int(event['sentence_quota']),
        target_words=event.get('tts_target_words', 20 * int(event['sentence_quota'])),
        scene_context=event.get('scene_context') or {},
        narrative_role=event.get('story_role') or event.get('narrative_function') or 'development',
        allowed_names=event.get('allowed_character_names') or [], facts=facts,
        **({'transcript_context_only': context} if context else {}))


def paragraph_prompt(events, context, previous=None, error=None):
    from character_identity import render_identity_context
    people = context.get('characters') or []
    summaries = [render_identity_context(str(s)[:600], people) for s in context.get('all_part_summaries') or []]
    following = [dict(row, event=render_identity_context(row.get('event', ''), people))
                 for row in context.get('next_batch_context') or [] if isinstance(row, dict)]
    return (
        'VIET RECAP THEO CUM SU KIEN, MOI CUM LA MOT DOAN KE LIEN MACH. '
        'CHI VIET LOI NGUOI KE, KHONG CHEP HOI THOAI, KHONG TRICH DAN TRUC TIEP. '
        'Whisper/transcript_context_only CHI GIUP HIEU COT TRUYEN VA NGU CANH; '
        'khong chep cau Whisper vao recap, khong dat loi nhan vat trong ngoac kep, '
        'khong viet dang kich ban doi thoai. Neu can y nghia loi noi de hieu mach '
        'thi dien dat gian tiep bang loi nguoi ke, chi khi nguon ro rang. '
        'Phien am vo nghia, nhieu hoac mau thuan thi bo qua; khong doan lai cau noi. '
        'Doc toan bo facts va loi thoai cua cum truoc khi viet; khong viet cac cau roi rac roi ghep lai. '
        'Moi event_id tra paragraph co DUNG sentence_quota cau hoan chinh. '
        'So cau quyet dinh DO CHI TIET: it cau thi co dong cac dien bien lien quan; '
        'nhieu cau thi ke ro nhung chi tiet that su co trong facts/loi thoai. '
        'Mot fact co the chua nhieu chi tiet de ke bang nhieu cau; khong bat moi fact '
        'bang mot cau va khong tao them su kien de du quota. '
        'Moi cau phai them thong tin khac, khong dien dat lai cung mot y de lap cho trong. '
        'Giu cac mat xich quan trong va thu tu dien bien: nguyen nhan da xac nhan, '
        'hanh dong, ket qua. Khong bia quan he nhan qua, noi tam, nguoi noi hay hanh dong. '
        'scene_context chi de hieu mach, khong tu no chung minh mot tinh tiet moi. '
        'Dau doan noi tiep DA KE; cuoi doan dan toi TIEP THEO khi co co so. '
        'Dung ten/dai tu nhat quan, doi tuyen thi gioi thieu lai chu the. '
        'Khong ep "vi vay", "dung luc do" hay "tuy nhien" khi bang chung khong chung minh. '
        'Hinh chung minh hanh dong va dien bien. Transcript chi la ngu canh, '
        'khong chung minh nhan vat/hinh anh/dong tac dang xuat hien. '
        'evidence_kind=dialogue chi danh dau nguon cho y ke GIAN TIEP da ro nghia; '
        'khong phai cho phep trich loi thoai, suy doan nguoi noi hay noi tam. '
        'Facts co source_order theo thu tu nguon: ke dien bien theo thu tu do, '
        'khong quay lai hanh dong truoc sau khi da ke ket qua/phan ung ve sau. '
        'Khong tu them ket qua chien thang, cam xuc hay quyet dinh roi di khi facts khong co. '
        'Phan ket la uu tien thap: cum narrative_role=ending chi ke ngan gon '
        'ket qua that su duoc facts chung minh, khong tong ket dai, khong rut bai hoc, '
        'khong tao them loi chia tay hay ep co ket cuc khi nguon chua ro. '
        'evidence_kind cua cau phai dung loai fact da dan: khong gan dialogue '
        'cho fact visual chi vi transcript_context_only co loi noi gan canh. '
        'target_words la goi y, khong them cau dem de du tu. '
        'Viet xong paragraph moi liet ke sentence_evidence THEO THU TU CAC CAU, '
        'sentence_evidence phai co DUNG sentence_quota muc: moi CAU mot muc, '
        'khong phai moi fact hay moi loai bang chung mot muc. '
        'Vi du quota=1 thi paragraph co 1 cau va sentence_evidence co 1 muc. '
        'Neu mot cau can ca hinh va loi thoai nhung schema chi cho mot evidence_kind, '
        'hay viet lai cau de noi dung duoc mot loai bang chung ho tro, khong them muc. '
        'moi muc co fact_ids va evidence_kind (visual/dialogue). Cac cau ke chi tiet '
        'khac nhau duoc tham chieu cung fact_id khi fact thuc su ho tro cac chi tiet do. '
        'Khong dua ID hay timestamp vao paragraph; khong tao visual_id/trigger. '
        'Chi tra JSON scenes gom event_id, paragraph, sentence_evidence.\n'
        'PHONG CACH: ' + str(context.get('style') or '') + '\n'
        'YEU CAU: ' + str(context.get('custom_prompt') or '') + '\n'
        'MACH TRUYEN: ' + json.dumps(summaries, ensure_ascii=False) + '\n'
        'DA KE: ' + json.dumps([{'tts': str(r.get('tts') or '')[-600:]}
            for r in (context.get('previous_part_tail') or [])[-3:]], ensure_ascii=False) + '\n'
        'TIEP THEO: ' + json.dumps(following, ensure_ascii=False) + '\n'
        'CAC CUM: ' + json.dumps([paragraph_card(e) for e in events], ensure_ascii=False)
        + ('\nSUA RIENG CUM LOI, GIU SO CAU: ' + str(error) + '\nBAN TRUOC: '
           + json.dumps(previous, ensure_ascii=False) if error else ''))


def bind_paragraph(row, event, characters=None):
    """Split only AFTER writing; bind each claim to existing source evidence."""
    from recap_pipeline import _narration_text_normalizer
    paragraph = row.get('paragraph')
    if not isinstance(paragraph, str) or re.search(r'\[(?:TIME|BEAT|SHOT)=', paragraph, re.I):
        raise ValueError('PARAGRAPH_TEXT_INVALID:' + event['event_id'])
    paragraph = _narration_text_normalizer(characters or [])(paragraph)
    if has_direct_dialogue(paragraph):
        raise ValueError('RECAP_DIRECT_DIALOGUE:' + event['event_id']
                         + ':rewrite as narrator-only recap; do not quote or copy Whisper')
    texts = sentence_texts(paragraph)
    quota = int(event['sentence_quota'])
    if len(texts) != quota or len(sentences(paragraph)) != quota:
        raise ValueError(f'PARAGRAPH_SENTENCE_COUNT:{event["event_id"]}:expected={quota}:actual={len(texts)}')
    keys = [re.sub(r'[^\w\s]', '', t.casefold()).strip() for t in texts]
    if len(set(keys)) != len(keys):
        raise ValueError('PARAGRAPH_REPEATED_SENTENCE:' + event['event_id'])
    evidence = row.get('sentence_evidence')
    if not isinstance(evidence, list) or len(evidence) != quota:
        actual = len(evidence) if isinstance(evidence, list) else 'not_list'
        raise ValueError(f'PARAGRAPH_EVIDENCE_COUNT:{event["event_id"]}:expected={quota}:actual={actual}; '
                         'one evidence item per sentence, not per fact or evidence kind')
    facts = {f['fact_id']: f for f in supported_facts(event)}
    sources = [dict(s, start=s.get('start') or timecode(float(s['start_seconds'])),
                    end=s.get('end') or timecode(float(s['end_seconds'])))
               for s in event.get('visual_sources') or []]
    visuals = visual_catalog(sources)
    units, previous_start = [], -1.0
    for sentence_index, (text, proof) in enumerate(zip(texts, evidence), 1):
        if not isinstance(proof, dict):
            raise ValueError('PARAGRAPH_PROOF_INVALID:' + event['event_id'])
        ids = proof.get('fact_ids')
        if not isinstance(ids, list) or not ids or any(not isinstance(fid, str) or fid not in facts for fid in ids):
            raise ValueError('PARAGRAPH_PROOF_INVALID:' + event['event_id'])
        proof_shots = {sid for fid in ids for sid in facts[fid]['shot_ids']}
        cited = [v for v in visuals if set(v['shot_ids']) & proof_shots]
        kind = proof.get('evidence_kind')
        if kind == 'dialogue' and not any(facts[fid].get('evidence_kind') == 'dialogue'
                and str(facts[fid].get('spoken_text') or '').strip() for fid in ids):
            raise ValueError(f'PARAGRAPH_EVIDENCE_KIND_MISMATCH:{event["event_id"]}:sentence={sentence_index}; '
                f'fact_ids={ids}; dialogue requires a cited dialogue fact with clear source speech; '
                'Whisper is context only. Rewrite unsupported claims from the actual facts, do not relabel them.')
        owners = [v for v in cited if v['start_seconds'] >= previous_start]
        if not owners and kind == 'dialogue' and cited:
            # A supported indirect statement may use a nearby reaction image;
            # it must not force the picture track to replay an earlier action.
            limit = max(v['end_seconds'] for v in cited) + 6
            owners = [v for v in visuals if previous_start <= v['start_seconds'] <= limit]
        if not owners:
            times = [(v['start_seconds'], v['end_seconds']) for v in cited]
            raise ValueError(f'SOURCE_WRITER_ACTION_ORDER_INVALID:{event["event_id"]}:sentence={sentence_index}; '
                f'fact_ids={ids}; previous_source_start={previous_start}; cited_source_windows={times}; '
                'rewrite this paragraph in source_order using supported facts; do not invent an outcome or change evidence labels')
        visual = min(owners, key=lambda v: v['start_seconds'])
        previous_start = visual['start_seconds']
        units.append(dict(text=text, fact_ids=ids, visual_id=visual['visual_id'],
                          evidence_kind=proof.get('evidence_kind')))
    scene = bind_written_scenes({'scenes': [{'event_id': event['event_id'], 'sentences': units}]},
        [dict(event, visual_sources=sources)], characters=characters)[0]
    scene.update(evidence_sentence_plan_version=VERSION, paragraph_plan_version=2,
                 sentence_slots=copy.deepcopy(event['sentence_slots']))
    for index, proof in enumerate(scene['sentence_proofs'], 1):
        proof['slot_id'] = f'{event["event_id"]}_C{index:03d}'
    scene['source_grounded_signature'] = signature(scene)
    return scene


def writing_batches(events, max_sentences=24, max_chars=18000):
    batch, count, size = [], 0, 0
    for event in events:
        slots = event['sentence_slots']
        length = len(json.dumps(paragraph_card(event) if event.get('paragraph_plan_version') == 2 else slots,
                                ensure_ascii=False))
        if batch and (count + len(slots) > max_sentences or size + length > max_chars):
            yield batch
            batch, count, size = [], 0, 0
        batch.append(event)
        count += len(slots)
        size += length
    if batch:
        yield batch


def writing_schema(events):
    if all(e.get('paragraph_plan_version') == 2 for e in events):
        # One shared shape avoids combinatorial provider-side schema branching.
        # Event-specific quotas/ownership remain enforced by bind_paragraph.
        quotas = [int(e['sentence_quota']) for e in events]
        return {'type': 'object', 'properties': {'scenes': {'type': 'array',
            'minItems': len(events), 'maxItems': len(events), 'items': {
                'type': 'object', 'properties': {
                'event_id': {'type': 'string', 'enum': [e['event_id'] for e in events]},
                'paragraph': {'type': 'string'},
                'sentence_evidence': {'type': 'array', 'minItems': min(quotas), 'maxItems': max(quotas),
                    'items': {'type': 'object', 'properties': {
                        'fact_ids': {'type': 'array', 'minItems': 1, 'items': {'type': 'string',
                            'enum': list(dict.fromkeys(f['fact_id'] for e in events for f in supported_facts(e)))}},
                        'evidence_kind': {'type': 'string', 'enum': ['visual', 'dialogue']}},
                        'required': ['fact_ids', 'evidence_kind']}}},
                'required': ['event_id', 'paragraph', 'sentence_evidence']}}}, 'required': ['scenes']}
    return {'type': 'object', 'properties': {'scenes': {'type': 'array',
        'minItems': len(events), 'maxItems': len(events), 'items': {
            'type': 'object', 'properties': {
                'event_id': {'type': 'string', 'enum': [e['event_id'] for e in events]},
                'sentences': {'type': 'array', 'items': {'type': 'object', 'properties': {
                    'slot_id': {'type': 'string', 'enum': [s['slot_id'] for e in events for s in e['sentence_slots']]},
                    'text': {'type': 'string'}}, 'required': ['slot_id', 'text']}}},
            'required': ['event_id', 'sentences']}}}, 'required': ['scenes']}


def writing_prompt(events, context, previous=None, error=None):
    if all(e.get('paragraph_plan_version') == 2 for e in events):
        return paragraph_prompt(events, context, previous, error)
    cards = [dict(event_id=e['event_id'], scene_context=e.get('scene_context') or {},
                  allowed_names=e.get('allowed_character_names') or [],
                  target_words=e.get('tts_target_words', 20 * len(e['sentence_slots'])),
                  sentences=e['sentence_slots']) for e in events]
    return (
        'VIET RECAP TIENG VIET TU KE HOACH CAU DA KHOA. Khong can xem lai video. '
        'Moi slot_id tra DUNG mot cau hoan chinh, ket thuc bang dau cham/hoi/than; '
        'giu nguyen thu tu event va slot. Tong so cau bang so slot, khong them/bot. '
        'target_words la goi y do dai cho ca cum, uu tien cau tu nhien khoang 15-25 tu '
        'khi du bang chung; khong them cau dem hay tinh tiet de keo dai. '
        'Moi cau chi dien dat focus cua slot; khong them hanh dong, nhan vat, loi thoai, '
        'noi tam hay ket qua ngoai bang chung. evidence_kind=dialogue chi chung minh '
        'noi dung duoc noi, khong chung minh hanh dong nhin thay hay nguoi noi. '
        'spoken_text la loi thoai dung vung bang chung, duoc dung de hieu/noi mach '
        'khi noi ro nguyen nhan; khong tu quy nguoi noi hay suy dien quan he nhan qua. '
        'Doc ca cum truoc khi viet: noi cac cau thanh doan ke tu nhien, tranh lap '
        'chu ngu va lap y. Chi dung quan he nguyen nhan/ket qua khi focus chung minh; '
        'khong tu noi hai tuyen bang "dung luc". Khong doc SHOT/fact/slot ID trong loi ke. '
        'Khong viet timestamp, visual_id, fact_ids hay trigger; backend se gan tu ke hoach. '
        'Chi tra JSON scenes, moi scene co event_id va sentences gom slot_id, text.\n'
        'PHONG CACH: ' + str(context.get('style') or '') + '\n'
        'YEU CAU: ' + str(context.get('custom_prompt') or '') + '\n'
        'MACH TRUYEN: ' + json.dumps([str(s)[:600] for s in context.get('all_part_summaries') or []], ensure_ascii=False) + '\n'
        'DA KE: ' + json.dumps([{'tts': str(r.get('tts') or '')[-600:]} for r in (context.get('previous_part_tail') or [])[-3:]], ensure_ascii=False) + '\n'
        'TIEP THEO: ' + json.dumps(context.get('next_batch_context') or [], ensure_ascii=False) + '\n'
        'KE HOACH: ' + json.dumps(cards, ensure_ascii=False)
        + ('\nSUA DUNG LOI: ' + str(error) + '\nBAN TRUOC: ' + json.dumps(previous, ensure_ascii=False)
           if error else ''))


def bind_planned_scenes(payload, events, characters=None):
    rows = payload.get('scenes') if isinstance(payload, dict) else None
    if (not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows)
            or [r.get('event_id') for r in rows] != [e['event_id'] for e in events]):
        raise ValueError('SENTENCE_PLAN_EVENT_IDS_INVALID')
    bound = []
    for row, event in zip(rows, events):
        if event.get('paragraph_plan_version') == 2:
            bound.append(bind_paragraph(row, event, characters))
            continue
        units = row.get('sentences')
        slots = event['sentence_slots']
        if (not isinstance(units, list) or any(not isinstance(u, dict) for u in units)
                or [u.get('slot_id') for u in units] != [s['slot_id'] for s in slots]):
            raise ValueError('SENTENCE_PLAN_SLOT_IDS_INVALID:' + event['event_id'])
        if any(not isinstance(u.get('text'), str) or len(sentences(u['text'])) != 1 for u in units):
            raise ValueError('SENTENCE_PLAN_ONE_SENTENCE_REQUIRED:' + event['event_id'])
        raw = {'scenes': [{'event_id': event['event_id'], 'sentences': [
            dict(text=u['text'], fact_ids=s['fact_ids'], visual_id=s['visual_id'], evidence_kind=s['evidence_kind'])
            for u, s in zip(units, slots)]}]}
        scene = bind_written_scenes(raw, [event], characters=characters)[0]
        scene['evidence_sentence_plan_version'] = VERSION
        scene['sentence_slots'] = copy.deepcopy(slots)
        for proof, slot in zip(scene['sentence_proofs'], slots):
            proof['slot_id'] = slot['slot_id']
        scene['source_grounded_signature'] = signature(scene)
        bound.append(scene)
    return bound


def write_batch(events, context, generate, candidates=None, save=None, characters=None):
    """Keep valid event rows while retrying only missing/invalid events once."""
    from recap_visual_media import parse_writing_response
    ready, error = {}, None
    drafts = list(candidates or [])
    def accept(payload):
        nonlocal error
        rows = payload.get('scenes') if isinstance(payload, dict) else None
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            error = 'SENTENCE_PLAN_SCHEMA_INVALID'
            return
        seen = set()
        lookup = {e['event_id']: e for e in events}
        for row in rows:
            eid = row.get('event_id')
            if not isinstance(eid, str) or eid not in lookup or eid in seen:
                raise ValueError('SENTENCE_PLAN_EVENT_IDS_INVALID')
            seen.add(eid)
        for row in rows:
            eid = row['event_id']
            if eid in ready:
                continue
            try:
                ready[eid] = bind_planned_scenes({'scenes': [row]}, [lookup[eid]], characters)[0]
            except ValueError as exc:
                error = str(exc)
    for draft in reversed(drafts):
        try:
            accept(draft)
        except ValueError as exc:
            error = str(exc)
    for _ in range(2):
        missing = [e for e in events if e['event_id'] not in ready]
        if not missing:
            break
        raw = generate(missing, writing_prompt(missing, context, drafts[-1] if drafts else None, error))
        try:
            payload = parse_writing_response(raw)
            drafts.append(payload)
            drafts = drafts[-2:]
            accept(payload)
        except ValueError as exc:
            error = str(exc)
        if save:
            # A later interrupted repair must not evict valid rows from the
            # first draft merely because only two raw candidates are retained.
            completed = {'scenes': [({'event_id': e['event_id'],
                'paragraph': ready[e['event_id']]['tts'], 'sentence_evidence': [
                    {'fact_ids': p['fact_ids'], 'evidence_kind': p['evidence_kind']}
                    for p in ready[e['event_id']]['sentence_proofs']]}
                if e.get('paragraph_plan_version') == 2 else
                {'event_id': e['event_id'], 'sentences': [
                    {'slot_id': p['slot_id'], 'text': p['text']}
                    for p in ready[e['event_id']]['sentence_proofs']]})
                for e in events if e['event_id'] in ready]}
            save([completed] + drafts[-1:])
    if len(ready) != len(events):
        raise ValueError(error or 'SENTENCE_PLAN_INCOMPLETE')
    return [ready[e['event_id']] for e in events]


def text_audit_prompt(cards):
    return (
        'KIEM CHUNG LOI KE VOI BANG CHUNG DA QUAN SAT, khong viet lai. '
        'Moi scene_id tra faithful=true chi khi TAT CA menh de duoc observed_facts '
        'ho tro; khong chap nhan suy doan nguoi, hanh dong, noi tam, nguyen nhan hay '
        'ket qua. Chi kiem tra dieu cau thuc su ke, khong bat ke het facts. '
        'Nhieu cau duoc cung tham chieu mot fact neu ke cac chi tiet khac nhau '
        'da co trong fact/loi thoai. Neu cac cau chi lap lai cung mot y bang tu khac '
        'de lap quota, tra faithful=false va neu ro phan lap; khong nham voi cac '
        'buoc hanh dong khac nhau cua cung mot su kien. '
        'Loi thoai chi ho tro noi dung duoc noi. Khong goi day la da kiem chung video. '
        'Recap phai la loi nguoi ke: trich hoi thoai truc tiep hoac chep phien am '
        'vo nghia tu Whisper thi faithful=false; yeu cau ke gian tiep tu bang chung ro nghia. '
        'Neu bang chung mo ho va phai xem hinh lai: faithful=false, reason bat dau '
        '"visual_review_required:". Neu cau trai/ them y so voi facts: faithful=false, '
        'reason neu ro menh de sai de sua rieng. Tra JSON '
        '{"checks":[{"scene_id":"S0001","faithful":true,"reason":""}]}.\n'
        + json.dumps(cards, ensure_ascii=False))


AUDIT_SCHEMA = {'type': 'object', 'properties': {'checks': {'type': 'array', 'items': {
    'type': 'object', 'properties': {'scene_id': {'type': 'string'}, 'faithful': {'type': 'boolean'},
        'reason': {'type': 'string'}}, 'required': ['scene_id', 'faithful', 'reason']}}}, 'required': ['checks']}


def audit_planned_payload(payload, text_check, video_check, text_cache=None, video_cache=None,
                         save_text=None, save_video=None, cache_only=False,
                         source_identity=None, source_offset=0.0):
    """Text evidence first; video is reserved for explicit visual uncertainty."""
    from grounded_pipeline import audit_narrations
    checked = audit_narrations(payload, text_check, checkpoint=text_cache,
        save=save_text, max_attempts=0 if cache_only else 2)
    risky = {s['scene_id'] for p in checked['parts'] for s in p.get('scenes') or []
             if any(str(error).startswith('unsupported_narration:visual_review_required:')
                    for error in s.get('visual_alignment_errors') or [])}
    if risky:
        subset = copy.deepcopy(checked)
        for part in subset['parts']:
            part['scenes'] = [s for s in part.get('scenes') or [] if s['scene_id'] in risky]
        reviewed = audit_narrations(subset, video_check, checkpoint=video_cache,
            save=save_video, visual_evidence=True, source_identity=source_identity,
            visual_source_offset=source_offset, max_attempts=0 if cache_only else 1)
        replacements = {s['scene_id']: s for p in reviewed['parts'] for s in p.get('scenes') or []}
        for part in checked['parts']:
            part['scenes'] = [replacements.get(s['scene_id'], s) for s in part.get('scenes') or []]
    return checked
