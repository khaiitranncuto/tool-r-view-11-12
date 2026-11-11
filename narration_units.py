"""Backend-owned narration units: identity/count independent of punctuation."""
import copy
import json

from semantic_visuals import visual_catalog, timecode


def plan_narration_units(plan):
    from evidence_writer import supported_facts
    result = copy.deepcopy(plan)
    for group in result['events_by_part']:
        for event in group:
            facts = {f['fact_id']: f for f in supported_facts(event)}
            sources = [dict(s, start=s.get('start') or timecode(float(s['start_seconds'])),
                end=s.get('end') or timecode(float(s['end_seconds']))) for s in event['visual_sources']]
            visuals = sorted(visual_catalog(sources), key=lambda v: v['start_seconds'])
            if not visuals:
                raise ValueError('NARRATION_UNIT_VISUALS_MISSING:' + event['event_id'])
            slots, quota = [], int(event['sentence_quota'])
            for i in range(quota):
                a = len(visuals) * i // quota
                b = len(visuals) * (i + 1) // quota
                chosen = visuals[a:max(a + 1, b)]
                ids = list(dict.fromkeys(fid for v in chosen for fid in v['fact_ids'] if fid in facts))
                if not ids:
                    raise ValueError('NARRATION_UNIT_FACTS_MISSING:' + event['event_id'])
                slots.append(dict(unit_id=f'{event["event_id"]}_U{i + 1:03d}',
                    slot_id=f'{event["event_id"]}_C{i + 1:03d}', fact_ids=ids,
                    shot_ids=list(dict.fromkeys(s for v in chosen for s in v['shot_ids'])),
                    visual_ids=[v['visual_id'] for v in chosen],
                    focus=' '.join(facts[fid]['text'] for fid in ids)))
            event.update(sentence_slots=slots, narration_unit_version=1, visual_sources=sources)
    result['narration_unit_version'] = 1
    return result


def unit_schema(events):
    return {'type': 'object', 'properties': {'sentences': {'type': 'array', 'items': {
        'type': 'object', 'properties': {'event_id': {'type': 'string'},
        'unit_id': {'type': 'string'}, 'text': {'type': 'string'}},
        'required': ['event_id', 'unit_id', 'text']}}}, 'required': ['sentences']}


def unit_prompt(events, context, previous=None, error=None):
    from evidence_writer import VIETNAMESE_WRITER_RULE, paragraph_card
    from character_identity import render_identity_context
    people = context.get('characters') or []
    cards = []
    for event in events:
        card = paragraph_card(event)
        cards.append(dict(event_id=event['event_id'],
            role=card['narrative_role'], allowed_names=card['allowed_names'],
            scene_context=card['scene_context'],
            facts=[dict(f, text=render_identity_context(f['text'], people)) for f in card['facts']],
            units=[dict(unit_id=u['unit_id'], fact_ids=u['fact_ids'])
                   for u in event['sentence_slots']],
            **({'transcript_context_only': card['transcript_context_only']}
               if card.get('transcript_context_only') else {})))
    packet = dict(style=context.get('style') or '', requirements=context.get('custom_prompt') or '',
        story=[render_identity_context(str(s)[:600], people)
               for s in context.get('all_part_summaries') or []],
        before=[str(s.get('tts') or '')[-600:]
                for s in (context.get('previous_part_tail') or [])[-2:]],
        after=[render_identity_context(str(s.get('event') or '')[:600], people)
               for s in (context.get('next_batch_context') or [])[:2]], events=cards)
    if error:
        wanted = {e['event_id'] for e in events}
        packet['repair'] = dict(error=str(error), previous=[dict(event_id=r['event_id'],
            **({'units': [dict(unit_id=u.get('unit_id'), text=u.get('text'))
                         for u in r['narration_units']]} if r.get('narration_units') else
               {'text': r.get('paragraph') or r.get('tts') or ''}))
            for r in (previous or {}).get('scenes') or [] if r.get('event_id') in wanted])
    return (VIETNAMESE_WRITER_RULE +
        'DON VI LOI KE CO ID, KHONG DEM BANG DAU CHAM/PHAY. '
        'Đọc toàn cụm, viết thành mạch recap liền nhau rồi trả từng unit theo thứ tự. '
        'Mỗi unit_id có đúng một text, có thể gồm nhiều câu ngữ pháp và nhiều shot. '
        'Chỉ kể chi tiết trong facts được giao cho unit; nhiều unit dùng cùng fact '
        'phải kể các chi tiết khác nhau có thật, không lặp ý để đủ số lượng. '
        'Giữ đúng nhân vật, hành động và trình tự source_order; không tự thêm '
        'kết quả, nội tâm, quan hệ nhân quả hay suy đoán. scene_context, story và '
        'transcript chỉ giúp hiểu mạch, không chứng minh hành động đang thấy. '
        'Chỉ dùng lời người kể, không trích thoại hay sao chép Whisper; chỉ kể '
        'gián tiếp ý nghĩa lời nói khi fact dialogue có nội dung rõ ràng. '
        'Nối đoạn với before/after tự nhiên, không viết lại chúng; giữ tên nhất quán. '
        'Phần ending kể ngắn gọn kết quả được chứng minh, không thêm tổng kết. '
        'Trả đúng ID theo units, không thêm bớt. Backend kiểm số lượng và gắn hình. '
        'Chỉ trả JSON {"sentences":[{"event_id":"E0001","unit_id":"E0001_U001",'
        '"text":"Lời kể."}]}.\n'
        + json.dumps(packet, ensure_ascii=False, separators=(',', ':')))



def unit_response(payload, events):
    rows = payload.get('sentences') if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError('NARRATION_UNIT_SCHEMA_INVALID')
    known = {e['event_id']: e for e in events}
    groups = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('event_id') not in known:
            raise ValueError('NARRATION_UNIT_EVENT_INVALID')
        groups.setdefault(row['event_id'], []).append(row)
    result = []
    for eid, event in known.items():
        units = groups.get(eid, [])
        expected = [u['unit_id'] for u in event['sentence_slots']]
        error = None
        if [u.get('unit_id') for u in units] != expected:
            error = 'NARRATION_UNIT_IDS_INVALID:' + eid + '; expected=' + str(expected)
        elif any(not isinstance(u.get('text'), str) or not u['text'].strip() for u in units):
            error = 'NARRATION_UNIT_TEXT_EMPTY:' + eid
        facts = {f['fact_id']: f for f in event['observed_facts']}
        proofs = []
        for slot in event['sentence_slots']:
            kinds = {facts[fid].get('evidence_kind', 'visual') for fid in slot['fact_ids']}
            if len(kinds) != 1:
                error = 'NARRATION_UNIT_MIXED_SOURCE:' + slot['unit_id']
            proofs.append(dict(fact_ids=slot['fact_ids'], evidence_kind=next(iter(kinds))))
        result.append(dict(event_id=eid, paragraph=' '.join(u.get('text', '') for u in units if isinstance(u.get('text'), str)),
            narration_units=copy.deepcopy(units), sentence_evidence=proofs,
            **({'format_error': error} if error else {})))
    return {'scenes': result}
