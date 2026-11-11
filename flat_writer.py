"""Optional flat writer protocol. Original paragraph protocol stays available."""
import json
import os
from pathlib import Path

from evidence_writer import paragraph_prompt, supported_facts
from narrative_contract import sentence_texts


def writer_format():
    value = os.environ.get('RECAP_WRITER_FORMAT')
    if value is None:
        try:
            value = json.loads(Path(__file__).with_name('recap_writer_config.json').read_text(encoding='utf-8'))['format']
        except (OSError, ValueError, KeyError, TypeError):
            value = 'legacy'
    if value not in ('flat', 'legacy'):
        raise ValueError('RECAP_WRITER_FORMAT must be flat or legacy')
    return value


def flat_schema(events):
    if events and all(e.get('narration_unit_version') for e in events):
        from narration_units import unit_schema
        return unit_schema(events)
    # IDs/counts/ownership are checked locally, without large enums or bounds.
    return {'type': 'object', 'properties': {'sentences': {'type': 'array',
        'items': {'type': 'object', 'properties': {
            'event_id': {'type': 'string'}, 'text': {'type': 'string'},
            'fact_ids': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['event_id', 'text', 'fact_ids']}}}, 'required': ['sentences']}


def flat_prompt(events, context, previous=None, error=None):
    if events and all(e.get('narration_unit_version') for e in events):
        from narration_units import unit_prompt
        return unit_prompt(events, context, previous, error)
    original = paragraph_prompt(events, context)
    rules = original.split('Viet xong paragraph', 1)[0]
    cards = original.split('PHONG CACH:', 1)[1]
    return (rules +
        'Lap va viet TRON cum nhu mot doan ke lien mach truoc, sau do tra moi cau thanh mot hang. '
        'Khong lap tung cau doc lap theo tung fact. Moi cum co DUNG sentence_quota hang; '
        'moi hang text co DUNG mot cau hoan chinh. Giu thu tu cum va cau trong cum. '
        'Chi tra JSON {"sentences":[{"event_id":"E0001","text":"Cau ke.","fact_ids":["F1"]}]}. '
        'Khong tra paragraph, scenes, sentence_evidence, evidence_kind, timestamp hay visual_id. '
        'Chi dan fact_ids cua chinh event_id; khong tron fact visual va dialogue trong mot cau, '
        'hay dien dat lai tu mot loai bang chung ro rang. Backend tu suy ra loai bang chung.\n'
        'PHONG CACH:' + cards +
        ('\nSUA RIENG CUM LOI: ' + str(error) + '\nBAN TRUOC: ' + json.dumps(previous, ensure_ascii=False)
         if error else ''))


def flat_to_paragraphs(payload, events):
    if events and all(e.get('narration_unit_version') for e in events):
        from narration_units import unit_response
        return unit_response(payload, events)
    # Canonical cached/legacy drafts remain reusable and fully validated below.
    if isinstance(payload, dict) and isinstance(payload.get('scenes'), list) and 'sentences' not in payload:
        return payload
    units = payload.get('sentences') if isinstance(payload, dict) else None
    if not isinstance(units, list):
        raise ValueError('FLAT_WRITER_SCHEMA_INVALID: sentences must be a list')
    known = {e['event_id']: e for e in events}
    groups, defects, seen_blocks, last = {}, {}, set(), None
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get('event_id'), str) or unit['event_id'] not in known:
            raise ValueError('FLAT_WRITER_EVENT_ID_INVALID')
        eid = unit['event_id']
        if eid != last:
            if eid in seen_blocks:
                defects[eid] = 'FLAT_WRITER_INTERLEAVED_EVENT:' + eid
            seen_blocks.add(eid)
            last = eid
        groups.setdefault(eid, []).append(unit)
    rows = []
    for eid, event in known.items():
        if eid not in groups:
            continue
        facts = {f['fact_id']: f for f in supported_facts(event)}
        texts, proofs = [], []
        for index, unit in enumerate(groups[eid], 1):
            text, ids = unit.get('text'), unit.get('fact_ids')
            if not isinstance(text, str) or len(sentence_texts(text)) != 1:
                defects[eid] = f'FLAT_WRITER_ONE_SENTENCE_REQUIRED:{eid}:row={index}'
                continue
            if not isinstance(ids, list) or not ids or any(not isinstance(fid, str) or fid not in facts for fid in ids):
                defects[eid] = f'FLAT_WRITER_FACT_OWNERSHIP_INVALID:{eid}:row={index}'
                continue
            kinds = {facts[fid].get('evidence_kind', 'visual') for fid in ids}
            if len(kinds) != 1 or not kinds <= {'visual', 'dialogue'}:
                defects[eid] = f'FLAT_WRITER_MIXED_EVIDENCE:{eid}:row={index}; rewrite from one evidence kind'
                continue
            texts.append(text.strip())
            proofs.append(dict(fact_ids=list(ids), evidence_kind=next(iter(kinds))))
        rows.append(dict(event_id=eid, paragraph=' '.join(texts), sentence_evidence=proofs,
                         **({'format_error': defects[eid]} if eid in defects else {})))
    return {'scenes': rows}
