"""Exact spoken-sentence budgets, independent of scenes and paragraph count."""
import copy
import json
import math
import re


def spoken_text(value):
    return re.sub(r'\[(?:TIME|BEAT|SHOT|VOICE|SFX)=[^\]]*\]|<(?:VOICE|SFX)=[^>]*>', ' ', str(value), flags=re.I)


def sentences(value):
    return [re.sub(r'[.!?]+["”’»\x27\)\]]*$', '', s).strip() for s in sentence_texts(value)]


def sentence_texts(value):
    """Split a completed paragraph without changing its words or punctuation."""
    text = spoken_text(value).strip()
    pattern = r'(?<!\d)[.!?]+(?:["”’»\x27\)\]]*)\s*|(?<=\d)[!?]+\s*|(?<=\d)\.(?!\d)\s*'
    quoted = list(re.finditer(r'"[^"\n]*"|“[^”\n]*”|«[^»\n]*»|‘[^’\n]*’|(?<!\w)\x27[^\x27\n]+\x27(?!\w)', text))
    result, cursor = [], 0
    for match in re.finditer(pattern, text):
        owner = next((q for q in quoted if q.start() < match.start() < q.end()), None)
        if owner:
            # Punctuation inside a quotation does not end its surrounding prose.
            # At the closing quote, lowercase continuation still belongs to it.
            if re.search(r'\w', text[match.end():owner.end() - 1]):
                continue
            tail = text[owner.end():].lstrip()
            if tail and not tail[0].isupper():
                continue
        unit = text[cursor:match.end()].strip()
        if re.search(r'\w', unit):
            result.append(unit)
        cursor = match.end()
    if re.search(r'\w', text[cursor:]):
        result.append(text[cursor:].strip())
    return result


def words(value):
    return len(re.findall(r'\b\w+\b', spoken_text(value), re.UNICODE))


def constrain_repair_sentences(text, quota):
    """Join excess adjacent repair clauses without dropping or inventing words.

    Repairs/fallbacks sometimes turn one assigned sentence into several factual
    clauses. Semicolons preserve those clauses and their order as a compound
    sentence. A shortage is rejected; we never repeat text to fill a quota.
    """
    text = str(text).strip()
    quota = int(quota)
    count = len(sentences(text))
    if count == quota:
        return text
    if count < quota or quota < 1:
        raise ValueError(f'REPAIR_SENTENCE_QUOTA_MISMATCH: cần {quota} câu, nhận {count}; không thêm câu đệm.')
    units = sentence_texts(text)
    if len(units) != count:
        raise ValueError('REPAIR_SENTENCE_BOUNDARIES_AMBIGUOUS')
    groups = []
    for i in range(quota):
        group = units[len(units) * i // quota:len(units) * (i + 1) // quota]
        clauses = [re.sub(r'[.!?]+(["”’\)\]]*)$', r'\1', s).rstrip() for s in group]
        groups.append('; '.join(clauses) + '.')
    result = ' '.join(groups)
    tokens = lambda s: re.findall(r'\w+', s, re.UNICODE)
    if len(sentences(result)) != quota or tokens(result) != tokens(text):
        raise ValueError('REPAIR_SENTENCE_CONSTRAINT_UNSAFE')
    return result


def allocate_sentences(plan, target_count, target_words):
    result = copy.deepcopy(plan)
    rows = [e for part in result['events_by_part'] for e in part]
    target_count = int(target_count)
    if target_count < len(rows):
        raise ValueError('SENTENCE_BUDGET_TOO_SMALL: số tình huống vượt số câu; cần lập lại mạch truyện.')
    # Legacy plans retain their original safety limits. Situation paragraphs
    # receive a narration budget, not one sentence per observation record.
    capacities = [max(1, 2 * len(e.get('observed_facts') or [])) for e in rows]
    if result.get('situation_plan_version', 0) >= 4:
        from evidence_writer import supported_facts
        if any(not supported_facts(e) for e in rows):
            raise ValueError('SENTENCE_EVIDENCE_SHORTAGE: tinh huong khong co fact da xac nhan.')
        capacities = [target_count] * len(rows)
        closing = [i for i, e in enumerate(rows) if e.get('story_role') == 'ending']
        if closing and len(closing) < len(rows):
            for i in closing:
                capacities[i] = 1
            result['scene_contract']['ending_policy'] = {
                'max_sentences_per_ending_situation': 1,
                'ending_situation_count': len(closing), 'force_ending': False}
    footage = []
    if result.get('situation_plan_version', 0) >= 3:
        for row in rows:
            windows = sorted({(float(s['start_seconds']), float(s['end_seconds']))
                              for s in row.get('visual_sources') or []})
            cursor, seconds_available = -1.0, 0.0
            for start, end in windows:
                seconds_available += max(0.0, end - max(start, cursor))
                cursor = max(cursor, end)
            row['available_visual_seconds'] = seconds_available
            footage.append(seconds_available)
        # Match quota to selected footage, not the entire movie's duration.
        # A six-second shot cannot sustain six full narration sentences.
        seconds_per_sentence = max(3.0, min(6.0, sum(footage) / max(target_count, 1)))
        if result.get('situation_plan_version', 0) < 4:
            capacities = [min(cap, max(1, math.ceil(seconds / seconds_per_sentence)))
                          for cap, seconds in zip(capacities, footage)]
    if sum(capacities) < target_count:
        raise ValueError(f'SENTENCE_EVIDENCE_SHORTAGE: cần {target_count} câu, dữ kiện chỉ hỗ trợ tối đa {sum(capacities)}; không giảm mục tiêu.')
    weights = [(max(1, footage[i] / 6) if footage else max(1, len(e.get('observed_facts') or []))) *
               (1.5 if e.get('story_role') == 'climax' else 1) for i, e in enumerate(rows)]
    if result.get('situation_plan_version', 0) >= 4:
        weights = []
        for event in rows:
            facts = supported_facts(event)
            detail_words = sum(words(f['text']) for f in facts)
            importance = (1.5 if event.get('story_role') == 'climax' else
                          1.25 if event.get('story_role') == 'resolution' else
                          .25 if event.get('story_role') == 'ending' else 1.0)
            # Whisper is context, not extra visual detail to spend sentence quota on.
            weights.append(importance * max(1.0, len(facts) + detail_words / 20))
    counts = [1] * len(rows)
    for _ in range(target_count - len(rows)):
        candidates = [i for i in range(len(rows)) if counts[i] < capacities[i]]
        i = max(candidates, key=lambda i: weights[i] / counts[i])
        counts[i] += 1
    for row, count in zip(rows, counts):
        row.update(sentence_quota=count, tts_target_words=count * 20,
                   tts_min_words=count * 15, tts_max_words=count * 25)
    result['sentence_contract'] = {'target_count': target_count, 'target_words': int(target_words)}
    result['scene_contract']['target_sentence_count'] = target_count
    result['scene_contract']['sentence_tolerance_met'] = True
    return result


def chapter_prompt(events, context):
    cards = [{k: e.get(k) for k in ('event_id', 'sentence_quota', 'tts_target_words',
              'observed_facts', 'allowed_character_names', 'scene_context', 'story_beat_summary',
              'spoken_text', 'transcript_evidence', 'available_visual_seconds')}
             for e in events]
    for card in cards:
        card['observed_facts'] = [
            {'text': f.get('text', ''), 'spoken_text': f.get('spoken_text', '')}
            if isinstance(f, dict) else {'text': str(f), 'spoken_text': ''}
            for f in card.get('observed_facts') or []]
        card['preferred_character_names'] = [c['name'] for c in context.get('characters') or []
            if c.get('name') and c['name'] in (card.get('allowed_character_names') or [])]
    return (
        'VIET CHUONG RECAP THEO MACH TRUYEN, CHUA GHEP HINH. '
        'Doc tat ca tinh huong va mach truyen truoc khi viet. '
        'Moi event_id tra sentences la mang DUNG sentence_quota cau. Moi phan tu la dung mot cau '
        'ket thuc bang dau cham, hoi hoac than. Tong so cau BAT BUOC, khong dem so doan thay so cau. '
        'So tu chi tham khao, khong co gioi han tu bat buoc; uu tien ke tu nhien, dung so cau. '
        'Gom cac cau thanh loi ke lien mach theo nguyen nhan-hanh dong-ket qua; '
        'cau sau them dien bien moi, khong nhac lai muc tieu/cam xuc de du quota. '
        'Chi dung du kien duoc cung cap, khong bia noi tam, hoi thoai hay ket qua. '
        'spoken_text/transcript_evidence la loi thoai tai DUNG tinh huong: dung de hieu '
        'nguyen nhan, quyet dinh, noi dung cuoc noi chuyen thay vi ke "ho noi chuyen". '
        'Loi thoai khong chung minh hanh dong dang hien tren hinh, khong doan nguoi noi. '
        'available_visual_seconds la footage da co: viet gon vua hinh, khong keo dai '
        'bang cam xuc lap lai. Giu dung quota, moi cau them thong tin that. '
        'Chi ke hanh dong co observed_facts; thong tin chi nghe duoc phai ke nhu loi giai thich, '
        'khong bien thanh su kien moi dang xay ra. Dung cau noi chuyen tuyen ngan gon. '
        'Khong mo ta tung shot, khong tim hinh, khong viet timestamp hay visual_links. '
        'Tach ro chuyen tuyen/dia diem; khong ghep hai tuyen bang dung-luc neu chua xac nhan. '
        'Dung ten chinh da xac nhan, khong thay bang mo ta ngoai hinh. '
        'Khong ke logo, opening, credits, montage nhac ket. '
        '\nMACH TOAN PHIM: ' + json.dumps(context.get('all_part_summaries') or [], ensure_ascii=False)
        + '\nDA KE: ' + json.dumps(context.get('previous_part_tail') or [], ensure_ascii=False)
        + '\nTIEP THEO (chua ke): ' + json.dumps(context.get('next_batch_context') or [], ensure_ascii=False)
        + '\nPHONG CACH: ' + str(context.get('style') or '')
        + '\nYEU CAU: ' + str(context.get('custom_prompt') or '')
        + '\nTINH HUONG VA NGAN SACH: ' + json.dumps(cards, ensure_ascii=False)
        + '\nChi tra JSON {"scenes":[{"event_id":"E0001","sentences":["Cau thu nhat."]}]}.'
    )


def best_chapter_candidate(candidates, events):
    """Pick the closest complete candidate per event, preserving event order."""
    result = []
    for event in events:
        options = []
        for payload in candidates:
            rows = payload.get('scenes') or []
            if [r.get('event_id') for r in rows] != [e['event_id'] for e in events]:
                continue
            row = rows[len(result)]
            items = row.get('sentences')
            if not isinstance(items, list) or not items or any(not isinstance(s, str) or not s.strip() for s in items):
                continue
            text = ' '.join(s.strip() for s in items)
            count = len(sentences(text))
            options.append((abs(count - event['sentence_quota']), count < event['sentence_quota'], text))
        if not options:
            raise ValueError('CHAPTER_NO_USABLE_CANDIDATE:' + event['event_id'])
        text = min(options, key=lambda item: item[:2])[2]
        if len(sentences(text)) >= event['sentence_quota']:
            text = constrain_repair_sentences(text, event['sentence_quota'])
        result.append({'event_id': event['event_id'], 'tts': text})
    return result


def validate_chapter(payload, events):
    rows = payload.get('scenes') or []
    if [r.get('event_id') for r in rows] != [e['event_id'] for e in events]:
        raise ValueError('CHAPTER_IDS_INVALID')
    result = []
    for row, event in zip(rows, events):
        items = row.get('sentences')
        if (not isinstance(items, list) or not items
                or any(not isinstance(s, str) or not sentences(s)
                       or not re.search(r'[.!?]["”’\)]*$', s.strip()) for s in items)):
            raise ValueError('CHAPTER_SENTENCE_QUOTA_MISMATCH:' + event['event_id'])
        text = ' '.join(s.strip() for s in items)
        if len(sentences(text)) != event['sentence_quota']:
            raise ValueError('CHAPTER_SENTENCE_QUOTA_MISMATCH:' + event['event_id'])
        result.append({'event_id': event['event_id'], 'tts': text})
    return result


def verify_script(text, target_count, target_words, *, unit_count=None, unit_texts=None):
    if unit_texts is not None:
        expected = re.findall(r'\w+', spoken_text(' '.join(unit_texts)))
        actual = re.findall(r'\w+', spoken_text(text))
        if expected != actual or (unit_count is not None and len(unit_texts) != int(unit_count)):
            raise ValueError('FINAL_NARRATION_UNIT_TEXT_MISMATCH: timeline lost or changed planned narration')
        unit_count = len(unit_texts)
    count, total_words = (int(unit_count) if unit_count is not None else len(sentences(text))), words(text)
    if count != int(target_count):
        raise ValueError(f'FINAL_SENTENCE_COUNT_MISMATCH: cần đúng {target_count} câu, có {count}.')
    return count, total_words
