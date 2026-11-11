"""Bounded source review and coherent situation repair without changing shots."""
import copy
import json


def review_prompt(event, manifest):
    return (
        'Xem HÌNH THỰC TẾ của từng shot trong clip, kiểm tra lại các fact dưới đây. '
        'Không mặc định fact cũ hay nhận xét kiểm chứng là đúng. Phân biệt vào/ra, '
        'đến/đi và chủ thể bằng diễn biến nhìn thấy, không đoán từ cốt truyện. '
        'Mỗi fact_id trả đúng một mục: status=confirmed khi hình đủ rõ, text là '
        'dữ kiện tiếng Việt có dấu đã sửa hoặc giữ nguyên. Nếu không rõ hướng '
        'hành động, chỉ ghi điều quan sát chắc chắn được; nếu không đủ bằng chứng '
        'cho fact thì status=unclear. Không thêm nhân vật, hành động, ID hay shot. '
        'Chỉ trả JSON {"facts":[{"fact_id":"F1","status":"confirmed",'
        '"text":"Dữ kiện quan sát được."}]}.\nMANIFEST: '
        + json.dumps(manifest, ensure_ascii=False) + '\nFACTS: '
        + json.dumps(event['observed_facts'], ensure_ascii=False))


def reviewed_event(event, response):
    """Apply independent observations only; retain IDs, quotas and source ranges."""
    from evidence_writer import require_vietnamese_narration
    rows = response.get('facts') if isinstance(response, dict) else None
    facts = event['observed_facts']
    if (not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows)
            or len(rows) != len(facts)
            or {r.get('fact_id') for r in rows} != {f['fact_id'] for f in facts}):
        raise ValueError('SOURCE_FACT_REVIEW_IDS_INVALID')
    updates = {r['fact_id']: r for r in rows}
    result = copy.deepcopy(event)
    for fact in result['observed_facts']:
        row = updates[fact['fact_id']]
        if row.get('status') != 'confirmed' or not isinstance(row.get('text'), str):
            raise ValueError('SOURCE_FACT_REVIEW_UNRESOLVED:' + fact['fact_id'])
        text = row['text'].strip()
        require_vietnamese_narration(text, fact['fact_id'])
        fact.setdefault('original_observation_text', fact['text'])
        fact['text'] = text
        fact['raw_text'] = text
        fact['source_review_version'] = 1
    by_id = {f['fact_id']: f['text'] for f in result['observed_facts']}
    result['event'] = ' '.join(by_id.values())
    result['narration_focus'] = result['event']
    for slot in result.get('sentence_slots') or []:
        slot['focus'] = ' '.join(by_id[fid] for fid in slot['fact_ids'])
    # Descriptions and summaries are context, never proof over reviewed facts.
    for field in ('visual_sources', 'locked_visual_shots'):
        for visual in result.get(field) or []:
            ids = visual.get('fact_ids') or []
            if ids:
                visual['description'] = ' '.join(by_id[fid] for fid in ids if fid in by_id)
    result['source_fact_review_version'] = 1
    return result


def coherent_repair_context(scenes, scene_id, context):
    result = copy.deepcopy(context)
    index = next(i for i, s in enumerate(scenes) if s['scene_id'] == scene_id)
    result['previous_part_tail'] = [{'tts': s['tts']} for s in scenes[max(0, index - 2):index]]
    result['next_batch_context'] = [{'event': s['tts']} for s in scenes[index + 1:index + 3]]
    result['custom_prompt'] = str(result.get('custom_prompt') or '') + (
        '\nSỬA CẢ CỤM ĐANG LỖI thành một đoạn recap mạch lạc theo facts vừa '
        'được xem lại. Đoạn trước và sau chỉ là ngữ cảnh, không viết lại chúng. '
        'Nối ý tự nhiên, giữ chủ thể, tên gọi và trình tự nhất quán; chỉ dùng từ '
        'nối khi bằng chứng hỗ trợ, không bịa quan hệ nhân quả. Giữ nguyên '
        'unit_id, số đơn vị, fact_id và shot đã giao. Facts đã xem lại là nguồn '
        'cho hành động; bản cũ và tóm tắt có thể sai.\nCỤM CŨ: '
        + scenes[index]['tts'])
    return result
