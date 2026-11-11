import copy
import json
import math
import re
from semantic_visuals import semantic_visual_plan, visual_catalog
from evidence_granularity import separate_event_granularity


MIN_VISUAL_BEAT_CONFIDENCE = 0.78
MIN_VISUAL_BEAT_SECONDS = 2.0
MAX_VISUAL_BEAT_SECONDS = 3.0
TARGET_VISUAL_SHOT_SECONDS = 3.0
MAX_GROUNDED_VISUAL_SHOT_SECONDS = 3.5
# A semantic beat describes one visible action/reaction and is deliberately
# wider than a render shot.  The renderer divides it into short shots only
# after narration/TTS duration is known.
MAX_GROUNDED_VISUAL_BEAT_SECONDS = 15.0
MIN_BEAT_PREFIX_WORDS = 3
MIN_TTS_WORDS = 15
MAX_TTS_WORDS = 25
MIN_NARRATION_ANCHOR_SECONDS = 6.0
PLANNING_TTS_WORDS_PER_SECOND = 3.0
LONG_STORY_SECONDS = 600.0
MAX_STORY_BATCH_SECONDS = 900.0
MAX_STORY_BATCH_SCENES = 50
NARRATIVE_RELATIONS = {
    "opening", "continuation", "cause", "consequence", "contrast",
    "parallel", "time_jump", "location_change", "return",
}
EXPLICIT_TRANSITION_RELATIONS = {
    "consequence", "contrast", "parallel", "time_jump",
    "location_change", "return",
}
ABSTRACT_PHRASES = (
    "mọi chuyện", "điều đó", "tình hình", "một điều gì đó",
    "định mệnh", "số phận", "ý nghĩa cuộc sống", "bài học quý giá",
    "hiểu rằng", "nhận ra rằng", "thầm nghĩ", "tự nhủ", "trong lòng",
    "sâu thẳm", "dường như", "có vẻ như", "quyết tâm",
    "somehow", "things change", "the situation",
)
TERMINAL_STORY_PHRASES = (
    "hoàn thành nhiệm vụ", "hoàn thành chuyến đi", "kết thúc chuyến đi",
    "hoàn toàn bị đánh bại", "bị đánh bại hoàn toàn", "biến mất vào hư không",
    "bình yên trở lại", "để lại sự bình yên", "mọi chuyện kết thúc",
    "cuối cùng cũng kết thúc", "chào tạm biệt", "trở về cuộc sống bình thường",
)
REPETITION_STOPWORDS = {
    "và", "là", "của", "có", "đã", "đang", "được", "một", "những",
    "các", "khi", "sau", "trước", "rồi", "nhưng", "vẫn", "với", "để",
    "trong", "trên", "dưới", "này", "đó", "lại", "thì", "mà", "từ",
    "vào", "ra", "cho", "ở", "bị", "nên", "cùng", "về", "đến", "theo",
}

# Repeated use of these phrases usually means sparse evidence was padded
# instead of advancing the story. Prompts use this list as a soft deny-list.
NARRATION_PADDING_PHRASES = (
    "đầy hứa hẹn sắp tới", "cuộc sống mới tươi đẹp", "vô cùng vui mừng khôn xiết",
    "chuẩn bị tinh thần cho chặng đường tiếp theo", "sẵn sàng cho chuyến đi đầy hứa hẹn",
    "quyết tâm cao độ", "vô cùng nguy hiểm", "tàn ác khủng khiếp",
    "niềm vui sướng tột độ", "an toàn tuyệt đối", "đầy rẫy hiểm nguy",
    "sức mạnh tà ác", "bất chấp mọi hiểm nguy", "cảm thấy lo lắng",
    "những diễn biến ngầm", "tình thế đổi thay", "đoạn đối thoại tiếp nối",
    "tạo tiền đề cho bước ngoặt", "diễn biến quan trọng tiếp tục",
)
ENGLISH_PROSE_MARKERS = {
    "the", "and", "are", "was", "were", "with", "from", "they", "their",
    "his", "her", "home", "room", "meet", "discuss", "running", "found",
    "which", "used", "both", "into", "after", "before", "because",
    "wears", "wearing", "blue", "round", "red", "collar", "bell",
    "glasses", "yellow", "shirt", "black", "stripes", "dark", "mask",
    "green", "cloak", "clay", "statue", "ceramic", "monster",
}

VIETNAMESE_FUNCTION_WORDS = {
    "anh", "bạn", "bị", "các", "chỉ", "cho", "chúng", "cô", "cậu",
    "của", "đã", "đang", "đến", "để", "đó", "được", "giữa", "họ",
    "khi", "không", "lại", "làm", "mà", "mình", "một", "này", "nên",
    "người", "nhóm", "nhưng", "những", "ra", "rồi", "sau", "sẽ",
    "thì", "trong", "trước", "từ", "và", "vẫn", "về", "vì", "với",
}


def character_name_in_text(name, text):
    for match in re.finditer(rf'(?<!\w){re.escape(name)}(?!\w)', str(text), re.I):
        if str(name).casefold() == 'gian' and re.search(r'(?:thời|không)\s+$', str(text)[:match.start()], re.I):
            continue
        return True
    return False


def vietnamese_narration_issues(value):
    """Return only clear language violations, allowing foreign proper names."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ["empty"]
    tokens = [
        token.casefold() for token in re.findall(
            r"[^\W_]+", text, flags=re.UNICODE
        )
    ]
    token_set = set(tokens)
    issues = []
    english_hits = sum(token in ENGLISH_PROSE_MARKERS for token in tokens)
    vietnamese_hits = sum(token in VIETNAMESE_FUNCTION_WORDS for token in tokens)
    has_vietnamese_letters = bool(re.search(
        r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
        r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
        text.casefold(),
    ))
    if english_hits >= 3 and vietnamese_hits < 2:
        issues.append("english_prose")
    if len(tokens) >= 8 and not has_vietnamese_letters and vietnamese_hits < 3:
        issues.append("vietnamese_diacritics_missing")
    promotional = (
        "subscribe", "đăng ký kênh", "ủng hộ kênh", "like và share",
    )
    if any(phrase in text.casefold() for phrase in promotional):
        issues.append("promotional_text")
    return issues


UNKNOWN_ALIAS_PATTERNS = (
    (("blue", "round", "collar", "bell"), "mèo máy màu xanh"),
    (("glasses", "yellow", "shirt"), "cậu bé đeo kính"),
    (("dark", "mask", "cloak"), "kẻ đeo mặt nạ"),
    (("clay", "statue"), "quái vật tượng gốm"),
    (("ceramic", "statue"), "quái vật tượng gốm"),
    (("earthen", "monster"), "quái vật tượng gốm"),
    (("tiger", "skin"), "người mặc áo da thú"),
)


def _descriptive_character_alias(character, index=1):
    """Return a short Vietnamese label without exposing raw vision prose."""
    character = character if isinstance(character, dict) else {}
    name = re.sub(r"\s+", " ", str(character.get("name", ""))).strip()
    if name:
        return name
    alias = re.sub(
        r"\s+", " ", str(character.get("descriptive_alias", ""))
    ).strip().strip(".。")
    if alias and not _looks_like_english_prose(alias):
        return alias
    description = re.sub(
        r"\s+", " ", str(character.get("description", ""))
    ).strip().strip(".。")
    if description and not _looks_like_english_prose(description):
        return description
    signature = " ".join((
        str(character.get("visual_signature", "")),
        str(character.get("description", "")),
    )).casefold()
    for markers, label in UNKNOWN_ALIAS_PATTERNS:
        if all(marker in signature for marker in markers):
            return label
    role = re.sub(r"\s+", " ", str(character.get("role", ""))).strip()
    if role and not _looks_like_english_prose(role):
        return role
    return f"nhân vật chưa xác định {int(index)}"


def _character_display_labels(characters):
    labels = {}
    for index, character in enumerate(characters or [], start=1):
        if not isinstance(character, dict):
            continue
        character_id = str(character.get("id", "")).strip().upper()
        if not character_id:
            continue
        name = re.sub(r"\s+", " ", str(character.get("name", ""))).strip()
        labels[character_id] = (character.get('display_name') or name or character.get('neutral_alias')
                                or character.get('descriptive_alias') or character.get('alias') or f"nhân vật {index}")
    return labels


def _replace_internal_character_ids(text, labels=None):
    labels = {str(key).upper(): value for key, value in (labels or {}).items()}
    from character_identity import render_ids
    return render_ids(text, labels)


def _looks_like_english_prose(text):
    tokens = re.findall(r"[A-Za-z]+", str(text or "").lower())
    return sum(token in ENGLISH_PROSE_MARKERS for token in tokens) >= 3


def style_storytelling_prompt(style):
    """Translate the UI style choice into concrete narration behavior."""
    style_key = str(style or "dramatic").strip().lower()
    prompts = {
        "dramatic": (
            "Kể như một câu chuyện điện ảnh căng thẳng, bám sát quan hệ nguyên nhân-kết quả. "
            "Ưu tiên nguy cơ, lựa chọn, bước ngoặt và hậu quả; không cường điệu hoặc tả phong cảnh cho đủ lời."
        ),
        "humorous": (
            "Kể dí dỏm bằng nhận xét ngắn dựa trên hành động thật của nhân vật. Giữ cốt truyện rõ ràng, "
            "không chèn trò đùa vào mọi câu, không chế lời thoại và không biến nguy hiểm thành chuyện vô nghĩa."
        ),
        "shorts": (
            "Kể nhanh, trực tiếp và giàu động từ; đưa biến cố quan trọng lên trước. Loại cảnh trang trí, "
            "cảm xúc suy diễn và chi tiết không làm thay đổi câu chuyện; chuyển ý ngắn nhưng vẫn đủ nguyên nhân-kết quả."
        ),
        "news": (
            "Kể trung tính như bản tin: ai làm gì, ở đâu, chuyện gì xảy ra và hậu quả ra sao. "
            "Theo đúng trình tự thời gian, dùng dữ kiện nhìn thấy được, không giật gân và không suy diễn nội tâm."
        ),
        "tutorial": (
            "Kể theo hướng giải thích sâu: làm rõ mục tiêu, cách nhân vật xử lý vấn đề, lý do và kết quả. "
            "Vẫn giữ mạch truyện theo thời gian, tránh lặp lại cùng một bài học hoặc mô tả thao tác vụn vặt."
        ),
    }
    return prompts.get(style_key, prompts["dramatic"])


def _json_object(text):
    """Extract the largest complete JSON object from a model response.

    Gemini can still surround JSON with Markdown or a short explanation even
    when ``application/json`` is requested.  Slicing from the first ``{`` to
    the last ``}`` joins separate objects together and also mistakes braces in
    prose for JSON.  ``raw_decode`` instead stops at the matching closing brace
    while correctly respecting nested objects and braces inside strings.
    """
    source = str(text or "").lstrip("\ufeff").strip()
    if not source:
        raise ValueError("Gemini không trả về nội dung JSON.")

    decoder = json.JSONDecoder(strict=False)
    objects = []
    seen_spans = set()

    def collect(candidate, base_offset=0):
        for match in re.finditer(r"[\{\[]", candidate):
            start = match.start()
            try:
                value, end = decoder.raw_decode(candidate, start)
            except json.JSONDecodeError:
                continue
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
                value = value[0]
            if not isinstance(value, dict):
                continue
            span = (base_offset + start, base_offset + end)
            if span not in seen_spans:
                seen_spans.add(span)
                objects.append((end - start, value))

    # Prefer fenced payloads when present, but also scan the complete response
    # because models occasionally forget the closing Markdown fence.
    for fenced in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", source, re.I):
        collect(fenced.group(1), fenced.start(1))
    collect(source)

    if objects:
        # An explanation may contain a tiny JSON example before the real
        # result.  The requested payload is consistently the largest object.
        return max(objects, key=lambda item: item[0])[1]
    raise ValueError("Gemini không trả về JSON object hoàn chỉnh, hợp lệ.")


def parse_json_object(text):
    """Public parser used by the request pipeline and its validators."""
    return _json_object(text)


def _time_seconds(value):
    if value is None or isinstance(value, bool):
        raise ValueError(f"Timestamp không hợp lệ: {value}")
    source = str(value).strip()
    if not source:
        raise ValueError(f"Timestamp không hợp lệ: {value}")
    # Gemini sometimes copies reel offsets as raw seconds (for example 03.5)
    # instead of the requested MM:SS form. Treat a colon-free finite number as
    # seconds; all downstream serialization still uses canonical timestamps.
    if ":" not in source:
        try:
            seconds = float(source)
        except ValueError as exc:
            raise ValueError(f"Timestamp không hợp lệ: {value}") from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Timestamp không hợp lệ: {value}")
        return seconds
    fields = source.split(":")
    if len(fields) not in (2, 3):
        raise ValueError(f"Timestamp không hợp lệ: {value}")
    try:
        fields = [float(item) for item in fields]
    except ValueError as exc:
        raise ValueError(f"Timestamp không hợp lệ: {value}") from exc
    if any(not math.isfinite(item) or item < 0 for item in fields):
        raise ValueError(f"Timestamp không hợp lệ: {value}")
    if len(fields) == 2:
        minutes, seconds = fields
        if seconds >= 60:
            raise ValueError(f"Timestamp không hợp lệ: {value}")
        return minutes * 60 + seconds
    hours, minutes, seconds = fields
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Timestamp không hợp lệ: {value}")
    return hours * 3600 + minutes * 60 + seconds


def _format_time(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def three_part_targets(target_words):
    """Split a narration budget evenly without losing remainder words."""
    target_words = max(3, int(target_words))
    base, remainder = divmod(target_words, 3)
    return [base + (1 if index < remainder else 0) for index in range(3)]


def proportional_part_targets(weights, target_words):
    """Scale three model-selected story weights to one total word target."""
    target = max(3, int(target_words))
    cleaned = []
    for value in list(weights or [])[:3]:
        try:
            cleaned.append(max(0.0, float(value)))
        except (TypeError, ValueError):
            cleaned.append(0.0)
    while len(cleaned) < 3:
        cleaned.append(0.0)
    if sum(cleaned) <= 0:
        return three_part_targets(target)

    remaining = target - 3
    total_weight = sum(cleaned)
    exact = [remaining * weight / total_weight for weight in cleaned]
    allocated = [1 + int(value) for value in exact]
    leftover = target - sum(allocated)
    order = sorted(
        range(3),
        key=lambda index: exact[index] - int(exact[index]),
        reverse=True,
    )
    for index in order[:leftover]:
        allocated[index] += 1
    return allocated


def requested_word_range(target_words):
    """Keep narration close to the requested duration (97%-103%)."""
    target = max(1, int(target_words))
    return (
        target,
        max(1, math.ceil(target * 0.97)),
        max(1, math.floor(target * 1.03)),
    )


def expanded_word_target(target_words):
    """Preserve the user's requested word target without hidden expansion."""
    return max(1, int(target_words))


def required_tts_scene_count(target_words, required_event_count=0):
    """Plan about one sentence per 20 requested words, bounded by TTS limits."""
    target = max(1, int(target_words))
    minimum_for_words = 1 + math.ceil(max(0, target - 20) / MAX_TTS_WORDS)
    natural_count = max(1, math.floor(target / 20.0 + 0.5))
    maximum_for_words = max(1, target // MIN_TTS_WORDS)
    return max(
        max(0, int(required_event_count)),
        minimum_for_words,
        min(natural_count, maximum_for_words),
    )


def lock_story_scene_contract(events_by_part, target_words):
    """Assign immutable scene IDs and exact per-scene word budgets.

    Allocation follows story density already expressed by the selected event
    slots.  Climax/reversal slots receive spare words first, while the single
    ending remains capped at 20 words.
    """
    parts = [list(part or []) for part in list(events_by_part or [])[:3]]
    while len(parts) < 3:
        parts.append([])
    flat = [event for part in parts for event in part if isinstance(event, dict)]
    if not flat:
        raise ValueError("Khong co scene de khoa ngan sach TTS.")
    floors = [MIN_TTS_WORDS] * len(flat)
    caps = [
        20 if str(event.get("story_role", "")).lower() == "ending"
        else MAX_TTS_WORDS
        for event in flat
    ]
    requested = int(target_words)
    if requested < sum(floors) or requested > sum(caps):
        raise ValueError(
            f"WORD_SCENE_CONTRACT_INFEASIBLE: {len(flat)} scene chi chua duoc "
            f"{sum(floors)}-{sum(caps)} tu, khong the khoa {requested} tu."
        )
    budgets = list(floors)
    priorities = []
    for index, event in enumerate(flat):
        role = str(event.get("story_role", "development")).lower()
        function = str(event.get("narrative_function", "development")).lower()
        importance = str(event.get("selection_importance", "")).lower()
        score = (
            5 if role == "climax" or function in {"climax", "reversal"} else
            4 if importance == "critical" else
            3 if role == "resolution" else
            1 if role == "ending" else 2
        )
        priorities.append((score, index))
    remaining = requested - sum(budgets)
    while remaining:
        progressed = False
        for _score, index in sorted(priorities, reverse=True):
            if budgets[index] >= caps[index]:
                continue
            budgets[index] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise ValueError("Khong the phan bo het ngan sach tu cho scene da khoa.")
    for scene_index, (event, budget) in enumerate(zip(flat, budgets), start=1):
        event["scene_id"] = f"S{scene_index:04d}"
        event["tts_target_words"] = int(budget)
        event["tts_min_words"] = MIN_TTS_WORDS
        event["tts_max_words"] = 20 if str(
            event.get("story_role", "")
        ).lower() == "ending" else MAX_TTS_WORDS
        event["estimated_tts_seconds"] = round(
            int(budget) / PLANNING_TTS_WORDS_PER_SECOND, 2
        )
    for part in parts:
        ordered = sorted(part, key=lambda event: float(event.get("start_seconds", 0.0)))
        for index, event in enumerate(ordered):
            start = float(event.get("start_seconds", 0.0))
            own_end = float(event.get("end_seconds", start))
            next_start = (
                float(ordered[index + 1].get("start_seconds", own_end))
                if index + 1 < len(ordered) else float(
                    event.get("part_boundary_end_seconds", own_end)
                )
            )
            needed = float(event.get("estimated_tts_seconds", 0.0))
            available_end = max(own_end, next_start)
            narration_end = min(available_end, max(own_end, start + needed))
            if narration_end > own_end:
                event.setdefault("evidence_end_seconds", own_end)
                event["end_seconds"] = round(narration_end, 3)
                event["tts_visual_extended"] = True
            available = max(0.0, narration_end - start)
            event["available_visual_seconds"] = round(available, 2)
            event["visual_coverage_risk"] = bool(
                available + 0.25 < float(event.get("estimated_tts_seconds", 0.0))
            )
    return {
        "target_words": requested,
        "scene_count": len(flat),
        "part_scene_counts": [len(part) for part in parts],
        "part_word_targets": [
            sum(int(event.get("tts_target_words", 0)) for event in part)
            for part in parts
        ],
    }


def build_story_gap_supplement_prompt(plan_draft, error):
    """Ask a video-backed retry to retain old anchors and fill only gaps."""
    error_text = str(error)
    count_match = re.search(
        r"chỉ có\s+(\d+)\s+mốc;\s*cần ít nhất\s+(\d+)",
        error_text,
        flags=re.IGNORECASE,
    )
    deficit_rule = ""
    if count_match:
        current_count = int(count_match.group(1))
        required_count = int(count_match.group(2))
        deficit_rule = (
            f"- Phần bị lỗi hiện có {current_count} mốc và cần ít nhất "
            f"{required_count}; lượt này phải tìm thêm tối thiểu "
            f"{max(1, required_count - current_count)} mốc thật khác nhau.\n"
        )
    return f"""Xem lại TOÀN BỘ video, tập trung vào quãng cốt truyện còn trống được nêu trong lỗi.
Đây là lượt BỔ SUNG event map, không phải tạo lại từ đầu.

QUY TẮC BẮT BUỘC:
- Giữ nguyên mọi required_event hợp lệ trong JSON hiện tại: event_id, timestamp, event và part.
- Không xóa, đổi ID, đổi timestamp hoặc viết lại các mốc cũ.
- Chỉ thêm required_event mới vào đúng khoảng bị thiếu để tạo chuỗi nguyên nhân -> hành động -> thay đổi -> hệ quả.
- ID mới phải duy nhất và không trùng bất kỳ ID cũ nào.
- Không được chỉ chép lại JSON cũ. Mỗi lượt bổ sung phải tăng số required_event bằng sự kiện thật nhìn thấy trong video.
{deficit_rule}- Giữ thứ tự thời gian; ưu tiên nguyên nhân, bước ngoặt, hậu quả và cảnh nối đang thiếu thay vì phong cảnh.
- Giữ nguyên hai boundary. Trả lại TOÀN BỘ JSON hoàn chỉnh, không Markdown hay giải thích.
- Mỗi phần dài phải có đủ mốc đầu-giữa-cuối; không dồn mốc vào một đầu.

LỖI CẦN BỔ SUNG:
{error_text}

JSON HIỆN TẠI CẦN GIỮ:
{str(plan_draft)}
"""


def merge_story_plan_supplement(plan_draft, supplement_text):
    """Accumulate new anchors while deterministically preserving old ones."""
    previous = _json_object(plan_draft)
    supplement = _json_object(supplement_text)
    merged = copy.deepcopy(supplement)

    for key in ("boundary_1_seconds", "boundary_2_seconds"):
        if key in previous:
            merged[key] = previous[key]
    for key in ("story_contract", "characters"):
        if previous.get(key) and not merged.get(key):
            merged[key] = copy.deepcopy(previous[key])

    previous_parts = {
        int(part.get("part", 0) or 0): part
        for part in (previous.get("parts") or [])
        if isinstance(part, dict)
    }
    supplement_parts = {
        int(part.get("part", 0) or 0): copy.deepcopy(part)
        for part in (supplement.get("parts") or [])
        if isinstance(part, dict)
    }
    used_ids = {
        str(event.get("event_id", "")).strip().upper()
        for part in previous_parts.values()
        for event in (part.get("required_events") or [])
        if isinstance(event, dict) and str(event.get("event_id", "")).strip()
    }
    highest_numeric_id = max(
        (
            int(match.group(1))
            for event_id in used_ids
            for match in [re.fullmatch(r"E(\d+)", event_id)]
            if match
        ),
        default=0,
    )

    def fresh_event_id():
        nonlocal highest_numeric_id
        while True:
            highest_numeric_id += 1
            candidate = f"E{highest_numeric_id:04d}"
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    def event_signature(event):
        try:
            start = round(float(event.get("start_seconds")), 1)
            end = round(float(event.get("end_seconds")), 1)
        except (TypeError, ValueError):
            start, end = None, None
        text = re.sub(
            r"\s+", " ", str(event.get("event", ""))
        ).strip().casefold()
        return start, end, text

    known_signatures = {
        event_signature(event)
        for part in previous_parts.values()
        for event in (part.get("required_events") or [])
        if isinstance(event, dict)
    }
    known_ranges = {
        signature[:2] for signature in known_signatures
        if signature[0] is not None and signature[1] is not None
    }
    combined_parts = []
    for part_number in (1, 2, 3):
        old_part = previous_parts.get(part_number, {})
        new_part = supplement_parts.get(part_number, {})
        combined = copy.deepcopy(new_part or old_part)
        combined["part"] = part_number
        for key in ("summary", "end_state", "handoff_to_next"):
            if old_part.get(key) and not combined.get(key):
                combined[key] = old_part[key]

        old_events = [
            copy.deepcopy(event) for event in (old_part.get("required_events") or [])
            if isinstance(event, dict)
        ]
        old_ids = {
            str(event.get("event_id", "")).strip().upper()
            for event in old_events
            if str(event.get("event_id", "")).strip()
        }
        additions = []
        for event in new_part.get("required_events") or []:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id", "")).strip().upper()
            signature = event_signature(event)
            same_existing_range = (
                signature[0] is not None and signature[1] is not None
                and any(
                    abs(signature[0] - start) <= 2.0
                    and abs(signature[1] - end) <= 2.0
                    for start, end in known_ranges
                )
            )
            if signature in known_signatures or same_existing_range:
                continue
            addition = copy.deepcopy(event)
            if not event_id or event_id in used_ids:
                addition["event_id"] = fresh_event_id()
            else:
                addition["event_id"] = event_id
                used_ids.add(event_id)
            additions.append(addition)
            old_ids.add(str(addition["event_id"]).upper())
            known_signatures.add(signature)
            if signature[0] is not None and signature[1] is not None:
                known_ranges.add(signature[:2])
        combined["required_events"] = old_events + additions
        def event_start(event):
            try:
                return float(event.get("start_seconds"))
            except (TypeError, ValueError):
                return float("inf")
        combined["required_events"].sort(key=event_start)
        combined_parts.append(combined)
    merged["parts"] = combined_parts
    return json.dumps(merged, ensure_ascii=False)


def build_chunk_evidence_prompt(
    chunk_number, chunk_count, clip_duration_sec, global_start_sec,
    global_end_sec, candidate_count, story_outline=None, character_registry=None,
    primary_end_sec=None, previous_end_state=None, transcript_text="",
):
    """Ask one short video request for compact, reusable visual evidence."""
    registry = [
        {
            "id": str(item.get("id", "")).strip(),
            "name": str(item.get("name", "")).strip(),
            "alias": str(
                item.get("alias") or item.get("descriptive_alias") or ""
            ).strip(),
            "signature": str(
                item.get("signature") or item.get("visual_signature") or ""
            ).strip(),
        }
        for item in (character_registry or []) if isinstance(item, dict)
    ]
    ending_rule = (
        "Day la phan cuoi: event ket thuc that phai co importance=3."
        if int(chunk_number) == int(chunk_count) else
        "Day chua phai phan cuoi: khong tu ket luan phim da ket thuc."
    )
    primary_end = float(
        primary_end_sec if primary_end_sec is not None else global_end_sec
    )
    local_primary_end = max(
        0.0, min(float(clip_duration_sec), primary_end - float(global_start_sec))
    )
    previous_summary = ""
    if isinstance(previous_end_state, dict):
        previous_summary = str(previous_end_state.get("summary", "")).strip()
    outline_guide = [
        item for item in (story_outline or []) if isinstance(item, dict)
    ]
    transcript = str(transcript_text or "").strip()
    return f"""Xem het video phan {int(chunk_number)}/{int(chunk_count)}, tu
{float(global_start_sec):.2f}s den {float(global_end_sec):.2f}s cua video goc. CHUA viet loi TTS.

Video co the da bo audio de tiet kiem token. LỜI THOẠI WHISPER ben duoi dung timestamp TOAN CUC;
dung no lam bang chung cho cau noi/ten duoc goi. Video la bang chung cho hanh dong nhin thay.
Khong gan loi thoai cho mot nguoi chi vi nguoi do dang xuat hien tren hinh.

NHIEM VU DUY NHAT: liet ke rong cac bien co that theo thu tu thoi gian bang JSON ngan.
- Muc tieu khoang {int(candidate_count)} bien co, nhung chi ghi thay doi that ve hanh dong, muc tieu,
  nguy co, thong tin, quan he hoac ket qua; khong tao canh gia de du so.
- Doan yen binh duoc tra it hon; moc nay KHONG PHAI gioi han output va khong phai ly do cat ngang cao trao.
- Tach cac buoc khac nhau cua bien co lon: phat hien, quyet dinh, hanh dong, dao chieu, ket qua.
- `events` CHI chua su kien cu the: ai lam gi, thay doi/ket qua gi, moc hinh bat dau-ket thuc that.
  Gan `granularity="atomic"`. Mot su kien co the co nhieu cau, khong ep thoi luong co dinh.
- Tom tat ca chang nhu "nhom ban den the gioi moi va xay dung cuoc song" dua vao `story_context`,
  KHONG dua vao events cung hang voi cac hanh dong chi tiet ben trong.
- Neu chua tach duoc su kien cu the, gan `granularity="uncertain"` de kiem tra; khong chia deu timestamp
  de tao event gia. Hanh dong lien tuc dai van co the la atomic neu video chung minh no.
- start/end la giay CUC BO trong clip, tu 0 den {float(clip_duration_sec):.2f}.
- Phan chinh ket thuc tai giay cuc bo {local_primary_end:.2f}; doan sau la vung chong lan, van ghi event moi.
- Su kien dang do phai co continues_from_previous/continues_to_next; vung chong lan khong duoc lam mat event.
- event viet tieng Viet ngan gon; khong viet van recap, khong viet tieng Anh.
- Bat buoc tach nghia vao cause, action, change va consequence: nguyen nhan, hanh dong chinh,
  thay doi ngay trong canh va hau qua sau canh. Neu mot muc khong co bang chung thi de rong, khong bia.
- visual_detail ghi hinh anh cu the co the dung de cat shot. Cac muc khong duoc chi chep lai cung mot cau event.
- Trong CUNG luot xem nay, moi event tra `visual_shots` la cac BEAT HINH co y nghia nam ben trong
  start/end cua event: hanh dong, phan ung, phat hien, thay doi hoac ket qua nhin thay truc tiep.
  Beat dai linh hoat theo hanh dong, thuong 1-8 giay, toi da 15 giay; khong ep 2-3 giay va khong ep
  so luong co dinh. Neu khong chac timestamp thi tra it hon, khong chia deu hay doan frame.
  description cua moi beat can noi ro ai lam gi, voi ai/vat gi va ket qua nhin thay;
  khong chep cung mot mo ta chung cho nhieu hanh dong khac nhau.
  Day la vung hanh dong cho tung y ke; backend se dat diem cat sau khi co word timing TTS.
- importance: 0=phu, 1=noi mach, 2=chinh, 3=buoc ngoat/cao trao/ket thuc.
- role chi dung development, climax, resolution hoac ending.
- cluster dung cung ma C01, C02... cho cac buoc thuoc cung mot bien co lon.
- who chi dung ID nhan vat. Nguoi chua co ten phai co alias mo ta ngan, khong bia ten rieng.
- Neu nhan vat khop so khoa thi giu dung name/alias; khong tu doi ten.
- {ending_rule}

SO NHAN VAT DA KHOA TU CAC DOAN TRUOC: {json.dumps(registry, ensure_ascii=False, separators=(',', ':'))}
TRANG THAI CUOI CLIP TRUOC: {previous_summary or 'khong co'}
DINH HUONG TU LUOT XEM TOAN PHIM (video trong clip nay moi la bang chung):
{json.dumps(outline_guide, ensure_ascii=False, separators=(',', ':'))}
LỜI THOẠI WHISPER TRONG CUA SO NAY:
{transcript or 'khong co; dung audio video neu fallback co audio'}

Chi tra MOT JSON object, khong Markdown, chi dung schema nay:
{{"characters":[{{"id":"C1","name":"","alias":"cậu bé áo da thú",
"signature":"tóc đen, áo da thú","confidence":0.6}}],
"story_context":[],"events":[{{"id":"E001","granularity":"atomic","start":1.0,"end":8.5,"who":["C1"],
"event":"cậu bé phát hiện ngôi làng bị tấn công rồi chạy đi cầu cứu",
"visual_detail":"cậu bé nhìn thấy lửa rồi chạy khỏi làng",
"cause":"ngôi làng bất ngờ bị tấn công","action":"cậu bé chạy đi cầu cứu",
"change":"cậu chuyển từ hoảng sợ sang hành động",
"consequence":"tin về cuộc tấn công được mang ra ngoài",
"visual_shots":[{{"start_seconds":1.2,"end_seconds":4.5,
"description":"cậu bé nhìn thấy lửa rồi quay người chạy khỏi làng"}}],
"importance":2,"cluster":"C01","role":"development"}}]}}
"""

    # Legacy verbose schema retained below only for compatibility with older
    # prompt snapshots; the compact return above is the active contract.
    ending_rule = (
        "- Day la doan CUOI: bat buoc lay resolution/ending that neu co va gan story_role=ending; bo credit/outro."
        if int(chunk_number) == int(chunk_count) else
        "- Day chua phai doan cuoi: khong tu ket luan cau chuyen da ket thuc."
    )
    outline_guide = list(story_outline or [])
    registry = [
        item for item in (character_registry or []) if isinstance(item, dict)
    ]
    outline_rule = (
        "\nDINH HUONG TU LUOT XEM TOAN PHIM (chi la ban do tim canh; video trong "
        "clip nay moi la bang chung):\n"
        + json.dumps(outline_guide, ensure_ascii=False, separators=(",", ":"))
        + "\n- Uu tien xac minh cac beat tren; neu outline sai thi sua theo clip, khong bia cho khop.\n"
        if outline_guide else ""
    )
    registry_rule = (
        "\nSO NHAN VAT DA KHOA TU CAC DOAN TRUOC (chi dung ten khi khop bang chung):\n"
        + json.dumps(registry, ensure_ascii=False, separators=(",", ":"))
        + "\n- Neu nhan vat khop mot ID trong so nay, dung dung ten chuan; khong rut gon, "
          "bien am hoac tu tao bien the cua ten.\n"
        + "- Neu chua du bang chung de khop, de name rong va dung descriptive_alias; "
          "khong doan ten de lam JSON dep hon.\n"
        if registry else ""
    )
    primary_end = float(
        primary_end_sec if primary_end_sec is not None else global_end_sec
    )
    local_primary_end = max(
        0.0, min(float(clip_duration_sec), primary_end - float(global_start_sec))
    )
    previous_state_rule = (
        "\nTRANG THAI CUOI CLIP TRUOC (chi la du lieu noi mach, clip hien tai van la bang chung):\n"
        + json.dumps(previous_end_state, ensure_ascii=False, separators=(",", ":"))
        + "\n- Doi chieu start_state voi trang thai nay; neu hanh dong dang tiep dien, bat buoc danh dau "
          "continues_from_previous=true.\n"
        if isinstance(previous_end_state, dict) and previous_end_state else ""
    )
    return f"""Xem toan bo clip ngan dinh kem. Day la doan {int(chunk_number)}/{int(chunk_count)}
tu {float(global_start_sec):.2f}s den {float(global_end_sec):.2f}s cua video goc.

Chi trich xuat bang chung cot truyen nhin thay/nghe thay trong clip; CHUA viet loi TTS va CHUA loai
su kien theo thoi luong recap. Day la luot THU THAP RONG; viec chon canh se lam sau khi da co toan phim.
- Muc tieu khoang {int(candidate_count)} bien co/hinh anh de GHI NHAN, tang dan theo thoi gian; day la
  moc tham khao toi thieu cho doan binh thuong, KHONG PHAI gioi han output va chua phai chon canh recap.
  Doan yen binh duoc tra it hon; doan hanh dong day dac duoc tra nhieu hon. Khong chen canh phu de du so luong.
- Phan chinh cua clip ket thuc tai giay cuc bo {local_primary_end:.2f}; phan sau do la vung du phong de xem
  cho het su kien dang do. Van phai ghi nhan moi su kien moi trong vung du phong, khong dung o moc phan chinh.
- Neu su kien bat dau truoc moc phan chinh nhung chua co ket qua, dat `continues_to_next=true`.
  Neu dau clip dang tiep noi mot su kien da bat dau o clip truoc, dat `continues_from_previous=true`.
- `completion_status` chi duoc complete/open/uncertain. Chi dung complete khi da thay ket qua, doi muc tieu,
  doi tuyen truyen hoac chuyen boi canh ro rang; khong tu ket luan vi clip sap het.
- Neu it bien co lon, dung cac hanh dong, phan ung, thay doi dia diem hoặc cau noi quan trong
  co that lam bang chung phu; khong tach gia mot hanh dong thanh nhieu event trung lap.
- Timestamp phai la thoi gian CUC BO trong clip, tu 0 den {float(clip_duration_sec):.2f}s.
- Uu tien nguyen nhan, hanh dong, thay doi, hau qua, buoc ngoat, cao trao va ket cuc.
- Bo credit, outro, phong canh va hanh dong lap khong lam thay doi cau chuyen.
- Moi event phai tao it nhat mot thay doi moi ve muc tieu, nguy co, quan he, dia diem quan trong,
  vat pham quan trong hoac ket qua. Canh chi lap lai cung trang thai phai dung chung repeat_group
  hoac bo, khong viet lai bang tu khac de tao event moi.
- Ngoai climax/final action, gom chuoi hanh dong cung mot muc tieu thanh mot event co cause/change/
  consequence ro rang. Chi tach tung cu danh, ne tranh, phan don trong chuoi doi dau chinh.
- Khong suy dien noi tam hay bia loi thoai.
- Nhan vat dung ID C1, C2... nhat quan trong rieng clip; ghi ten/mo ta de luot text sau doi chieu.
- Neu chua nghe thay ten hoac chua co bang chung chac chan, KHONG doan ten rieng. De `name` rong,
  dat `identity_status="unknown"`, tao `descriptive_alias` TIENG VIET ngan gon nhu "quai vat tuong gom"
  hoac "ke deo mat na", va ghi dac diem nhin thay trong `visual_signature` chi de doi chieu noi bo.
- `identity_confidence` tu 0 den 1: chi dat tu 0.90 khi ten duoc goi ro trong clip hoac co bang chung truc tiep.
- Tat ca event, visual_detail, cause, change va consequence CHI DUOC viet bang TIENG VIET co dau.
  Tuyet doi khong viet cau tieng Anh hoac tron tieng Anh vao cau tieng Viet; chi ten rieng duoc giu nguyen.
- C1, C2... chi duoc xuat hien trong truong id/characters; tuyet doi khong chen ma C vao cau van.
- Trong cau van, dung ten nhan vat neu biet; neu chua biet ten thi dung mo ta tieng Viet ngan gon.
- Moi event co visual_detail cu the va importance main/bridge.
- Lay SU KIEN VA DANH MUC SHOT trong CUNG luot xem nay. Co gang lay visual_shot chinh xac cho moi event.
  Neu da quan sat chac chan su kien nhung chua dinh vi shot ngan, GIU event, de visual_shots=[] va
  visual_grounding_missing=true. Thieu shot KHONG co nghia su kien khong co that; se chon hinh sau loi ke.
- Mot event la mot thay doi cu the, thuong khong qua 90 giay. Tom tat dai nhieu phut chi de trong
  story_context, khong dua vao events. Chi dua vao event_review_candidates khi CHUA CHAC SU KIEN co that,
  khong phai chi vi thieu shot. Khong doan timestamp tu transcript hay tri nho phim.
- Moi event bat buoc co start_seconds/end_seconds bao tron dung bien co. `visual_shots` la cac BEAT HINH
  co y nghia ben trong event (hanh dong, phan ung, phat hien, thay doi hoac ket qua), khong phai shot render.
  Do dai beat LINH HOAT theo hanh dong, thuong 1.0-8.0 giay va toi da 15 giay; khong ep 2-3 giay,
  khong ep so luong co dinh. Neu chua chac moc shot thi giu vung su kien da quan sat va de visual_shots=[],
  tuyet doi khong chia deu thoi gian hay doan frame de lap cho trong.
- story_role chi duoc development, climax, resolution hoac ending.
- Moi event ghi `local_event_id` duy nhat trong clip (L001, L002...), `narrative_function` va
  `depends_on_local`. narrative_function chi duoc setup, inciting_incident, development, turning_point,
  false_victory, reversal, climax, resolution, ending. Neu event sau khong the hieu khi thieu event truoc,
  phai ghi ID event truoc vao depends_on_local.
- Moi event ghi `state_before`, `state_after`, `introduces` va `event_kind` de luot chon toan cuc khong
  cat roi nguyen nhan. event_kind chi duoc discovery, decision, preparation, action, transformation,
  attack, defense, reversal, reveal, consequence, resolution, explanation, travel hoac reaction.
- Voi mot chuoi hanh dong, tach rieng cac thay doi that: phat hien -> chuan bi/bien doi -> ra don ->
  doi thu phan ung/xuat hien -> dao chieu -> ket qua. Khong tom tat ca chuoi thanh mot event chung.
- `repeat_group` la nhan ngan giong nhau cho cac event chi lap lai CUNG mot trang thai/ket qua trong clip;
  de rong neu event them thong tin, hanh dong hoac ket qua moi.
- Moi event phai tach nghia thanh `subject`, `action`, `object`, `result`. Day la du lieu cho backend
  chong lap y, khong phai cau van. `fact_key` la nhan ASCII ngan theo mau subject_action_object_result;
  neu hai event chi dien dat lai CUNG mot thay doi thi bat buoc dung cung fact_key va repeat_group.
- Khong dung fact_key chung cho cac buoc that su khac nhau nhu chuan bi, tan cong, dao chieu va ket qua.
- `event_cluster` la ID ngan dung chung cho TOAN BO mot chuoi bien co khong duoc cat roi, vi du:
  phat hien nguoi bi bat -> bien doi -> danh tay sai -> lo quan kieu -> quai vat xuat hien. Khong dung
  event_cluster cho di chuyen, an uong, phong canh, phan ung lap hoac ca clip mot cach chung chung.
- Moc {int(candidate_count)} la ngan sach cho CAC EVENT THONG THUONG, khong phai ly do cat ngang cao trao.
  Neu chuoi chinh dai, giu du moi thay doi that cua chuoi va bo event phu de tong output van gon.
- Khong duoc tra mot event chi noi ket qua nhu "giai cuu thanh cong" neu clip con hien thi cac hanh dong
  truc tiep tao ra ket qua do. Ket qua chi hop le khi cac buoc quan trong truoc no cung co event rieng.
- Khong duoc bo chuoi mo dau neu no gioi thieu nhan vat, bien co pha vo trang thai ban dau, ngoi lang/gia dinh
  bi tan cong, hoac dua nhan vat sang thoi gian/dia diem khac.
{ending_rule}
{outline_rule}
{registry_rule}
{previous_state_rule}

Chi tra JSON hop le, khong Markdown:
{{"start_state":{{"summary":"trang thai luc clip bat dau","active_characters":["C1"],"time_context":""}},
"end_state":{{"summary":"trang thai khi clip ket thuc","active_characters":["C1"],"unresolved":""}},
"characters":[{{"id":"C1","name":"","descriptive_alias":"kẻ đeo mặt nạ","description":"","role":"",
"visual_signature":"","identity_status":"unknown","identity_confidence":0.0}}],
"events":[{{"local_event_id":"L001","start_seconds":1.0,"end_seconds":4.0,"event":"...",
"visual_detail":"...","cause":"...","change":"...","consequence":"...",
"visual_shots":[{{"start_seconds":1.2,"end_seconds":3.8,"description":"hanh dong nhin thay truc tiep"}}],
"subject":"","action":"","object":"","result":"","fact_key":"",
"state_before":"...","state_after":"...","introduces":"","event_kind":"action","repeat_group":"",
  "event_cluster":"CLIMAX_01","parent_cluster_id":"RESCUE_TRIBE",
  "continues_from_previous":false,"continues_to_next":false,"completion_status":"complete",
"characters":["C1"],"temporal_relation":"continuation","narrative_function":"development",
"depends_on_local":[],"required_for":"",
"importance":"main","story_role":"development"}}]}}
"""


def build_full_video_outline_prompt(
    video_duration_sec, target_words, style="dramatic", custom_prompt="",
    transcript_text="",
):
    """Ask one full-video pass for the final duration-aware recap scene map."""
    extras = str(custom_prompt or "").strip()
    target_minutes = max(0.1, float(target_words) / 150.0)
    # This pass only establishes global chapters and identities. Detailed event
    # IDs are collected once from shorter chapter clips in the next pass.
    suggested_beats = min(
        20, max(6, math.ceil(float(video_duration_sec) / (12 * 60)))
    )
    transcript = str(transcript_text or "").strip()
    return f"""Bạn là một biên kịch recap chuyên nghiệp. Hãy xem TOÀN BỘ video đính kèm từ đầu
đến kết thúc thật để lập BẢN ĐỒ CHƯƠNG cho phim dài {float(video_duration_sec):.2f} giây.
Đây là lượt hiểu cốt truyện tổng quát; CHƯA viết TTS và CHƯA liệt kê mọi event nhỏ.
Video có thể đã bỏ audio để tiết kiệm token. Phần LỜI THOẠI WHISPER bên dưới có timestamp toàn cục;
dùng nó để hiểu lời nói, còn hình ảnh trong video dùng để xác nhận hành động/nhân vật. Nếu hai nguồn
mâu thuẫn, không đoán; hạ confidence và chỉ ghi điều cả hai nguồn hỗ trợ.

MỤC TIÊU PHÂN CẢNH:
- `story_beats` ở lượt này là các CHƯƠNG LỚN liên tiếp, mục tiêu khoảng {suggested_beats} chương.
- Các chương phải phủ từ mở đầu thật tới kết thúc thật. Mỗi chương thường dài 8-15 phút; không gom cả phim
  thành vài beat rời rạc và không bỏ khoảng thời gian dài chỉ vì ít hành động.
- Mỗi chương ghi trạng thái đầu, mục tiêu/xung đột, thay đổi chính và trạng thái cuối để lượt clip sau tìm event ID.
- Giữ đủ mở đầu cần thiết, biến cố khởi phát, mục tiêu, xung đột, bước ngoặt, cao trào,
  hậu quả, giải quyết và kết thúc thật để người chưa xem vẫn hiểu câu chuyện.
- Không được bỏ một thông tin ngắn nếu nó làm thay đổi hoàn cảnh, mục tiêu, nguy hiểm hoặc kết quả,
  ví dụ ngôi làng bị phá hủy, bí mật được phát hiện hoặc một nhân vật thất bại.
- Gộp hoặc bỏ cảnh sinh hoạt, bắt cá, nấu ăn, di chuyển, phong cảnh, chờ đợi và hành động lặp
  nếu bỏ chúng không làm người xem hiểu sai câu chuyện. Không chọn cảnh chỉ vì nó dài hoặc nhiều chuyển động.
- Với chương hành động/cao trào, chỉ khoanh đúng toàn bộ vùng thời gian của chuỗi đối đầu và ghi các mốc lớn.
  Lượt xem clip sau sẽ tách chi tiết từng cú đánh, né tránh, phản đòn, đảo chiều và kết quả.
- Đánh dấu `is_final_action=true` cho các beat thuộc CHUỖI HÀNH ĐỘNG CHÍNH CUỐI CÙNG, tính từ lúc đối đầu
  thực sự bắt đầu tới hành động quyết định và kết quả trực tiếp. Không nhầm cảnh hành động muộn với phần kết.
- Phần kết chỉ giữ MỘT beat ngắn về kết quả cuối/trạng thái cuối; không kể lại toàn bộ câu chuyện, bài học,
  ăn mừng dài hoặc credit.
- Giữa hai beat, nếu video thật sự có cảnh nối, ghi `bridge_from_previous` gồm timestamp, các sự thật nhìn thấy
  và quan hệ. Không có bằng chứng thì để null; không tự bịa việc lên đường, lần dấu, gặp nhau hay quyết định.
- Trước khi trả JSON, tự kiểm tra: nếu bỏ beat nào khiến mất nguyên nhân, mục tiêu, bước ngoặt
  hoặc kết quả thì thêm lại; nếu beat không tạo thông tin mới thì gộp hoặc bỏ.

MỨC ĐỘ KỂ:
- `detailed`: cao trào, chiến đấu, bước ngoặt hoặc biến cố lớn; được dành nhiều lời và nhiều shot.
- `normal`: diễn biến cần thiết để hiểu nguyên nhân và tiến trình.
- `brief`: bối cảnh/cảnh nối cần giữ nhưng chỉ kể rất ngắn.
- Thời lượng recap là ràng buộc cao nhất: recap ngắn phải cắt `brief` trước; `detailed` được nhiều lời hơn,
  còn `brief` chỉ làm bối cảnh hoặc câu nối ngắn.

NHÂN VẬT:
- Tạo `characters` như sổ nhân vật cố định P01, P02... cho toàn video.
- Dựa vào khuôn mặt, tóc, trang phục, giọng nói, tên được gọi và vai trò xuyên cảnh.
- Không tạo ID mới chỉ vì đổi góc quay, trang phục hoặc biến hình. Không gộp hai người chỉ vì giống nhau.
- Chưa chắc tên thì để `name` rỗng, dùng `neutral_alias` tiếng Việt và hạ `confidence`; tuyệt đối không bịa tên.
- Lập hồ sơ nhân vật GỐC cho toàn phim: mỗi người một ID P cố định, không dùng lại ID cho người khác.
  Ghi ít nhất hai đặc điểm nhận dạng bền vững trong stable_features; tách quần áo/biến đổi ngoại hình
  vào appearance_variants. Ghi reference_seconds là các mốc thấy rõ nhân vật, không dùng cảnh intro.
  Gemini là nguồn tên chính: dùng tên nhận diện chắc chắn từ hình, ngữ cảnh hoặc âm thanh của nguồn.
  Whisper chỉ tham khảo; không đổi tên đã nhận diện chỉ vì phiên âm sai. Chưa chắc thì dùng biệt danh.
  name_evidence là object gồm kind=dialogue/onscreen_text/unknown, quote nguyên văn,
  start_seconds/end_seconds toàn cục, shot_ids (nếu có), association=direct_address/self_introduction/onscreen_label/third_person/uncertain.
  Tên chỉ được nhắc đến (third_person) chưa chứng minh người trong hình mang tên đó.
  Trong summary/cause/outcome của story_beats, tham chiếu người bằng [CHAR=P01],
  không chép tên chưa xác nhận vào mạch truyện; backend sẽ hiển thị bằng sổ nhân vật.
  Ghi face_features, body_features, distinctive_features khi thấy rõ, không bịa để đủ trường.
  neutral_alias luôn là mô tả ngắn phân biệt người đó (ví dụ "cậu bé đeo kính"), không lặp lại tên riêng.
- Ghi non_story_ranges chính xác cho logo, opening, credits, ending montage/outro. Những cảnh bay lượn
  trong nhạc kết/credits không phải chuyến phiêu lưu mới. Chỉ đánh dấu khi thấy bằng chứng trong video,
  không cắt tùy tiện theo phần trăm thời lượng; cảnh mở đầu có diễn biến thật vẫn thuộc cốt truyện.

HÌNH ẢNH:
- Mỗi chương cung cấp 1-4 `visual_regions` rộng để định hướng việc cắt clip ở lượt sau.
- Đây là VÙNG NGUỒN, backend sẽ tự chia thành shot mục tiêu 2-3 giây; không cần liệt kê hàng trăm shot.
- Vùng phải đúng nhân vật/hành động, không lấy logo, credit hoặc hình phụ không liên quan.
- Timestamp dùng giây toàn cục, tăng theo thời gian và nằm trong video dài {float(video_duration_sec):.2f}s.
- Mọi câu mô tả phải là tiếng Việt có dấu, ngoại trừ tên riêng và các giá trị schema cố định.
- Hướng kể sau này: {style_storytelling_prompt(style)}
{('- Yeu cau them: ' + extras) if extras else ''}

LỜI THOẠI WHISPER TOÀN PHIM (có thể rỗng khi phải fallback sang audio gốc):
{transcript or 'không có; hãy dùng audio trong video nếu hiện diện'}

Chỉ trả JSON hợp lệ:
{{"story_contract":{{"premise":"","protagonist_goal":"","central_conflict":"","ending_meaning":""}},
"non_story_ranges":[{{"start_seconds":0,"end_seconds":1,"kind":"logo|opening|credits|outro|ending_montage","confidence":0.0}}],
"characters":[{{"id":"P01","name":"","aliases":[],"neutral_alias":"","visual_signature":"",
"stable_features":[],"face_features":[],"body_features":[],"distinctive_features":[],"appearance_variants":[],"reference_seconds":[],"name_evidence":{{"kind":"unknown","quote":"","start_seconds":0,"end_seconds":0,"shot_ids":[],"association":"uncertain"}},"role":"","confidence":0.0}}],
"story_beats":[{{"beat_id":"GB001","start_seconds":0,"end_seconds":10,
"summary":"","cause":"","state_before":"","state_after":"","outcome":"","characters":[""],
"narration_facts":["sự thật 1 theo thứ tự","sự thật 2 theo thứ tự"],
"depends_on":[],"temporal_relation":"continuation","confidence":0.9,
"narrative_function":"setup","story_role":"development","detail_level":"normal","is_final_action":false,
"bridge_from_previous":{{"start_seconds":0,"end_seconds":0,"relation":"location_change",
"facts":["sự thật nhìn thấy"],"confidence":0.9}},
"visual_regions":[{{"start_seconds":0,"end_seconds":6,"description":""}}]}}]}}
"""


def normalize_full_video_outline(text, video_duration_sec):
    """Validate a small optional map produced by the full-video overview pass."""
    payload = _json_object(text)
    duration = max(1.0, float(video_duration_sec))
    allowed_functions = {
        "setup", "inciting_incident", "development", "turning_point",
        "false_victory", "reversal", "climax", "resolution", "ending",
    }
    beats = []
    raw_id_by_position = []
    for index, item in enumerate(payload.get("story_beats") or [], start=1):
        if not isinstance(item, dict):
            continue
        try:
            start = min(duration, max(0.0, float(item.get("start_seconds", 0))))
            end = min(duration, max(start + 0.01, float(item.get("end_seconds", start))))
        except (TypeError, ValueError):
            continue
        summary = re.sub(r"\s+", " ", str(item.get("summary", ""))).strip()
        if not summary or end <= start:
            continue
        function = str(item.get("narrative_function", "development")).strip().lower()
        if function not in allowed_functions:
            function = "development"
        role = str(item.get("story_role", "development")).strip().lower()
        if role not in {"development", "climax", "resolution", "ending"}:
            role = "development"
        importance = str(item.get("importance", "main")).strip().lower()
        if importance not in {"critical", "main", "support"}:
            importance = "main"
        raw_id = str(item.get("beat_id", "")).strip().upper()
        raw_id_by_position.append(raw_id or f"__ROW_{index}")
        relation = str(item.get("temporal_relation", "unknown")).strip().lower()
        if relation not in {
            "continuation", "time_jump", "flashback", "flashforward",
            "parallel", "return", "unknown",
        }:
            relation = "unknown"
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.75))))
        except (TypeError, ValueError):
            confidence = 0.75
        detail_level = str(item.get("detail_level", "")).strip().lower()
        if detail_level not in {"detailed", "normal", "brief"}:
            detail_level = ""
        is_final_action = bool(item.get("is_final_action", False))
        bridge = None
        raw_bridge = item.get("bridge_from_previous")
        if isinstance(raw_bridge, dict):
            try:
                bridge_start = max(0.0, float(raw_bridge.get("start_seconds")))
                bridge_end = min(duration, float(raw_bridge.get("end_seconds")))
                bridge_confidence = min(
                    1.0, max(0.0, float(raw_bridge.get("confidence", 0.0)))
                )
            except (TypeError, ValueError):
                bridge_start = bridge_end = bridge_confidence = 0.0
            bridge_relation = str(
                raw_bridge.get("relation", "unknown")
            ).strip().lower()
            if bridge_relation not in {
                "continuation", "cause", "consequence", "contrast",
                "time_jump", "location_change", "parallel", "return",
            }:
                bridge_relation = "unknown"
            bridge_facts = [
                re.sub(r"\s+", " ", str(value)).strip()[:180]
                for value in (raw_bridge.get("facts") or [])
                if str(value).strip()
            ][:4]
            if (
                bridge_end > bridge_start
                and bridge_confidence >= 0.65
                and bridge_facts
                and bridge_relation != "unknown"
            ):
                bridge = {
                    "start_seconds": bridge_start,
                    "end_seconds": bridge_end,
                    "relation": bridge_relation,
                    "facts": bridge_facts,
                    "confidence": bridge_confidence,
                }
        visual_regions = []
        for raw_region in (item.get("visual_regions") or []):
            if not isinstance(raw_region, dict):
                continue
            try:
                region_start = max(0.0, float(raw_region.get("start_seconds")))
                region_end = min(duration, float(raw_region.get("end_seconds")))
            except (TypeError, ValueError):
                continue
            if region_end <= region_start:
                continue
            visual_regions.append({
                "start_seconds": region_start,
                "end_seconds": region_end,
                "description": re.sub(
                    r"\s+", " ", str(raw_region.get("description", ""))
                ).strip()[:180],
            })
        visual_regions.sort(key=lambda row: (
            row["start_seconds"], row["end_seconds"]
        ))
        beats.append({
            "_raw_beat_id": raw_id_by_position[-1],
            "beat_id": f"GB{len(beats) + 1:03d}",
            "start_seconds": start,
            "end_seconds": end,
            "summary": summary[:320],
            "cause": re.sub(r"\s+", " ", str(item.get("cause", ""))).strip()[:160],
            "state_before": re.sub(
                r"\s+", " ", str(item.get("state_before", ""))
            ).strip()[:160],
            "state_after": re.sub(
                r"\s+", " ", str(item.get("state_after", ""))
            ).strip()[:160],
            "outcome": re.sub(r"\s+", " ", str(item.get("outcome", ""))).strip()[:160],
            "narration_facts": [
                re.sub(r"\s+", " ", str(value)).strip()[:220]
                for value in (item.get("narration_facts") or [])
                if str(value).strip()
            ][:24],
            "characters": [
                str(value).strip() for value in (item.get("characters") or [])
                if str(value).strip()
            ][:8],
            "narrative_function": function,
            "story_role": role,
            "importance": importance,
            "detail_level": detail_level,
            "is_final_action": is_final_action,
            "bridge_from_previous": bridge,
            "visual_regions": visual_regions[:12],
            "depends_on": [
                str(value).strip().upper()
                for value in (item.get("depends_on") or [])
                if str(value).strip()
            ][:8],
            "temporal_relation": relation,
            "confidence": confidence,
        })
    beats.sort(key=lambda row: (row["start_seconds"], row["end_seconds"]))
    if not beats:
        raise ValueError("Full-video outline khong co story beat hop le.")
    ending_positions = [
        index for index, beat in enumerate(beats)
        if beat.get("story_role") == "ending"
        or beat.get("narrative_function") == "ending"
    ]
    ending_inference = {
        "status": "model_labeled" if ending_positions else "missing",
        "beat_id": beats[ending_positions[-1]]["beat_id"] if ending_positions else "",
        "reason": "",
    }
    if not ending_positions:
        # A frequent harmless model error is labeling the real final state as
        # resolution/development. Promote it only when objective safeguards are
        # present; this is classification of existing evidence, not invention.
        for candidate_index in range(len(beats) - 1, max(-1, len(beats) - 4), -1):
            candidate = beats[candidate_index]
            near_real_end = float(candidate.get("end_seconds", 0)) >= duration * 0.88
            has_terminal_state = bool(
                str(candidate.get("outcome", "")).strip()
                or str(candidate.get("state_after", "")).strip()
            )
            still_active_climax = bool(
                candidate.get("is_final_action")
                or candidate.get("story_role") == "climax"
                or candidate.get("narrative_function") == "climax"
            )
            if near_real_end and has_terminal_state and not still_active_climax:
                candidate["story_role"] = "ending"
                candidate["narrative_function"] = "ending"
                candidate["detail_level"] = "brief"
                candidate["is_final_action"] = False
                candidate["ending_inferred_by_backend"] = True
                ending_positions = [candidate_index]
                ending_inference = {
                    "status": "backend_inferred",
                    "beat_id": candidate["beat_id"],
                    "reason": "near_video_end_with_outcome_or_final_state",
                }
                break
    for index in ending_positions[:-1]:
        beats[index]["story_role"] = "resolution"
        beats[index]["narrative_function"] = "resolution"
        if beats[index].get("detail_level") == "brief":
            beats[index]["detail_level"] = "normal"
    if len(beats) > 150:
        critical = [
            row for row in beats
            if row["importance"] == "critical"
            or row["narrative_function"] in {
                "inciting_incident", "turning_point", "false_victory",
                "reversal", "climax", "resolution", "ending",
            }
        ]
        selected = {row["beat_id"]: row for row in critical[:150]}
        for bucket in range(150):
            if len(selected) >= 150:
                break
            position = min(len(beats) - 1, math.floor(bucket * len(beats) / 150))
            selected.setdefault(beats[position]["beat_id"], beats[position])
        beats = sorted(
            selected.values(), key=lambda row: (row["start_seconds"], row["end_seconds"])
        )[:150]
    for index, beat in enumerate(beats):
        bridge = beat.get("bridge_from_previous")
        if index == 0 or not bridge:
            beat["bridge_from_previous"] = None
            continue
        previous = beats[index - 1]
        if not (
            float(previous["start_seconds"]) <= float(bridge["start_seconds"])
            < float(bridge["end_seconds"]) <= float(beat["end_seconds"])
        ):
            beat["bridge_from_previous"] = None
    id_map = {}
    for index, beat in enumerate(beats, start=1):
        new_id = f"GB{index:03d}"
        id_map[beat.pop("_raw_beat_id", new_id)] = new_id
        beat["beat_id"] = new_id
        if beat.get("bridge_from_previous"):
            beat["bridge_from_previous"]["bridge_id"] = f"BR{index:03d}"
    for beat in beats:
        beat["depends_on"] = [
            id_map[value] for value in beat.get("depends_on", [])
            if value in id_map and id_map[value] != beat["beat_id"]
        ]
    final_ending = next((
        beat for beat in reversed(beats)
        if beat.get("story_role") == "ending"
        or beat.get("narrative_function") == "ending"
    ), None)
    ending_inference["beat_id"] = (
        final_ending.get("beat_id", "") if final_ending else ""
    )
    contract = payload.get("story_contract")
    characters = [
        copy.deepcopy(row) for row in (payload.get("characters") or [])
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    ]
    from situation_recap import normalize_non_story_ranges
    return {
        "story_contract": contract if isinstance(contract, dict) else {},
        "non_story_ranges": normalize_non_story_ranges(payload.get('non_story_ranges'), duration),
        "characters": characters,
        "story_beats": beats,
        "ending_inference": ending_inference,
    }


def extract_complete_recap_scenes(partial_text):
    """Recover only fully closed scene objects from an interrupted JSON stream."""
    source = str(partial_text or "")
    decoder = json.JSONDecoder(strict=False)
    recovered = []
    seen = set()
    for match in re.finditer(r"\{", source):
        try:
            value, _ = decoder.raw_decode(source, match.start())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        tts = re.sub(r"\s+", " ", str(value.get("tts", ""))).strip()
        beat_id = str(value.get("story_beat_id", "")).strip().upper()
        if not tts or not beat_id or "start" not in value or "end" not in value:
            continue
        try:
            start = _time_seconds(value.get("start"))
            end = _time_seconds(value.get("end"))
        except ValueError:
            continue
        if end <= start:
            continue
        key = (beat_id, tts.casefold())
        if key in seen:
            continue
        seen.add(key)
        row = copy.deepcopy(value)
        row["story_beat_id"] = beat_id
        row["tts"] = tts
        row["start"] = _format_time(start)
        row["end"] = _format_time(end)
        recovered.append(row)
    recovered.sort(key=lambda row: (_time_seconds(row["start"]), row["story_beat_id"]))
    return recovered


def story_outline_for_window(outline, global_start_sec, global_end_sec):
    """Return overlapping full-story beats plus one beat of edge context."""
    beats = list((outline or {}).get("story_beats") or [])
    if not beats:
        return []
    start = float(global_start_sec)
    end = float(global_end_sec)
    indices = [
        index for index, beat in enumerate(beats)
        if float(beat.get("end_seconds", 0)) > start
        and float(beat.get("start_seconds", 0)) < end
    ]
    if not indices:
        nearest = min(
            range(len(beats)),
            key=lambda index: abs(float(beats[index].get("start_seconds", 0)) - start),
        )
        indices = [nearest]
    first = max(0, min(indices) - 1)
    last = min(len(beats), max(indices) + 2)
    return copy.deepcopy(beats[first:last])


def plan_adaptive_story_boundaries(
    outline, video_duration_sec, target_chunk_seconds=600.0,
    min_chunk_seconds=300.0, max_chunk_seconds=720.0,
    strict_max_chunk=False,
):
    """Place the same approximate number of cuts at natural story boundaries.

    The full-video outline is advisory, so malformed or sparse maps fall back
    to equal chunks.  Structural beats and their direct dependencies form
    no-cut intervals; no extra Gemini request is needed.
    """
    duration = max(0.01, float(video_duration_sec))
    target = max(60.0, float(target_chunk_seconds))
    count = max(1, math.ceil(duration / target))
    equal = [duration * index / count for index in range(count + 1)]
    beats = sorted(
        [copy.deepcopy(row) for row in (outline or {}).get("story_beats", [])
         if isinstance(row, dict)],
        key=lambda row: (
            float(row.get("start_seconds", 0.0)),
            float(row.get("end_seconds", 0.0)),
        ),
    )
    if count == 1 or len(beats) < 2:
        return equal

    by_id = {
        str(row.get("beat_id", "")).strip().upper(): row
        for row in beats if str(row.get("beat_id", "")).strip()
    }
    structural = {
        "inciting_incident", "turning_point", "false_victory", "reversal",
        "climax", "resolution",
    }
    protected_intervals = []
    for beat in beats:
        function = str(beat.get("narrative_function", "")).strip().lower()
        role = str(beat.get("story_role", "")).strip().lower()
        importance = str(beat.get("importance", "")).strip().lower()
        if not (
            function in structural or role in {"climax", "resolution"}
            or importance == "critical"
        ):
            continue
        members = [beat]
        for dependency in beat.get("depends_on") or []:
            dependency_beat = by_id.get(str(dependency).strip().upper())
            # A distant setup can be causally required without being the same
            # continuous action sequence. Only lock nearby dependencies.
            if dependency_beat and (
                float(beat.get("start_seconds", 0.0))
                - float(dependency_beat.get("end_seconds", 0.0)) <= 90.0
            ):
                members.append(dependency_beat)
        start = min(float(row.get("start_seconds", 0.0)) for row in members)
        end = max(float(row.get("end_seconds", start)) for row in members)
        protected_intervals.append((max(0.0, start), min(duration, end)))

    # Consecutive structural beats separated by only a short pause are one
    # continuous climax for cutting purposes.
    merged_intervals = []
    for start, end in sorted(protected_intervals):
        if (
            merged_intervals
            and start - merged_intervals[-1][1] <= 30.0
            and end - merged_intervals[-1][0] <= float(max_chunk_seconds)
        ):
            merged_intervals[-1] = (
                merged_intervals[-1][0], max(merged_intervals[-1][1], end)
            )
        else:
            merged_intervals.append((start, end))
    protected_intervals = merged_intervals

    natural_points = {0.0, duration}
    for left, right in zip(beats, beats[1:]):
        left_end = min(duration, float(left.get("end_seconds", 0.0)))
        right_start = max(0.0, float(right.get("start_seconds", left_end)))
        natural_points.add(
            (left_end + right_start) / 2.0
            if right_start >= left_end else left_end
        )
    natural_points.update(
        min(duration, max(0.0, float(row.get(key, 0.0))))
        for row in beats for key in ("start_seconds", "end_seconds")
    )
    natural_points = sorted(natural_points)

    boundaries = [0.0]
    minimum = min(float(min_chunk_seconds), duration / count * 0.75)
    # Keep the hard semantic window bounded. The caller uses 12-18 minutes;
    # the extra three-minute physical lookahead is added only by the clipper.
    maximum = max(
        float(max_chunk_seconds),
        duration / count * (1.05 if strict_max_chunk else 1.35),
    )
    for cut_index in range(1, count):
        previous = boundaries[-1]
        remaining = count - cut_index
        ideal = duration * cut_index / count
        low = max(previous + minimum, duration - remaining * maximum)
        high = min(previous + maximum, duration - remaining * minimum)
        candidates = [point for point in natural_points if low <= point <= high]
        if not candidates:
            boundaries.append(min(high, max(low, ideal)))
            continue

        def cut_cost(point):
            protected_penalty = sum(
                1 for start, end in protected_intervals
                if start + 0.5 < point < end - 0.5
            )
            ordinary_penalty = sum(
                1 for beat in beats
                if float(beat.get("start_seconds", 0.0)) + 0.5 < point
                < float(beat.get("end_seconds", 0.0)) - 0.5
            )
            return (
                protected_penalty * duration * 10.0
                + ordinary_penalty * duration
                + abs(point - ideal)
            )

        boundaries.append(min(candidates, key=cut_cost))
    boundaries.append(duration)
    if any(right - left <= 0.01 for left, right in zip(boundaries, boundaries[1:])):
        return equal
    return boundaries


def normalize_chunk_evidence(
    text, global_start_sec, clip_duration_sec, chunk_number=None,
    primary_end_sec=None,
):
    """Convert one chunk's local evidence timestamps to the global timeline."""
    payload = _json_object(text)
    duration = max(0.01, float(clip_duration_sec))
    offset = max(0.0, float(global_start_sec))
    characters = [
        copy.deepcopy(item) for item in (payload.get("characters") or [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    character_labels = {}
    character_ids = {}
    for character_index, character in enumerate(characters, start=1):
        source_id = str(character.get("id", "")).strip().upper()
        character_id = (
            f"CH{int(chunk_number):03d}_{source_id}"
            if chunk_number is not None else source_id
        )
        name = str(character.get("name", "")).strip()
        description = str(
            character.get("description") or character.get("alias") or ""
        ).strip()
        try:
            confidence = float(
                character.get("identity_confidence", character.get("confidence", 0.0))
            )
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        status = str(character.get("identity_status", "")).strip().lower()
        if status not in {"confirmed", "probable", "unknown"}:
            status = "confirmed" if name and confidence >= 0.90 else (
                "probable" if name else "unknown"
            )
        character["source_id"] = source_id
        character["id"] = character_id
        character["proposed_name"] = str(character.get('proposed_name') or name)
        character["identity_status"] = status
        character["identity_confidence"] = confidence
        character["visual_signature"] = re.sub(
            r"\s+", " ", str(
                character.get("visual_signature")
                or character.get("signature") or description
            )
        ).strip()
        character_ids[source_id] = character_id
        # Prose remains readable, but the event's structured character list
        # keeps the stable chunk-qualified identity for later reconciliation.
        character["descriptive_alias"] = _descriptive_character_alias(
            character, character_index
        )
        if not name and str(character.get("alias", "")).strip():
            character["descriptive_alias"] = re.sub(
                r"\s+", " ", str(character.get("alias", ""))
            ).strip()
        character_labels[source_id] = character["descriptive_alias"]
    events = []
    raw_event_id_map = {}
    for item in payload.get("events") or []:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start_seconds", item.get("start")))
            end = float(item.get("end_seconds", item.get("end")))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= start < end <= duration + 1.0):
            continue
        event = str(item.get("event", "")).strip()
        if not event:
            continue
        row = copy.deepcopy(item)
        for field in (
            "event", "visual_detail", "cause", "change", "consequence",
            "state_before", "state_after", "introduces", "subject",
            "action", "object", "result",
        ):
            row[field] = (str(row.get(field, "")) if row.get('identity_tokens_rendered') else
                          _replace_internal_character_ids(row.get(field, ""), character_labels))
            # Never let raw vision prose become a character name in narration.
            for character in characters:
                if row.get('identity_tokens_rendered'):
                    break
                alias = str(character.get("descriptive_alias", "")).strip()
                for raw_label in (
                    character.get("description"), character.get("visual_signature"),
                ):
                    raw_label = str(raw_label or "").strip()
                    if alias and raw_label:
                        row[field] = re.sub(
                            re.escape(raw_label), alias, row[field], flags=re.IGNORECASE
                        )
        row["start_seconds"] = offset + max(0.0, start)
        row["end_seconds"] = offset + min(duration, end)
        visual_sources = []
        for raw_shot in item.get("visual_shots") or []:
            if not isinstance(raw_shot, dict):
                continue
            try:
                shot_start = float(raw_shot.get("start_seconds"))
                shot_end = float(raw_shot.get("end_seconds"))
            except (TypeError, ValueError):
                continue
            shot_duration = shot_end - shot_start
            if not (
                0.0 <= shot_start < shot_end <= duration + 0.05
                and start - 0.05 <= shot_start
                and shot_end <= end + 0.05
                and shot_duration <= MAX_GROUNDED_VISUAL_BEAT_SECONDS
            ):
                continue
            description = _replace_internal_character_ids(
                raw_shot.get("description", ""), character_labels
            )
            visual_sources.append({
                "start_seconds": offset + shot_start,
                "end_seconds": offset + shot_end,
                "event": description or row.get("visual_detail") or event,
                "verified_visual_shot": True,
                **({"shot_id": raw_shot['shot_id'], "fact_ids": list(raw_shot.get('fact_ids') or [])}
                   if raw_shot.get('shot_id') else {}),
            })
        # A genuinely short event is itself a safe shot. Broad event windows
        # are never silently converted into a guessed frame range.
        if (
            not visual_sources and not row.get('grounded_evidence_version')
            and end - start <= MAX_GROUNDED_VISUAL_BEAT_SECONDS
        ):
            visual_sources.append({
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "event": row.get("visual_detail") or event,
                "verified_visual_shot": True,
            })
        row["visual_sources"] = visual_sources
        row["visual_grounding_missing"] = not bool(visual_sources)
        compact_who = item.get("who", item.get("characters", []))
        if not isinstance(compact_who, list):
            compact_who = [compact_who]
        row["characters"] = [
            (
                character_ids.get(
                    str(value).strip().upper(), str(value).strip().upper()
                )
                if chunk_number is not None else
                character_labels.get(str(value).strip(), str(value).strip())
            )
            for value in compact_who
            if str(value).strip()
        ]
        row["evidence_id"] = (
            f"EV{int(chunk_number):03d}_{len(events) + 1:04d}"
            if chunk_number is not None else f"EV{len(events) + 1:04d}"
        )
        for source in row["visual_sources"]:
            source["evidence_id"] = row["evidence_id"]
        for fact in row.get('grounded_facts') or []:
            if not fact.get('identity_tokens_rendered'):
                fact['text'] = _replace_internal_character_ids(fact.get('text', ''), character_labels)
            fact['source_event_id'] = row['evidence_id']
        local_event_id = str(
            item.get("local_event_id") or item.get("id")
            or f"L{len(events) + 1:03d}"
        ).strip().upper()
        row["local_event_id"] = local_event_id
        raw_event_id_map[local_event_id] = row["evidence_id"]
        temporal_relation = str(
            item.get("temporal_relation", "continuation")
        ).strip().lower()
        row["temporal_relation"] = (
            temporal_relation if temporal_relation in {
                "continuation", "time_jump", "flashback", "flashforward",
                "parallel", "return", "unknown",
            } else "unknown"
        )
        narrative_function = str(
            item.get("narrative_function", "development")
        ).strip().lower()
        row["narrative_function"] = (
            narrative_function if narrative_function in {
                "setup", "inciting_incident", "development", "turning_point",
                "false_victory", "reversal", "climax", "resolution", "ending",
            } else "development"
        )
        compact_role = str(
            item.get("story_role") or item.get("role") or "development"
        ).strip().lower()
        row["story_role"] = (
            compact_role if compact_role in {
                "development", "climax", "resolution", "ending"
            } else "development"
        )
        if row["story_role"] in {"climax", "resolution", "ending"}:
            row["narrative_function"] = row["story_role"]
        raw_dependencies = item.get("depends_on_local") or []
        if not isinstance(raw_dependencies, list):
            raw_dependencies = [raw_dependencies]
        row["depends_on_local"] = [
            str(value).strip().upper() for value in raw_dependencies
            if str(value).strip()
        ]
        row["required_for"] = re.sub(
            r"\s+", " ", str(item.get("required_for", ""))
        ).strip()
        event_kind = str(item.get("event_kind", "action")).strip().lower()
        row["event_kind"] = (
            event_kind if event_kind in {
                "discovery", "decision", "preparation", "action",
                "transformation", "attack", "defense", "reversal", "reveal",
                "consequence", "resolution", "explanation", "travel", "reaction",
            } else "action"
        )
        row["repeat_group"] = re.sub(
            r"[^a-zA-Z0-9_-]+", "-", str(item.get("repeat_group", "")).strip()
        ).strip("-")[:80]
        row["fact_key"] = re.sub(
            r"[^a-zA-Z0-9_-]+", "-", str(item.get("fact_key", "")).strip()
        ).strip("-")[:120].casefold()
        raw_cluster = re.sub(
            r"[^a-zA-Z0-9_-]+", "-", str(
                item.get("event_cluster") or item.get("cluster") or ""
            ).strip()
        ).strip("-")[:80]
        row["event_cluster"] = (
            f"CH{int(chunk_number):03d}_{raw_cluster}"
            if raw_cluster and chunk_number is not None else raw_cluster
        )
        raw_parent_cluster = re.sub(
            r"[^a-zA-Z0-9_-]+", "-",
            str(item.get("parent_cluster_id", "")).strip(),
        ).strip("-")[:100]
        row["parent_cluster_id"] = raw_parent_cluster.casefold()
        row["continues_from_previous"] = bool(
            item.get("continues_from_previous", False)
        )
        row["continues_to_next"] = bool(item.get("continues_to_next", False))
        compact_importance = item.get("importance", "main")
        if isinstance(compact_importance, (int, float)):
            row["importance_score"] = max(0, min(3, int(compact_importance)))
            row["importance"] = (
                "main" if row["importance_score"] >= 2 else "bridge"
            )
        completion_status = str(
            item.get("completion_status", "complete")
        ).strip().lower()
        row["completion_status"] = (
            completion_status
            if completion_status in {"complete", "open", "uncertain"}
            else "uncertain"
        )
        if row["continues_to_next"]:
            row["completion_status"] = "open"
        events.append(row)
    if not events:
        raise ValueError("Gemini khong tra ve bang chung hop le cho doan video.")
    for event in events:
        event["depends_on"] = [
            raw_event_id_map[value] for value in event.pop("depends_on_local", [])
            if value in raw_event_id_map
            and raw_event_id_map[value] != event["evidence_id"]
        ]
    events.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))
    start_state = copy.deepcopy(payload.get("start_state"))
    end_state = copy.deepcopy(payload.get("end_state"))
    for state in (start_state, end_state):
        if not isinstance(state, dict):
            continue
        active = state.get("active_characters") or []
        if not isinstance(active, list):
            active = [active]
        state["active_characters"] = [
            character_ids.get(str(value).strip().upper(), str(value).strip().upper())
            for value in active if str(value).strip()
        ]
    normalized = {
        "chunk_number": int(chunk_number) if chunk_number is not None else None,
        "start_seconds": offset,
        "end_seconds": offset + duration,
        "primary_end_seconds": min(
            offset + duration,
            max(offset, float(
                primary_end_sec if primary_end_sec is not None
                else offset + duration
            )),
        ),
        "start_state": start_state if isinstance(start_state, dict) else {},
        "end_state": end_state if isinstance(end_state, dict) else {},
        "characters": characters,
        "events": events,
        "story_context": copy.deepcopy(payload.get("story_context") or []),
        "plot_gaps": copy.deepcopy(payload.get("plot_gaps") or []),
        "confirmed_non_story_shots": copy.deepcopy(payload.get("confirmed_non_story_shots") or []),
    }
    return separate_event_granularity([normalized], review_legacy=False)[0][0]


def _close_truncated_json(source):
    """Best-effort repair for a JSON response cut off near its end."""
    source = str(source or "").strip().replace("```json", "").replace("```", "")
    first_object = source.find("{")
    first_array = source.find("[")
    starts = [value for value in (first_object, first_array) if value >= 0]
    if not starts:
        raise ValueError("Gemini khong tra ve JSON de sua.")
    source = source[min(starts):]
    source = re.sub(r",\s*([}\]])", r"\1", source)
    stack = []
    in_string = False
    escaped = False
    for character in source:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}" and stack:
            expected = "[" if character == "]" else "{"
            if stack[-1] == expected:
                stack.pop()
    if in_string:
        source += '"'
    source = source.rstrip().rstrip(",")
    source += "".join("]" if value == "[" else "}" for value in reversed(stack))
    source = re.sub(r",\s*([}\]])", r"\1", source)
    return source


def _recover_chunk_payload(text, clip_duration_sec, global_start_sec=0.0):
    """Repair common Gemini JSON/timestamp mistakes without inventing actions."""
    try:
        payload = _json_object(text)
    except ValueError:
        repaired = _close_truncated_json(text)
        parsed = json.loads(repaired, strict=False)
        payload = {"events": parsed} if isinstance(parsed, list) else parsed
    if not isinstance(payload, dict):
        payload = {"events": []}

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raw_events = next((
            payload.get(name) for name in ("scenes", "timeline", "event_list")
            if isinstance(payload.get(name), list)
        ), [])
    if not raw_events and any(payload.get(name) for name in ("event", "summary", "description")):
        raw_events = [payload]

    duration = max(0.01, float(clip_duration_sec))
    offset = max(0.0, float(global_start_sec))
    repaired_events = []
    total = max(1, len(raw_events))
    for index, raw_event in enumerate(raw_events):
        item = {"event": raw_event} if isinstance(raw_event, str) else copy.deepcopy(raw_event)
        if not isinstance(item, dict):
            continue
        event_text = next((
            str(item.get(name, "")).strip()
            for name in ("event", "summary", "description", "action", "visual_detail")
            if str(item.get(name, "")).strip()
        ), "")
        if not event_text:
            continue
        item["event"] = event_text

        def timestamp(*names):
            for name in names:
                value = item.get(name)
                if value in (None, ""):
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    try:
                        return float(_time_seconds(value))
                    except (TypeError, ValueError):
                        continue
            return None

        start = timestamp("start_seconds", "start", "start_time", "time_start")
        end = timestamp("end_seconds", "end", "end_time", "time_end")
        # Gemini sometimes returns timestamps from the full video even though
        # the prompt requests local clip time.
        if offset > 0 and start is not None and start > duration and offset <= start <= offset + duration + 1:
            start -= offset
        if offset > 0 and end is not None and end > duration and offset <= end <= offset + duration + 1:
            end -= offset

        bin_start = duration * index / total
        bin_end = duration * (index + 1) / total
        if start is None:
            start = bin_start
            item["timestamp_repaired"] = True
        if end is None:
            end = max(start + min(30.0, max(1.0, duration / total)), bin_end)
            item["timestamp_repaired"] = True
        start = min(duration - 0.01, max(0.0, start))
        end = min(duration, max(start + 0.01, end))
        if end <= start:
            end = min(duration, start + max(0.01, min(30.0, duration / total)))
            item["timestamp_repaired"] = True
        if end <= start:
            start = max(0.0, duration - 0.01)
            end = duration
            item["timestamp_repaired"] = True
        item["start_seconds"] = start
        item["end_seconds"] = end
        item["recovery_confidence"] = 0.45 if item.get("timestamp_repaired") else 0.75
        repaired_events.append(item)
    payload["events"] = repaired_events
    return payload


def _chunk_candidate_score(candidate):
    events = list(candidate.get("events") or [])
    detail_fields = ("visual_detail", "cause", "change", "consequence", "subject", "action", "result")
    detail_score = sum(
        bool(str(event.get(field, "")).strip())
        for event in events for field in detail_fields
    )
    repaired_timestamps = sum(bool(event.get("timestamp_repaired")) for event in events)
    return len(events) * 100 + detail_score - repaired_timestamps * 2


def recover_chunk_evidence(
    responses, global_start_sec, clip_duration_sec, chunk_number=None,
    primary_end_sec=None,
):
    """Salvage the best evidence from invalid retries and never abort a video."""
    responses = list(responses or [])
    candidates = []
    for response in responses:
        try:
            payload = _recover_chunk_payload(
                response, clip_duration_sec, global_start_sec=global_start_sec,
            )
            if not payload.get("events"):
                continue
            normalized = normalize_chunk_evidence(
                json.dumps(payload, ensure_ascii=False),
                global_start_sec,
                clip_duration_sec,
                chunk_number=chunk_number,
                primary_end_sec=primary_end_sec,
            )
            candidates.append(normalized)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    if candidates:
        best = copy.deepcopy(max(candidates, key=_chunk_candidate_score))
        merged_events = list(best.get("events") or [])
        signatures = {
            (
                round(float(event.get("start_seconds", 0.0)), 1),
                round(float(event.get("end_seconds", 0.0)), 1),
                re.sub(r"\W+", " ", str(event.get("event", "")).casefold()).strip(),
            )
            for event in merged_events
        }

        def duplicates_existing(candidate):
            candidate_tokens = set(re.findall(
                r"\w+", str(candidate.get("event", "")).casefold()
            ))
            candidate_start = float(candidate.get("start_seconds", 0.0))
            candidate_end = float(candidate.get("end_seconds", candidate_start))
            for existing in merged_events:
                existing_tokens = set(re.findall(
                    r"\w+", str(existing.get("event", "")).casefold()
                ))
                union = candidate_tokens | existing_tokens
                similarity = len(candidate_tokens & existing_tokens) / max(1, len(union))
                existing_start = float(existing.get("start_seconds", 0.0))
                existing_end = float(existing.get("end_seconds", existing_start))
                overlaps = min(candidate_end, existing_end) >= max(candidate_start, existing_start) - 2.0
                if overlaps and similarity >= 0.65:
                    return True
            return False

        for candidate in candidates:
            for event in candidate.get("events") or []:
                signature = (
                    round(float(event.get("start_seconds", 0.0)), 1),
                    round(float(event.get("end_seconds", 0.0)), 1),
                    re.sub(r"\W+", " ", str(event.get("event", "")).casefold()).strip(),
                )
                if signature not in signatures and not duplicates_existing(event):
                    merged_events.append(copy.deepcopy(event))
                    signatures.add(signature)
        merged_events.sort(key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ))
        for index, event in enumerate(merged_events, start=1):
            event["evidence_id"] = (
                f"EV{int(chunk_number):03d}_{index:04d}"
                if chunk_number is not None else f"EV{index:04d}"
            )
            event["depends_on"] = []
            event["recovered_from_retry"] = True
        best["events"] = merged_events
        best["recovered_from_invalid_json"] = True
        best["recovery_attempt_count"] = len(responses)
        return best

    # Last resort: preserve timeline continuity without claiming an action
    # that the model never returned. Later visual review may replace this row.
    duration = max(0.01, float(clip_duration_sec))
    fallback_payload = {
        "characters": [],
        "events": [{
            "start_seconds": 0.0,
            "end_seconds": duration,
            "event": "Nội dung đoạn video chưa được nhận dạng rõ và cần đối chiếu lại bằng hình ảnh.",
            "visual_detail": "Dữ liệu Gemini không đủ để mô tả cảnh một cách đáng tin cậy.",
            "importance": "bridge",
            "story_role": "development",
            "event_kind": "explanation",
            "needs_visual_review": True,
            "recovery_confidence": 0.0,
        }],
    }
    fallback = normalize_chunk_evidence(
        json.dumps(fallback_payload, ensure_ascii=False),
        global_start_sec,
        duration,
        chunk_number=chunk_number,
        primary_end_sec=primary_end_sec,
    )
    fallback["recovered_from_invalid_json"] = True
    fallback["fallback_evidence"] = True
    fallback["recovery_attempt_count"] = len(responses)
    return fallback


def plan_evidence_replenishment(
    chunk_evidence, missing_count, max_chunks=4, excluded_indices=None,
):
    """Choose under-covered chunks and spread a small evidence deficit over them."""
    missing = max(0, int(missing_count))
    if not missing:
        return []
    excluded = {int(value) for value in (excluded_indices or [])}
    ranked = []
    for chunk_index, chunk in enumerate(chunk_evidence or []):
        if chunk_index in excluded:
            continue
        try:
            chunk_start = float(chunk.get("start_seconds", 0.0))
            chunk_end = float(chunk.get("end_seconds", chunk_start))
        except (TypeError, ValueError):
            continue
        duration = max(0.01, chunk_end - chunk_start)
        events = sorted(
            [event for event in (chunk.get("events") or []) if isinstance(event, dict)],
            key=lambda event: float(event.get("start_seconds", chunk_start)),
        )
        cursor = chunk_start
        largest_gap = 0.0
        for event in events:
            try:
                start = max(chunk_start, float(event.get("start_seconds", cursor)))
                end = min(chunk_end, float(event.get("end_seconds", start)))
            except (TypeError, ValueError):
                continue
            largest_gap = max(largest_gap, max(0.0, start - cursor))
            cursor = max(cursor, end)
        largest_gap = max(largest_gap, max(0.0, chunk_end - cursor))
        density = len(events) / duration
        # Prefer real uncovered time; low-density chunks break ties.
        ranked.append((largest_gap, -density, duration, chunk_index))
    ranked.sort(reverse=True)
    selected = ranked[:max(1, min(int(max_chunks), missing, len(ranked)))]
    if not selected:
        return []
    allocations = [0] * len(selected)
    for index in range(missing):
        allocations[index % len(selected)] += 1
    return [
        {"chunk_index": selected[index][3], "requested_count": count}
        for index, count in enumerate(allocations) if count
    ]


def plan_targeted_evidence_gap_windows(
    chunk_evidence, missing_count, video_duration_sec,
    max_windows=4, window_seconds=90.0,
):
    """Choose a few short windows in the sparsest timeline regions.

    Event midpoints are used instead of broad event ranges because one broad
    summary may span several minutes while still hiding intermediate actions.
    The returned windows never imply a full-chapter rewatch.
    """
    missing = max(0, int(missing_count))
    duration = max(1.0, float(video_duration_sec))
    if not missing:
        return []
    points = []
    for chunk_index, chunk in enumerate(chunk_evidence or []):
        for event in chunk.get("events") or []:
            if not isinstance(event, dict):
                continue
            try:
                start = max(0.0, min(duration, float(event.get("start_seconds", 0.0))))
                end = max(start, min(duration, float(event.get("end_seconds", start))))
            except (TypeError, ValueError):
                continue
            points.append(((start + end) / 2.0, chunk_index))
    points.sort()
    anchors = [(0.0, 0)] + points + [
        (duration, max(0, len(chunk_evidence or []) - 1))
    ]
    candidates = []
    for (left, left_chunk), (right, right_chunk) in zip(anchors, anchors[1:]):
        gap = max(0.0, right - left)
        if gap < 12.0:
            continue
        center = (left + right) / 2.0
        span = min(max(30.0, float(window_seconds)), max(30.0, gap))
        start = max(0.0, center - span / 2.0)
        end = min(duration, start + span)
        start = max(0.0, end - span)
        midpoint = (start + end) / 2.0
        chunk_index = next((
            index for index, chunk in enumerate(chunk_evidence or [])
            if float(chunk.get("start_seconds", 0.0)) <= midpoint
            < float(chunk.get("end_seconds", duration))
        ), left_chunk if gap else right_chunk)
        candidates.append({
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "chunk_index": int(chunk_index),
            "gap_seconds": round(gap, 3),
        })
    candidates.sort(key=lambda row: (-row["gap_seconds"], row["start_seconds"]))
    selected = []
    # Keep reserve windows even when the numerical deficit is small: Gemini
    # may legitimately find no new event in some windows. The caller checks
    # capacity after every clip and stops early as soon as the target is met.
    limit = max(1, int(max_windows))
    for candidate in candidates:
        overlap = any(
            min(candidate["end_seconds"], row["end_seconds"])
            - max(candidate["start_seconds"], row["start_seconds"])
            > 0.50 * min(
                candidate["end_seconds"] - candidate["start_seconds"],
                row["end_seconds"] - row["start_seconds"],
            )
            for row in selected
        )
        if overlap:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    if not selected:
        return []
    allocations = [1] * len(selected)
    remaining = max(0, missing - len(selected))
    for index in range(remaining):
        allocations[index % len(selected)] += 1
    for index, row in enumerate(selected):
        row["requested_count"] = allocations[index]
    return selected


def build_evidence_replenishment_prompt(chunk, requested_count):
    """Ask the attached chunk for additional, non-duplicated story events."""
    chunk_start = float(chunk.get("start_seconds", 0.0))
    chunk_end = float(chunk.get("end_seconds", chunk_start))
    duration = max(0.01, chunk_end - chunk_start)
    existing = []
    for event in chunk.get("events") or []:
        if not isinstance(event, dict):
            continue
        row = copy.deepcopy(event)
        try:
            row["start_seconds"] = round(
                float(row.get("start_seconds", chunk_start)) - chunk_start, 3
            )
            row["end_seconds"] = round(
                float(row.get("end_seconds", chunk_start)) - chunk_start, 3
            )
        except (TypeError, ValueError):
            continue
        row.pop("visual_sources", None)
        existing.append(row)
    return f"""Xem lai TOAN BO clip video dinh kem de tim them bang chung cot truyen THAT.
Clip dai {duration:.2f}s. Can toi da {int(requested_count)} event bo sung.

QUY TAC BAT BUOC:
- Chi tra event nhin thay/nghe thay ro trong video; khong bia, khong suy dien, khong tao event dem.
- Khong lap lai event da co ben duoi, ke ca khi dung cau chu khac.
- Timestamp la thoi gian CUC BO trong clip, tu 0 den {duration:.2f}s.
- Moi event phai la mot hanh dong, phan ung, cau noi, chuyen canh hoac thay doi co ich cho mach truyen.
- Gan granularity="atomic" cho su kien cu the. Tom tat nhieu bien co chi dua vao story_context,
  khong tao event tong quat chong len cac event chi tiet. Chua chac thi gan granularity="uncertain".
- Neu video khong con du event that, tra it hon so luong yeu cau; tuyet doi khong lap cho du.
- characters chi dung ID da co trong DANH SACH NHAN VAT; khong doan them ten.
- Van ban phai bang tieng Viet co dau. Khong viet loi TTS.
- Event moi bat buoc co start_seconds/end_seconds dung bien co. Khong chia deu timestamp hay ep shot 2-3 giay.

DANH SACH NHAN VAT:
{json.dumps(chunk.get('characters') or [], ensure_ascii=False)}

EVENT DA CO (timestamp cuc bo):
{json.dumps(existing, ensure_ascii=False)}

Chi tra JSON hop le:
{{"story_context":[],"events":[{{"granularity":"atomic","start_seconds":1.0,"end_seconds":4.0,"event":"",
"visual_detail":"","cause":"","change":"","consequence":"",
"visual_shots":[],
"characters":["CH001_C1"],"temporal_relation":"continuation",
"narrative_function":"development","depends_on":[],"required_for":"",
"importance":"main","story_role":"development"}}]}}
"""


def merge_evidence_replenishment(chunk, text, return_diagnostics=False):
    """Validate and merge supplemental events without weakening locked evidence."""
    result = copy.deepcopy(chunk)
    payload = _json_object(text)
    chunk_start = float(result.get("start_seconds", 0.0))
    chunk_end = float(result.get("end_seconds", chunk_start))
    duration = max(0.01, chunk_end - chunk_start)
    characters = {
        str(character.get("id", "")).strip().upper(): character
        for character in (result.get("characters") or [])
        if isinstance(character, dict) and str(character.get("id", "")).strip()
    }
    source_ids = {
        str(character.get("source_id", "")).strip().upper(): local_id
        for local_id, character in characters.items()
        if str(character.get("source_id", "")).strip()
    }
    existing = [
        event for event in (result.get("events") or []) if isinstance(event, dict)
    ]
    existing_texts = {
        re.sub(r"\W+", " ", str(event.get("event", "")).casefold()).strip()
        for event in existing
    }
    used_ids = {
        str(event.get("evidence_id", "")).strip().upper() for event in existing
    }
    chunk_number = int(result.get("chunk_number") or 0)
    next_id = 1
    added = 0
    diagnostics = {
        "invalid_shape": 0, "invalid_timestamp": 0, "empty_event": 0,
        "duplicate_text": 0, "duplicate_time": 0, "accepted": 0,
        "visual_repaired": 0,
    }

    event_by_id = {
        str(event.get("evidence_id", "")).strip().upper(): event
        for event in existing if str(event.get("evidence_id", "")).strip()
    }
    for repair in payload.get("visual_repairs") or []:
        if not isinstance(repair, dict):
            continue
        evidence_id = str(repair.get("evidence_id", "")).strip().upper()
        target = event_by_id.get(evidence_id)
        if target is None or not target.get("visual_grounding_missing"):
            continue
        target_start = float(target.get("start_seconds", chunk_start))
        target_end = float(target.get("end_seconds", target_start))
        repaired_sources = []
        for raw_shot in repair.get("visual_shots") or []:
            if not isinstance(raw_shot, dict):
                continue
            try:
                local_shot_start = float(raw_shot.get("start_seconds"))
                local_shot_end = float(raw_shot.get("end_seconds"))
            except (TypeError, ValueError):
                continue
            global_shot_start = chunk_start + local_shot_start
            global_shot_end = chunk_start + local_shot_end
            if not (
                target_start - 0.05 <= global_shot_start < global_shot_end
                <= target_end + 0.05
                and global_shot_end - global_shot_start
                <= MAX_GROUNDED_VISUAL_BEAT_SECONDS + 0.05
            ):
                continue
            repaired_sources.append({
                "evidence_id": evidence_id,
                "start_seconds": global_shot_start,
                "end_seconds": global_shot_end,
                "event": re.sub(
                    r"\s+", " ", str(raw_shot.get("description", ""))
                ).strip() or str(target.get("visual_detail") or target.get("event") or ""),
                "verified_visual_shot": True,
            })
        if repaired_sources:
            target["visual_sources"] = repaired_sources
            target["visual_grounding_missing"] = False
            added += 1
            diagnostics["visual_repaired"] += 1

    for item in payload.get("events") or []:
        if not isinstance(item, dict):
            diagnostics["invalid_shape"] += 1
            continue
        try:
            local_start = float(item.get("start_seconds"))
            local_end = float(item.get("end_seconds"))
        except (TypeError, ValueError):
            diagnostics["invalid_timestamp"] += 1
            continue
        if not (0.0 <= local_start < local_end <= duration + 1.0):
            diagnostics["invalid_timestamp"] += 1
            continue
        event_text = re.sub(r"\s+", " ", str(item.get("event", ""))).strip()
        normalized_text = re.sub(r"\W+", " ", event_text.casefold()).strip()
        if not event_text:
            diagnostics["empty_event"] += 1
            continue
        if normalized_text in existing_texts:
            diagnostics["duplicate_text"] += 1
            continue
        global_start = chunk_start + max(0.0, local_start)
        global_end = chunk_start + min(duration, local_end)
        if any(
            abs(global_start - float(event.get("start_seconds", -99.0))) <= 1.0
            and abs(global_end - float(event.get("end_seconds", -99.0))) <= 1.0
            for event in existing
        ):
            diagnostics["duplicate_time"] += 1
            continue
        row = copy.deepcopy(item)
        row["start_seconds"] = global_start
        row["end_seconds"] = global_end
        row["event"] = event_text
        for field in ("visual_detail", "cause", "change", "consequence"):
            row[field] = re.sub(r"\s+", " ", str(row.get(field, ""))).strip()
        mapped_characters = []
        for value in item.get("characters") or []:
            identity = str(value).strip().upper()
            mapped = identity if identity in characters else source_ids.get(identity)
            if mapped and mapped not in mapped_characters:
                mapped_characters.append(mapped)
        row["characters"] = mapped_characters
        temporal = str(item.get("temporal_relation", "continuation")).lower()
        row["temporal_relation"] = temporal if temporal in {
            "continuation", "time_jump", "flashback", "flashforward",
            "parallel", "return", "unknown",
        } else "unknown"
        narrative = str(item.get("narrative_function", "development")).lower()
        row["narrative_function"] = narrative if narrative in {
            "setup", "inciting_incident", "development", "turning_point",
            "false_victory", "reversal", "climax", "resolution", "ending",
        } else "development"
        dependencies = item.get("depends_on") or []
        row["depends_on"] = [
            str(value).strip().upper() for value in dependencies
            if str(value).strip().upper() in used_ids
        ]
        row["importance"] = (
            "bridge" if str(item.get("importance", "")).lower() == "bridge"
            else "main"
        )
        role = str(item.get("story_role", "development")).lower()
        row["story_role"] = role if role in {
            "development", "climax", "resolution", "ending",
        } else "development"
        while f"EVS{chunk_number:03d}_{next_id:04d}" in used_ids:
            next_id += 1
        row["evidence_id"] = f"EVS{chunk_number:03d}_{next_id:04d}"
        visual_sources = []
        for raw_shot in item.get("visual_shots") or []:
            if not isinstance(raw_shot, dict):
                continue
            try:
                shot_start = float(raw_shot.get("start_seconds"))
                shot_end = float(raw_shot.get("end_seconds"))
            except (TypeError, ValueError):
                continue
            if not (
                local_start - 0.05 <= shot_start < shot_end <= local_end + 0.05
                and shot_end - shot_start
                <= MAX_GROUNDED_VISUAL_BEAT_SECONDS + 0.05
            ):
                continue
            visual_sources.append({
                "evidence_id": row["evidence_id"],
                "start_seconds": chunk_start + shot_start,
                "end_seconds": chunk_start + shot_end,
                "event": re.sub(
                    r"\s+", " ", str(raw_shot.get("description", ""))
                ).strip() or row.get("visual_detail") or event_text,
                "verified_visual_shot": True,
            })
        if not visual_sources and local_end - local_start <= MAX_GROUNDED_VISUAL_BEAT_SECONDS:
            visual_sources.append({
                "evidence_id": row["evidence_id"],
                "start_seconds": global_start,
                "end_seconds": global_end,
                "event": row.get("visual_detail") or event_text,
                "verified_visual_shot": True,
            })
        row["visual_sources"] = visual_sources
        row["visual_grounding_missing"] = not bool(visual_sources)
        row["source"] = "video_replenishment"
        used_ids.add(row["evidence_id"])
        existing_texts.add(normalized_text)
        existing.append(row)
        next_id += 1
        added += 1
        diagnostics["accepted"] += 1
    existing.sort(key=lambda event: (
        float(event.get("start_seconds", 0.0)),
        float(event.get("end_seconds", 0.0)),
    ))
    result["events"] = existing
    result["story_context"] = list(result.get("story_context") or []) + list(payload.get("story_context") or [])
    result = separate_event_granularity([result], review_legacy=False)[0][0]
    if return_diagnostics:
        return result, added, diagnostics
    return result, added


def build_targeted_visual_shot_repair_prompt(events, clip_windows, attempt_number):
    """Ask only for missing short shots, using offsets inside attached clips."""
    windows_by_id = {
        str(window.get("evidence_id", "")).strip().upper(): window
        for window in (clip_windows or []) if isinstance(window, dict)
    }
    cards = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        evidence_id = str(event.get("evidence_id", "")).strip().upper()
        window = windows_by_id.get(evidence_id)
        if not window:
            continue
        cards.append({
            "evidence_id": evidence_id,
            "event": str(event.get("event", "")).strip(),
            "visual_detail": str(event.get("visual_detail", "")).strip(),
            "cause": str(event.get("cause", "")).strip(),
            "consequence": str(event.get("consequence", "")).strip(),
            "required_distinct_beats": event_required_visual_shot_count(event),
            "already_verified_shots": len(verified_event_visual_sources(event)),
            "clip_duration_seconds": round(
                float(window["end_seconds"]) - float(window["start_seconds"]), 3
            ),
        })
    return f"""Xem cac clip ngan dinh kem de BO SUNG HINH cho dung cac evidence_id ben duoi.
Day la luot {int(attempt_number)}/2 cho ca tac vu. Moi clip duoc gui ngay sau nhan ID cua no.

QUY TAC:
- Khong viet lai event, khong them ID, khong viet TTS.
- Voi moi ID, tim beat hinh co y nghia ben trong event; khong lap lai beat da co.
- Beat dai linh hoat theo hanh dong, thuong 1-8 giay, toi da {MAX_GROUNDED_VISUAL_BEAT_SECONDS:.1f} giay.
  Day chua phai shot render 2-3 giay; backend se cat shot sau khi co TTS.
- start_offset/end_offset la giay CUC BO tinh tu dau clip dinh kem cua ID do.
- Neu clip khong co hinh phu hop, dat found=false; tuyet doi khong doan timestamp.

ID CAN SUA:
{json.dumps(cards, ensure_ascii=False)}

Chi tra JSON:
{{"repairs":[{{"evidence_id":"EV...","found":true,"visual_shots":[{{"start_offset":1.0,"end_offset":3.8,"description":"hinh thay truc tiep"}}]}}]}}
"""


def apply_targeted_visual_shot_repairs(
    chunk_evidence, response_text, clip_windows,
):
    """Apply only valid shot offsets to the requested missing evidence IDs."""
    chunks = copy.deepcopy(chunk_evidence or [])
    payload = _json_object(response_text)
    windows = {
        str(window.get("evidence_id", "")).strip().upper(): window
        for window in (clip_windows or []) if isinstance(window, dict)
        and str(window.get("evidence_id", "")).strip()
    }
    event_by_id = {
        str(event.get("evidence_id", "")).strip().upper(): event
        for chunk in chunks for event in (chunk.get("events") or [])
        if isinstance(event, dict) and str(event.get("evidence_id", "")).strip()
    }
    repaired = []
    for repair in payload.get("repairs") or []:
        if not isinstance(repair, dict) or repair.get("found") is not True:
            continue
        evidence_id = str(repair.get("evidence_id", "")).strip().upper()
        target = event_by_id.get(evidence_id)
        window = windows.get(evidence_id)
        if target is None or window is None:
            continue
        clip_start = float(window.get("start_seconds", 0.0))
        clip_end = float(window.get("end_seconds", clip_start))
        sources = []
        for raw_shot in repair.get("visual_shots") or []:
            if not isinstance(raw_shot, dict):
                continue
            try:
                start_offset = float(raw_shot.get("start_offset"))
                end_offset = float(raw_shot.get("end_offset"))
            except (TypeError, ValueError):
                continue
            source_start = clip_start + start_offset
            source_end = clip_start + end_offset
            if not (
                clip_start - 0.05 <= source_start < source_end <= clip_end + 0.05
                and source_end - source_start
                <= MAX_GROUNDED_VISUAL_BEAT_SECONDS + 0.05
            ):
                continue
            sources.append({
                "evidence_id": evidence_id,
                "start_seconds": source_start,
                "end_seconds": source_end,
                "event": re.sub(
                    r"\s+", " ", str(raw_shot.get("description", ""))
                ).strip() or str(target.get("visual_detail") or target.get("event") or ""),
                "verified_visual_shot": True,
                "targeted_visual_repair": True,
            })
        if sources:
            combined = list(verified_event_visual_sources(target))
            seen = {
                (round(float(row["start_seconds"]), 2), round(float(row["end_seconds"]), 2))
                for row in combined
            }
            for source in sources:
                key = (
                    round(float(source["start_seconds"]), 2),
                    round(float(source["end_seconds"]), 2),
                )
                if key not in seen:
                    seen.add(key)
                    combined.append(source)
            combined.sort(key=lambda row: (
                float(row.get("start_seconds", 0.0)),
                float(row.get("end_seconds", 0.0)),
            ))
            target["visual_sources"] = combined
            target["visual_grounding_missing"] = (
                len(combined) < event_required_visual_shot_count(target)
            )
            repaired.append(evidence_id)
    return chunks, repaired


def drop_unrepairable_visual_events(chunk_evidence, evidence_ids):
    """Remove only events that remain visually unverifiable after retries."""
    chunks = copy.deepcopy(chunk_evidence or [])
    rejected = {
        str(value).strip().upper() for value in (evidence_ids or [])
        if str(value).strip()
    }
    removed = 0
    for chunk in chunks:
        kept = []
        for event in chunk.get("events") or []:
            if not isinstance(event, dict):
                continue
            evidence_id = str(event.get("evidence_id", "")).strip().upper()
            if evidence_id in rejected:
                removed += 1
                continue
            event["depends_on"] = [
                value for value in (event.get("depends_on") or [])
                if str(value).strip().upper() not in rejected
            ]
            kept.append(event)
        chunk["events"] = kept
    return chunks, removed


def normalize_global_evidence_audit(text, chunk_evidence, video_duration_sec):
    """Validate a text-only audit and keep all uncertain claims non-destructive."""
    payload = _json_object(text)
    duration = max(1.0, float(video_duration_sec))
    known_local_ids = {
        str(character.get("id", "")).strip().upper()
        for chunk in chunk_evidence or []
        for character in (chunk.get("characters") or [])
        if isinstance(character, dict) and str(character.get("id", "")).strip()
    }
    known_evidence_ids = {
        str(event.get("evidence_id", "")).strip().upper()
        for chunk in chunk_evidence or []
        for event in (chunk.get("events") or [])
        if isinstance(event, dict) and str(event.get("evidence_id", "")).strip()
    }
    canonical = []
    canonical_ids = set()
    for item in payload.get("canonical_characters") or []:
        if not isinstance(item, dict):
            continue
        character_id = str(item.get("id", "")).strip().upper()
        if not re.fullmatch(r"P\d+", character_id) or character_id in canonical_ids:
            continue
        row = copy.deepcopy(item)
        row["id"] = character_id
        canonical.append(row)
        canonical_ids.add(character_id)

    # A name explicitly read in a chunk with high confidence is stronger than
    # a later text-only cross-chunk guess. Preserve those identities in the
    # canonical registry even when the audit model omits them.
    name_to_canonical_id = {
        str(item.get("name", "")).strip().casefold(): str(item["id"]).upper()
        for item in canonical if str(item.get("name", "")).strip()
    }
    confirmed_local_names = {}
    next_canonical_number = 1
    while f"P{next_canonical_number}" in canonical_ids:
        next_canonical_number += 1
    for chunk in chunk_evidence or []:
        for character in chunk.get("characters") or []:
            if not isinstance(character, dict):
                continue
            local_id = str(character.get("id", "")).strip().upper()
            name = re.sub(
                r"\s+", " ", str(
                    character.get("proposed_name") or character.get("name") or ""
                )
            ).strip()
            try:
                confidence = float(character.get("identity_confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            status = str(character.get("identity_status", "")).strip().lower()
            if character.get('character_registry_version', 0) >= 2 and character.get('name_status') != 'confirmed':
                continue
            if not local_id or not name or status != "confirmed" or confidence < 0.90:
                continue
            canonical_id = name_to_canonical_id.get(name.casefold())
            if canonical_id is None:
                while f"P{next_canonical_number}" in canonical_ids:
                    next_canonical_number += 1
                canonical_id = f"P{next_canonical_number}"
                next_canonical_number += 1
                canonical_ids.add(canonical_id)
                name_to_canonical_id[name.casefold()] = canonical_id
                canonical.append({
                    "id": canonical_id,
                    "name": name,
                    "descriptive_alias": str(
                        character.get("descriptive_alias", "")
                    ).strip(),
                    "description": str(character.get("description", "")).strip(),
                    "role": str(character.get("role", "")).strip(),
                    "source": "confirmed_chunk_identity",
                })
            confirmed_local_names[local_id] = canonical_id

    links = []
    seen_local_ids = set()
    allowed_basis = {
        "explicit_name", "direct_transition", "distinctive_visual",
        "multiple_evidence", "uncertain",
    }
    for item in payload.get("identity_links") or []:
        if not isinstance(item, dict):
            continue
        local_id = str(item.get("local_id", "")).strip().upper()
        canonical_id = str(item.get("canonical_id", "")).strip().upper()
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        basis = str(item.get("basis", "uncertain")).strip().lower()
        if (
            local_id not in known_local_ids or local_id in seen_local_ids
            or canonical_id not in canonical_ids
        ):
            continue
        links.append({
            "local_id": local_id,
            "canonical_id": canonical_id,
            "confidence": min(1.0, max(0.0, confidence)),
            "basis": basis if basis in allowed_basis else "uncertain",
            "reason": str(item.get("reason", "")).strip(),
        })
        seen_local_ids.add(local_id)

    # Explicit high-confidence chunk names are immutable. Replace any
    # conflicting audit link instead of allowing a hero to become a villain.
    links_by_local_id = {link["local_id"]: link for link in links}
    for local_id, canonical_id in confirmed_local_names.items():
        links_by_local_id[local_id] = {
            "local_id": local_id,
            "canonical_id": canonical_id,
            "confidence": 1.0,
            "basis": "explicit_name",
            "reason": "Ten da duoc xac nhan voi confidence >= 0.90 trong chunk video.",
        }
    links = list(links_by_local_id.values())

    contradictions = []
    for item in payload.get("contradictions") or []:
        if not isinstance(item, dict):
            continue
        evidence_ids = [
            str(value).strip().upper() for value in (item.get("evidence_ids") or [])
            if str(value).strip().upper() in known_evidence_ids
        ]
        if evidence_ids:
            row = copy.deepcopy(item)
            row["evidence_ids"] = evidence_ids
            contradictions.append(row)

    windows = []
    for item in (payload.get("review_windows") or [])[:6]:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item.get("start_seconds")))
            end = min(duration, float(item.get("end_seconds")))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if end - start > 120.0:
            midpoint = (start + end) / 2.0
            start = max(0.0, midpoint - 60.0)
            end = min(duration, start + 120.0)
        evidence_ids = [
            str(value).strip().upper() for value in (item.get("evidence_ids") or [])
            if str(value).strip().upper() in known_evidence_ids
        ]
        windows.append({
            "start_seconds": start,
            "end_seconds": end,
            "reason": str(item.get("reason", "")).strip(),
            "evidence_ids": evidence_ids,
            "questions": [
                str(value).strip() for value in (item.get("questions") or [])
                if str(value).strip()
            ],
        })
    comparisons = []
    for item in (payload.get("identity_comparisons") or [])[:3]:
        if not isinstance(item, dict):
            continue
        try:
            first_start = max(0.0, float(item.get("first_start_seconds")))
            first_end = min(duration, float(item.get("first_end_seconds")))
            second_start = max(0.0, float(item.get("second_start_seconds")))
            second_end = min(duration, float(item.get("second_end_seconds")))
        except (TypeError, ValueError):
            continue
        if first_end <= first_start or second_end <= second_start:
            continue
        if first_end - first_start > 60.0:
            first_end = first_start + 60.0
        if second_end - second_start > 60.0:
            second_end = second_start + 60.0
        local_ids = [
            str(value).strip().upper() for value in (item.get("local_ids") or [])
            if str(value).strip().upper() in known_local_ids
        ]
        if len(local_ids) < 2:
            continue
        comparisons.append({
            "first_start_seconds": first_start,
            "first_end_seconds": first_end,
            "second_start_seconds": second_start,
            "second_end_seconds": second_end,
            "reason": str(item.get("reason", "")).strip(),
            "local_ids": local_ids,
            "question": str(item.get("question", "")).strip(),
        })
    return {
        "canonical_characters": canonical,
        "identity_links": links,
        "contradictions": contradictions,
        "review_windows": windows,
        "identity_comparisons": comparisons,
    }


def add_objective_evidence_gap_windows(audit, chunk_evidence, video_duration_sec):
    """Add only observable coverage gaps that a text-only audit cannot see."""
    result = copy.deepcopy(audit)
    windows = list(result.get("review_windows") or [])
    duration = max(1.0, float(video_duration_sec))
    all_events = sorted(
        [
            event for chunk in chunk_evidence or []
            for event in (chunk.get("events") or []) if isinstance(event, dict)
        ],
        key=lambda item: float(item.get("start_seconds", 0.0)),
    )
    # A missing opening is objectively detectable. Do not infer its content;
    # send the uncovered footage back for verification.
    if all_events:
        first_start = float(all_events[0].get("start_seconds", 0.0))
        if first_start > 30.0:
            windows.insert(0, {
                "start_seconds": 0.0,
                "end_seconds": min(duration, max(45.0, min(120.0, first_start + 5.0))),
                "reason": "Video co khoang mo dau hon 30 giay khong co event nao duoc trich xuat.",
                "evidence_ids": [],
                "questions": [
                    "Khoang mo dau co gioi thieu nhan vat, bien co khoi phat, "
                    "tham hoa hoac chuyen thoi gian bi bo sot khong?"
                ],
                "objective_gap": True,
            })
        for open_event in [
            event for event in all_events
            if (
                bool(event.get("continues_to_next"))
                or str(event.get("completion_status", "")).lower() == "open"
            ) and not bool(event.get("continuation_stitched"))
        ][:2]:
            event_start = float(open_event.get("start_seconds", 0.0))
            event_end = float(open_event.get("end_seconds", event_start))
            windows.append({
                "start_seconds": max(0.0, event_start - 15.0),
                "end_seconds": min(duration, event_end + 180.0),
                "reason": "Su kien o bien clip van dang do va chua noi duoc voi clip sau.",
                "evidence_ids": [str(open_event.get("evidence_id", "")).upper()],
                "questions": [
                    "Su kien nay ket thuc ra sao, hay tiep noi vao cum su kien nao o clip sau?"
                ],
                "objective_gap": True,
            })
        # A 45-second event-free hole can already hide a complete causal beat.
        # Review the largest bounded set instead of waiting for a three-minute
        # failure, which previously let many action transitions disappear.
        internal_gaps = []
        for earlier, later in zip(all_events, all_events[1:]):
            gap_start = float(earlier.get("end_seconds", 0.0))
            gap_end = float(later.get("start_seconds", 0.0))
            if gap_end - gap_start >= 45.0:
                internal_gaps.append((gap_end - gap_start, gap_start, gap_end))
        for _, gap_start, gap_end in sorted(internal_gaps, reverse=True)[:4]:
            midpoint = (gap_start + gap_end) / 2.0
            windows.append({
                "start_seconds": max(0.0, midpoint - 60.0),
                "end_seconds": min(duration, midpoint + 60.0),
                "reason": "Co khoang hon 45 giay giua hai event ma khong co bang chung trung gian.",
                "evidence_ids": [],
                "questions": [
                    "Khoang nay co buoc ngoat, thay doi muc tieu, doi tuyen nhan vat "
                    "hoac quan he nhan-qua bi bo sot khong?"
                ],
                "objective_gap": True,
            })
        last_event = all_events[-1]
        last_end = float(last_event.get("end_seconds", 0.0))
        last_role = str(last_event.get("story_role", "")).lower()
        if duration - last_end > 45.0 and last_role != "ending":
            windows.append({
                "start_seconds": max(last_end, duration - 120.0),
                "end_seconds": duration,
                "reason": "Duoi video con hon 90 giay nhung event cuoi chua duoc xac nhan la ending.",
                "evidence_ids": [str(last_event.get("evidence_id", "")).upper()],
                "questions": ["Co resolution hoac ending that bi bo sot khong?"],
                "objective_gap": True,
            })
    # Deduplicate strongly overlapping windows and keep the bounded API budget.
    deduplicated = []
    for window in windows:
        start = float(window.get("start_seconds", 0.0))
        end = float(window.get("end_seconds", 0.0))
        if any(
            min(end, float(existing["end_seconds"]))
            - max(start, float(existing["start_seconds"]))
            >= 0.7 * min(end - start, float(existing["end_seconds"]) - float(existing["start_seconds"]))
            for existing in deduplicated
        ):
            continue
        deduplicated.append(window)
        if len(deduplicated) >= 6:
            break
    result["review_windows"] = deduplicated
    return result


def build_evidence_review_prompt(window, audit, chunk_evidence):
    """Ask Gemini to resolve only one disputed interval using attached video."""
    relevant_ids = set(window.get("evidence_ids") or [])
    relevant_events = [
        event for chunk in chunk_evidence or [] for event in (chunk.get("events") or [])
        if str(event.get("evidence_id", "")).upper() in relevant_ids
    ]
    missing_rule = (
        "- Day la khoang trong do BACKEND phat hien. Duoc tra `missing_events` cho bien co that nhin thay "
        "trong clip; start_seconds/end_seconds phai la thoi gian TOAN CUC nam trong cua so kiem chung."
        if window.get("objective_gap") else
        "- khong tao event moi hoac missing_events vi day khong phai khoang trong do backend xac nhan."
    )
    return f"""Xem ky TOAN BO clip kiem chung dinh kem. Clip nay bat dau tai
{float(window['start_seconds']):.2f}s cua video goc. Chi giai quyet nghi van duoi day; khong viet TTS,
khong sua timestamp cua event da co.

NGHI VAN: {json.dumps(window, ensure_ascii=False)}
NHAN VAT DE XUAT: {json.dumps(audit.get('canonical_characters', []), ensure_ascii=False)}
BANG CHUNG LIEN QUAN: {json.dumps(relevant_events, ensure_ascii=False)}

- Chi xac nhan danh tinh khi clip co bang chung truc tiep. Neu van khong ro, de `confirmed=false`.
- Event repair chi duoc sua event/visual_detail/cause/change/consequence/temporal_relation cua evidence_id co san.
- temporal_relation chi duoc continuation, time_jump, flashback, flashforward, parallel, return, unknown.
- Khong suy dien noi tam, ten, quan he hoac ket cuc khong hien dien trong clip.
{missing_rule}

Chi tra JSON:
{{"identity_verifications":[{{"local_id":"CH001_C1","canonical_id":"P1",
"confirmed":true,"confidence":0.0,"basis":"explicit_name|direct_transition|distinctive_visual|uncertain"}}],
"event_repairs":[{{"evidence_id":"EV001_0001","event":"","visual_detail":"","cause":"",
"change":"","consequence":"","temporal_relation":"unknown"}}],
"missing_events":[{{"start_seconds":1.0,"end_seconds":4.0,"event":"","visual_detail":"",
"cause":"","change":"","consequence":"","characters":["P1"],"temporal_relation":"continuation",
"narrative_function":"inciting_incident","depends_on":[],"required_for":""}}],
"resolved":true,"notes":""}}
"""


def build_identity_comparison_prompt(comparison, audit, chunk_evidence):
    """Verify one cross-time identity using a two-interval comparison reel."""
    local_ids = set(comparison.get("local_ids") or [])
    appearances = [
        character for chunk in chunk_evidence or []
        for character in (chunk.get("characters") or [])
        if str(character.get("id", "")).upper() in local_ids
    ]
    return f"""Video dinh kem la reel ghep HAI khoang o hai thoi diem khac nhau cua cung mot phim.
Hay doi chieu nhan vat, khong viet TTS va khong sua event.

NGHI VAN: {json.dumps(comparison, ensure_ascii=False)}
CAC LAN XUAT HIEN: {json.dumps(appearances, ensure_ascii=False)}
NHAN VAT CHUAN DE XUAT: {json.dumps(audit.get('canonical_characters', []), ensure_ascii=False)}

- Chi xac nhan hai local_id la cung nguoi khi co dac diem rieng, chuyen tiep truc tiep, ten duoc goi,
  hoac nhieu dau hieu doc lap. Trang phuc chung chung, tuoi va mau toc rieng le la chua du.
- Neu khong chac, confirmed=false. Khong gan ten de lam cho output dep hon.

Chi tra JSON:
{{"identity_verifications":[{{"local_id":"CH001_C1","canonical_id":"P1",
"confirmed":true,"confidence":0.0,"basis":"explicit_name|direct_transition|distinctive_visual|uncertain"}}],
"event_repairs":[],"resolved":true,"notes":""}}
"""


def apply_verified_evidence_results(chunk_evidence, audit, verification_texts=None):
    """Apply only high-confidence identities or video-verified corrections."""
    result = copy.deepcopy(list(chunk_evidence or []))
    canonical = {
        str(item.get("id", "")).strip().upper(): item
        for item in audit.get("canonical_characters", []) if isinstance(item, dict)
    }
    locked_name_by_local = {}
    for chunk in result:
        for character in chunk.get("characters", []):
            if not isinstance(character, dict):
                continue
            try:
                confidence = float(character.get("identity_confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            name = str(
                character.get("proposed_name") or character.get("name") or ""
            ).strip()
            if (
                name and confidence >= 0.90
                and str(character.get("identity_status", "")).lower() == "confirmed"
            ):
                locked_name_by_local[
                    str(character.get("id", "")).strip().upper()
                ] = name
    verified_links = {}
    # Text-only links are safe only with strong evidence and high confidence.
    for link in audit.get("identity_links", []):
        local_id = str(link.get("local_id", "")).upper()
        canonical_id = str(link.get("canonical_id", "")).upper()
        locked_name = locked_name_by_local.get(local_id, "")
        target_name = str(canonical.get(canonical_id, {}).get("name", "")).strip()
        if (
            float(link.get("confidence", 0.0)) >= 0.90
            and link.get("basis") in {"explicit_name", "direct_transition", "multiple_evidence"}
            and not (
                locked_name and target_name
                and locked_name.casefold() != target_name.casefold()
            )
        ):
            verified_links[local_id] = canonical_id

    repairs = {}
    missing_events = []
    for text in verification_texts or []:
        payload = _json_object(text)
        for link in payload.get("identity_verifications") or []:
            if not isinstance(link, dict) or not bool(link.get("confirmed")):
                continue
            try:
                confidence = float(link.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            local_id = str(link.get("local_id", "")).strip().upper()
            canonical_id = str(link.get("canonical_id", "")).strip().upper()
            locked_name = locked_name_by_local.get(local_id, "")
            target_name = str(canonical.get(canonical_id, {}).get("name", "")).strip()
            conflicts_with_locked_name = bool(
                locked_name and target_name
                and locked_name.casefold() != target_name.casefold()
            )
            if (
                confidence >= 0.85 and canonical_id in canonical
                and not conflicts_with_locked_name
            ):
                verified_links[local_id] = canonical_id
        for repair in payload.get("event_repairs") or []:
            if isinstance(repair, dict) and str(repair.get("evidence_id", "")).strip():
                repairs[str(repair["evidence_id"]).strip().upper()] = repair
        review_window = payload.get("_review_window")
        if isinstance(review_window, dict) and bool(
            review_window.get("allow_missing_events")
        ):
            try:
                review_start = float(review_window.get("start_seconds"))
                review_end = float(review_window.get("end_seconds"))
            except (TypeError, ValueError):
                review_start = review_end = -1.0
            for missing in payload.get("missing_events") or []:
                if not isinstance(missing, dict):
                    continue
                try:
                    start = float(missing.get("start_seconds"))
                    end = float(missing.get("end_seconds"))
                except (TypeError, ValueError):
                    continue
                event_text = re.sub(r"\s+", " ", str(missing.get("event", ""))).strip()
                if not (review_start <= start < end <= review_end):
                    clip_duration = review_end - review_start
                    if 0.0 <= start < end <= clip_duration:
                        start += review_start
                        end += review_start
                if review_start <= start < end <= review_end and event_text:
                    row = copy.deepcopy(missing)
                    row["start_seconds"] = start
                    row["end_seconds"] = end
                    row["event"] = event_text
                    missing_events.append(row)

    known_local_ids = {
        str(character.get("id", "")).strip().upper()
        for chunk in result for character in chunk.get("characters", [])
        if isinstance(character, dict)
    }
    verified_links = {
        local_id: canonical_id for local_id, canonical_id in verified_links.items()
        if local_id in known_local_ids
    }
    local_characters = {
        str(character.get("id", "")).strip().upper(): character
        for chunk in result for character in chunk.get("characters", [])
        if isinstance(character, dict) and str(character.get("id", "")).strip()
    }
    for chunk in result:
        for character in chunk.get("characters", []):
            local_id = str(character.get("id", "")).strip().upper()
            canonical_id = verified_links.get(local_id)
            if canonical_id:
                character["canonical_id"] = canonical_id
                character["identity_status"] = "confirmed"
                character["name"] = str(canonical[canonical_id].get("name", "")).strip()
        for event in chunk.get("events", []):
            raw_character_ids = [
                str(value).strip().upper() for value in (event.get("characters") or [])
                if str(value).strip()
            ]
            event["local_character_ids"] = raw_character_ids
            readable_characters = []
            prose_replacements = {}
            token_labels, token_ids = {}, {}
            for local_id in raw_character_ids:
                source_character = local_characters.get(local_id, {})
                proposed_name = str(
                    source_character.get("proposed_name")
                    or source_character.get("name") or ""
                ).strip()
                canonical_id = verified_links.get(local_id)
                if canonical_id:
                    canonical_name = str(
                        canonical[canonical_id].get("name", "")
                    ).strip()
                    readable_characters.append(canonical_id)
                    if proposed_name and canonical_name and proposed_name != canonical_name:
                        prose_replacements[proposed_name] = canonical_name
                else:
                    neutral_label = _descriptive_character_alias(
                        source_character, len(readable_characters) + 1
                    )
                    readable_characters.append(neutral_label)
                    if proposed_name and proposed_name != neutral_label:
                        prose_replacements[proposed_name] = neutral_label
                label = (canonical[canonical_id].get('display_name') or canonical[canonical_id].get('name')
                         or canonical[canonical_id].get('neutral_alias')) if canonical_id else neutral_label
                for source_id in (local_id, str(source_character.get('source_id') or local_id).upper()):
                    token_labels[source_id] = label or _descriptive_character_alias(source_character, 1)
                    if canonical_id:
                        token_ids[source_id] = canonical_id
            event["characters"] = readable_characters
            for field in (
                "event", "visual_detail", "cause", "change", "consequence",
            ):
                prose = str(event.get(field, ""))
                from character_identity import safe_name_mentions, render_ids
                event.setdefault('raw_' + field, prose)
                event[field] = (render_ids(event['raw_' + field], token_labels, allow_bare=False)
                    if event.get('identity_tokens_rendered') else safe_name_mentions(prose, prose_replacements))
            for fact in event.get('grounded_facts') or []:
                fact.setdefault('raw_text', fact.get('text', ''))
                if fact.get('identity_tokens_rendered'):
                    fact['text'] = render_ids(fact['raw_text'], token_labels, allow_bare=False)
                    fact['canonical_actor_ids'] = [token_ids[i.upper()] for i in fact.get('actor_ids') or [] if i.upper() in token_ids]
                else:
                    fact['text'] = safe_name_mentions(fact.get('text', ''), prose_replacements)
            repair = repairs.get(str(event.get("evidence_id", "")).upper())
            if not repair:
                continue
            for field in (
                "event", "visual_detail", "cause", "change", "consequence",
                "temporal_relation",
            ):
                value = str(repair.get(field, "")).strip()
                if value:
                    event[field] = value
    existing_ranges = {
        (
            round(float(event.get("start_seconds", 0.0)), 2),
            round(float(event.get("end_seconds", 0.0)), 2),
            str(event.get("event", "")).casefold(),
        )
        for chunk in result for event in (chunk.get("events") or [])
    }
    next_review_id = 1
    for missing in sorted(
        missing_events, key=lambda item: float(item.get("start_seconds", 0.0))
    ):
        identity = (
            round(float(missing["start_seconds"]), 2),
            round(float(missing["end_seconds"]), 2),
            str(missing["event"]).casefold(),
        )
        if identity in existing_ranges:
            continue
        owner = next((
            chunk for chunk in result
            if float(chunk.get("start_seconds", 0.0)) <= float(missing["start_seconds"])
            < float(chunk.get("end_seconds", 0.0))
        ), None)
        if owner is None:
            continue
        while any(
            str(event.get("evidence_id", "")).upper() == f"EVR{next_review_id:04d}"
            for chunk in result for event in (chunk.get("events") or [])
        ):
            next_review_id += 1
        row = copy.deepcopy(missing)
        row["evidence_id"] = f"EVR{next_review_id:04d}"
        row["importance"] = (
            "bridge" if str(row.get("importance", "")).lower() == "bridge" else "main"
        )
        row["story_role"] = str(row.get("story_role", "development")).lower()
        row["source"] = "verified_missing_gap"
        owner.setdefault("events", []).append(row)
        owner["events"].sort(key=lambda item: (
            float(item.get("start_seconds", 0.0)),
            float(item.get("end_seconds", 0.0)),
        ))
        existing_ranges.add(identity)
        next_review_id += 1
    return result


def compact_story_evidence(chunk_evidence):
    """Strip audit-only detail before the text-only global outline request."""
    compact = []
    seen_ids = set()
    for chunk_index, chunk in enumerate(chunk_evidence or [], start=1):
        if not isinstance(chunk, dict):
            continue
        events = []
        for event in chunk.get("events") or []:
            if not isinstance(event, dict) or event.get("derived_visual_slot"):
                continue
            evidence_id = str(
                event.get("evidence_id") or event.get("local_event_id") or ""
            ).strip().upper()
            if not evidence_id:
                evidence_id = f"C{chunk_index:03d}E{len(events) + 1:04d}"
            if evidence_id in seen_ids:
                continue
            seen_ids.add(evidence_id)

            def short_text(key, limit):
                value = re.sub(r"\s+", " ", str(event.get(key, ""))).strip()
                return value[:limit]

            row = {
                "evidence_id": evidence_id,
                "start_seconds": event.get("start_seconds", 0),
                "end_seconds": event.get("end_seconds", 0),
                "event": short_text("event", 280),
                "narrative_function": str(
                    event.get("narrative_function", "development")
                ).strip().lower(),
                "importance": str(event.get("importance", "main")).strip().lower(),
            }
            optional = {
                "cause": short_text("cause", 140),
                "action": short_text("action", 180),
                "consequence": short_text("consequence", 140),
                "change": short_text("change", 140),
                "result": short_text("result", 140),
                "state_before": short_text("state_before", 140),
                "state_after": short_text("state_after", 140),
                "introduces": short_text("introduces", 100),
                "event_kind": str(event.get("event_kind", "")).strip().lower(),
                "event_cluster": str(event.get("event_cluster", "")).strip(),
                "repeat_group": str(event.get("repeat_group", "")).strip(),
                "required_for": short_text("required_for", 100),
                "temporal_relation": str(
                    event.get("temporal_relation", "continuation")
                ).strip().lower(),
                "story_role": str(event.get("story_role", "")).strip().lower(),
                # Whisper is evidence, not narration.  Keep a bounded excerpt
                # so the planner can understand dialogue without receiving the
                # entire film transcript or inventing quotes.
                "spoken_text": short_text("spoken_text", 320),
            }
            row.update({key: value for key, value in optional.items() if value})
            dependencies = [
                str(value).strip().upper()
                for value in (event.get("depends_on") or [])
                if str(value).strip()
            ]
            if dependencies:
                row["depends_on"] = dependencies[:8]
            characters = [
                str(value).strip() for value in (event.get("characters") or [])
                if str(value).strip()
            ]
            if characters:
                row["characters"] = characters[:8]
            events.append(row)
        outline_events = [
            row for row in events
            if not (
                row.get("importance") == "bridge"
                and row.get("narrative_function") == "development"
                and not any(row.get(key) for key in (
                    "cause", "action", "consequence", "change", "result",
                    "state_before", "state_after", "introduces", "required_for",
                    "story_role",
                ))
            )
        ]
        # A completely quiet chunk still keeps one real card so the outline can
        # see that time passed, without receiving every repeated movement shot.
        events = outline_events or events[:1]
        if events:
            compact_characters = []
            for character in chunk.get("characters") or []:
                if not isinstance(character, dict):
                    continue
                character_id = str(
                    character.get("canonical_id") or character.get("id") or ""
                ).strip()
                if not character_id:
                    continue
                compact_character = {"id": character_id}
                for key in (
                    "name", "proposed_name", "descriptive_alias",
                    "identity_status", "role",
                ):
                    value = re.sub(
                        r"\s+", " ", str(character.get(key, ""))
                    ).strip()
                    if value:
                        compact_character[key] = value[:120]
                compact_characters.append(compact_character)
            compact_chunk = {
                "chunk_number": int(chunk.get("chunk_number", chunk_index) or chunk_index),
                "chunk_start_seconds": chunk.get("chunk_start_seconds", 0),
                "chunk_end_seconds": chunk.get("chunk_end_seconds", 0),
                "events": events,
            }
            if compact_characters:
                compact_chunk["characters"] = compact_characters
            compact.append(compact_chunk)
    return compact


def split_compact_story_evidence(compact_evidence, maximum_events=60):
    """Split compact cards without splitting or duplicating an event."""
    limit = max(10, int(maximum_events))
    groups = []
    current = []
    current_count = 0
    for chunk in compact_evidence or []:
        events = list(chunk.get("events") or []) if isinstance(chunk, dict) else []
        offset = 0
        while offset < len(events):
            if current_count >= limit:
                groups.append(current)
                current = []
                current_count = 0
            take = min(limit - current_count, len(events) - offset)
            piece = {
                key: copy.deepcopy(value)
                for key, value in chunk.items() if key != "events"
            }
            piece["events"] = copy.deepcopy(events[offset:offset + take])
            current.append(piece)
            current_count += take
            offset += take
    if current:
        groups.append(current)
    return groups


def build_story_evidence_digest_prompt(evidence_group, group_number, group_count):
    """Ask for a small causal digest used only after a global-plan timeout."""
    return f"""Ban dang rut gon nhom bang chung {int(group_number)}/{int(group_count)}.
Khong co video trong luot nay. Khong bia su kien, ten rieng, noi tam hay timestamp.
- Gop cac hanh dong lap va canh di chuyen khong lam doi cot truyen.
- Giu setup can thiet, bien co khoi phat, muc tieu, tro ngai, turning point, reversal,
  climax, ket qua va ending.
- Khong gop mat cac mat xich khac nhau cua mot action cluster. Phai giu rieng phat hien/nguyen nhan,
  chuan bi hoac bien doi, tan cong/phan ung, dao chieu va ket qua neu evidence co cac buoc do.
- Gop cac evidence cung repeat_group neu chung chi lap lai cung mot trang thai hoac cam xuc.
- Giu source_evidence_ids de doi chieu lai bang chung goc.
- Toi da 24 story_beats, theo dung thu tu thoi gian.

BANG CHUNG:
{json.dumps(evidence_group, ensure_ascii=False, separators=(',', ':'))}

Chi tra JSON:
{{"story_beats":[{{"start_seconds":0,"end_seconds":0,"summary":"","cause":"",
"outcome":"","narrative_function":"development","source_evidence_ids":["EV..."]}}]}}
"""


def build_story_plan_from_digests_prompt(
    digests, video_duration_sec, target_words, style="dramatic", custom_prompt="",
    global_outline=None,
):
    """Build the same compact three-part plan from timeout-safe digests."""
    extras = str(custom_prompt or "").strip()
    outline_json = json.dumps(
        global_outline or {}, ensure_ascii=False, separators=(",", ":")
    )
    return f"""Ban la bien tap vien recap. Duoi day la cac story digest da rut gon tu bang chung video.
Khong co video trong luot nay va khong duoc bia them su kien/timestamp.
Hay chia cau chuyen thanh 3 phan tai hai buoc ngoat tu nhien cho video dai
{float(video_duration_sec):.2f}s va ngan sach {int(target_words)} tu.
- Ba phan lien tiep, phu den ending that.
- Giu nguyen nhan, muc tieu, reversal, climax, ket qua va ending.
- Phan bo trong so theo tam quan trong, khong theo so luong canh.
- Chon CAC CUM SU KIEN, khong chon mot canh don le. Mot cum action/cao trao phai giu du phat hien/
  nguyen nhan -> chuan bi/bien doi -> doi dau/dao chieu -> ket qua neu cac moc do co trong digest.
- `event_ids` va `required_context_ids` chi dung source_evidence_ids co trong digest. Khong rai deu theo timeline.
- Bat buoc tao `editorial_anchors` cho 6 chuc nang: opening, inciting_incident,
  protagonist_goal, climax, resolution, ending. Moi nhom chi dung ID bang chung co that.
- `opening` phai cho biet trang thai ban dau/nhan vat; `ending` phai la ket qua sau cao trao,
  khong dung credit, loi cam on hoac bai hoc chung chung.
- Loai canh lap trang thai/cam xuc; neu con thieu ngan sach thi mo rong cum quan trong, khong them canh trang tri.
- Huong ke: {style_storytelling_prompt(style)}
{('- Yeu cau them: ' + extras) if extras else ''}

STORY DIGESTS:
{json.dumps(list(digests or []), ensure_ascii=False, separators=(',', ':'))}

OUTLINE TU LUOT XEM TOAN PHIM (chi dinh huong, digest/evidence da xac minh la nguon timestamp):
{outline_json}

Chi tra JSON hop le:
{{"editorial_contract_version":1,
"editorial_anchors":{{"opening":["EV..."],"inciting_incident":["EV..."],
"protagonist_goal":["EV..."],"climax":["EV..."],"resolution":["EV..."],"ending":["EV..."]}},
"boundary_1_seconds":0.0,"boundary_2_seconds":0.0,"part_weights":[1,1,1],
"selected_event_clusters":[{{"cluster_id":"C001","summary":"","importance":"critical",
"event_ids":["EV..."],"required_context_ids":["EV..."],"reason":""}}],
"story_contract":{{"premise":"","protagonist_goal":"","central_conflict":"","theme":"","ending_meaning":""}},
"characters":[{{"id":"P1","name":"","description":"","role":""}}],
"parts":[{{"part":1,"summary":"","end_state":"","handoff_to_next":""}},
{{"part":2,"summary":"","end_state":"","handoff_to_next":""}},
{{"part":3,"summary":"","end_state":"","handoff_to_next":""}}]}}
"""


def build_story_plan_from_evidence_prompt(
    chunk_evidence, video_duration_sec, target_words,
    style="dramatic", custom_prompt="", global_outline=None,
):
    """Build the three-part story spine from locked chunk evidence only."""
    extras = str(custom_prompt or "").strip()
    evidence_json = json.dumps(
        compact_story_evidence(chunk_evidence),
        ensure_ascii=False, separators=(",", ":"),
    )
    outline_json = json.dumps(
        global_outline or {}, ensure_ascii=False, separators=(",", ":")
    )
    return f"""Ban la bien tap vien recap. Video da duoc xem theo tung doan ngan va bang chung
ben duoi da bi khoa. KHONG co video trong luot nay va khong duoc bia them su kien/timestamp.

Hay hop nhat nhan vat trung nhau va chia cau chuyen thanh 3 phan tai hai buoc ngoat tu nhien.
KHONG lap lai danh sach event trong output; backend se giu toan bo bang chung. Video dai
{float(video_duration_sec):.2f}s; ngan sach loi ke {int(target_words)} tu.
- ID dang CHxxx_Cx la danh tinh cuc bo cua tung clip, KHONG phai ten nhan vat. Neu character co
  `canonical_id` thi bat buoc dung ID P do. Neu van unknown, dung `descriptive_alias` tieng Viet ngan gon,
  on dinh va khong doan ten rieng.
- Doc `temporal_relation` de phan biet tiep dien, nhay thoi gian, hoi tuong, tuyen song song va quay lai.
- Neu mot chien thang bi dao nguoc boi event sau, mo ta do la ket qua tam thoi; khong tuyen bo ket thuc that.
- Ba phan lien tiep, phu den ending that; khong dung credit/outro lam ending.
- Giu du nguyen nhan, buoc ngoat, cao trao, ket qua va ket thuc.
- Chon event theo tam quan trong TOAN PHIM, khong chia deu theo thoi gian va khong dung so scene lam quota.
- Bat buoc tao `selected_event_clusters`. Moi cum la mot thay doi cot truyen hoan chinh, khong phai mot
  timestamp don le. Cum action/cao trao phai giu cac event co that theo thu tu: phat hien/nguyen nhan ->
  chuan bi hoac bien doi -> tan cong/phan ung -> dao chieu -> ket qua. Khong can ep buoc khong ton tai.
- Neu mot event gioi thieu quai vat, vu khi, nhan vat hoac ket qua moi, dua cac event truc tiep tao ra no vao
  `required_context_ids`; khong de mot moi de doa tu nhien xuat hien khong nguon goc.
- Gom cac event cung `repeat_group` hoac chi lap lai cung trang thai; giu event co thay doi/ket qua ro nhat.
- Ngan sach tu quyet dinh muc do chi tiet cua cum da chon. Neu can them tu, mo rong action cluster quan trong;
  khong chen phong canh, di chuyen, an uong, cho doi, cam on hay chia tay lap.
- event_id duy nhat E0001, E0002...; importance main/bridge.
- Bat buoc dat `editorial_contract_version=1` va tao `editorial_anchors` gom dung 6 nhom:
  opening, inciting_incident, protagonist_goal, climax, resolution, ending.
- Moi anchor chi dung `evidence_id` co trong BANG CHUNG KHOA. Opening phai dat truoc climax;
  resolution/ending phai nam sau climax. Ending phai cho thay ket qua/trang thai cuoi that, khong dung credit,
  loi cam on, bai hoc chung chung hoac mot tran danh con dang do.
- Dua moi ID anchor vao `selected_event_clusters` critical cung context truc tiep cua no; khong duoc bo anchor
  chi vi ngan sach ngan. Cat canh sinh hoat/lap y truoc.
- Tao story_contract, summary, end_state va handoff de giu mach ke.
- Huong ke: {style_storytelling_prompt(style)}
{('- Yeu cau them: ' + extras) if extras else ''}

BANG CHUNG KHOA THEO DOAN:
{evidence_json}

OUTLINE TU LUOT XEM TOAN PHIM (chi dinh huong; bang chung khoa o tren moi quyet dinh timestamp):
{outline_json}

Chi tra JSON hop le theo schema:
{{"editorial_contract_version":1,
"editorial_anchors":{{"opening":["EV001_0001"],"inciting_incident":["EV001_0002"],
"protagonist_goal":["EV001_0003"],"climax":["EV003_0004"],
"resolution":["EV003_0005"],"ending":["EV003_0006"]}},
"boundary_1_seconds":0.0,"boundary_2_seconds":0.0,"part_weights":[1,1,1],
"selected_event_clusters":[{{"cluster_id":"C001","summary":"","importance":"critical",
"event_ids":["EV001_0001"],"required_context_ids":["EV001_0002"],"reason":""}}],
"story_contract":{{"premise":"","protagonist_goal":"","central_conflict":"","theme":"","ending_meaning":""}},
"characters":[{{"id":"P1","name":"","description":"","role":""}}],
"parts":[{{"part":1,"summary":"","end_state":"","handoff_to_next":""}},
{{"part":2,"summary":"","end_state":"","handoff_to_next":""}},
{{"part":3,"summary":"","end_state":"","handoff_to_next":""}}]}}
"""


def assign_story_beats(events_by_part, max_beat_span_seconds=240.0):
    """Group visual evidence into story ideas without dropping any edit scene.

    A required event remains available as a visual scene, while ``story_beat_id``
    tells the writer that adjacent evidence can illustrate the same narrative
    change instead of becoming a new diary-like plot point.
    """
    result_parts = copy.deepcopy(list(events_by_part or []))
    beats_by_part = []
    next_beat_number = 1
    major_functions = {
        "setup", "inciting_incident", "turning_point", "false_victory",
        "reversal", "climax", "resolution", "ending",
    }
    function_priority = {
        "ending": 100, "resolution": 90, "climax": 85, "reversal": 80,
        "inciting_incident": 75, "turning_point": 70,
        "false_victory": 65, "setup": 55, "development": 10,
    }

    for events in result_parts:
        ordered = sorted(events, key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ))
        groups = []
        current = []
        group_start = None
        current_cluster = ""
        for event in ordered:
            start = float(event.get("start_seconds", 0.0))
            narrative_function = str(
                event.get("narrative_function", "development")
            ).lower()
            temporal_relation = str(
                event.get("temporal_relation", "continuation")
            ).lower()
            elapsed = 0.0 if group_start is None else start - group_start
            event_cluster = str(event.get("selection_cluster_id", "")).strip()
            begins_major_change = (
                narrative_function in major_functions
                or str(event.get("story_role", "")).lower()
                in {"climax", "resolution", "ending"}
                or bool(str(event.get("required_for", "")).strip())
            )
            temporal_break = temporal_relation in {
                "time_jump", "flashback", "flashforward", "parallel", "return",
            }
            should_split = bool(current) and (
                (event_cluster and current_cluster and event_cluster != current_cluster)
                or (not event_cluster and bool(current_cluster))
                or (event_cluster and not current_cluster)
                or elapsed >= float(max_beat_span_seconds)
                or (begins_major_change and elapsed >= 60.0)
                or (temporal_break and elapsed >= 90.0)
            )
            if should_split:
                groups.append(current)
                current = []
                group_start = None
            if not current:
                group_start = start
                current_cluster = event_cluster
            current.append(event)
        if current:
            groups.append(current)

        part_beats = []
        annotated = []
        for group in groups:
            core = max(group, key=lambda event: (
                function_priority.get(
                    str(event.get("narrative_function", "development")).lower(), 0
                ),
                20 if str(event.get("importance", "")).lower() == "main" else 0,
                10 if str(event.get("required_for", "")).strip() else 0,
            ))
            core_id = str(core.get("event_id", "")).strip().upper()
            is_climax_beat = (
                str(core.get("story_role", "")).lower() == "climax"
                or str(core.get("selection_importance", "")).lower() == "critical"
                or any(
                    str(event.get("story_role", "")).lower() == "climax"
                    or str(event.get("narrative_function", "")).lower()
                    in {"reversal", "climax"}
                    for event in group
                )
            )
            climax_spine = {}
            if is_climax_beat:
                # A climax may use many shots, but its causal spine must not be
                # reduced to the single core event. Prefer real evidence over
                # slots mechanically split only for visual coverage.
                spine_candidates = [
                    event for event in group
                    if not bool(event.get("derived_visual_slot"))
                    and str(event.get("event_id", "")).strip()
                ] or [
                    event for event in group
                    if str(event.get("event_id", "")).strip()
                ]
                if spine_candidates:
                    for spine_index, spine_event in enumerate(spine_candidates):
                        spine_id = str(spine_event["event_id"]).strip().upper()
                        if spine_index == 0:
                            spine_role = "pressure"
                        elif spine_index == len(spine_candidates) - 1:
                            spine_role = "outcome"
                        elif str(spine_event.get("event_kind", "")).lower() in {
                            "reversal", "transformation", "reveal",
                        }:
                            spine_role = "turning_point"
                        else:
                            spine_role = "action_chain"
                        climax_spine[spine_id] = spine_role
                    climax_spine[core_id] = "turning_point"
            beat_id = f"B{next_beat_number:03d}"
            next_beat_number += 1
            cause = next((
                re.sub(r"\s+", " ", str(event.get("cause", ""))).strip()
                for event in group
                if str(event.get("cause", "")).strip()
            ), "")
            outcome = next((
                re.sub(r"\s+", " ", str(
                    event.get("consequence") or event.get("change") or ""
                )).strip()
                for event in reversed(group)
                if str(event.get("consequence") or event.get("change") or "").strip()
            ), "")
            summary = re.sub(r"\s+", " ", str(core.get("event", ""))).strip()
            evidence_ids = [
                str(event.get("event_id", "")).strip().upper()
                for event in group if str(event.get("event_id", "")).strip()
            ]
            part_beats.append({
                "story_beat_id": beat_id,
                "summary": summary,
                "cause": cause,
                "outcome": outcome,
                "story_role": str(core.get("story_role", "development")).lower(),
                "evidence_ids": evidence_ids,
                "climax_spine_ids": list(climax_spine),
            })
            for position, event in enumerate(group, start=1):
                event_id = str(event.get("event_id", "")).strip().upper()
                event["story_beat_id"] = beat_id
                event["story_beat_summary"] = summary
                event["story_beat_cause"] = cause
                event["story_beat_outcome"] = outcome
                event["story_beat_position"] = position
                event["story_beat_size"] = len(group)
                event["climax_spine_role"] = climax_spine.get(event_id, "")
                # Duplicate shots were folded into visual_sources before this
                # stage, so every surviving selected event owns a distinct
                # narration fact even when several facts share one story beat.
                event["visual_support_only"] = False
                annotated.append(event)
        beats_by_part.append(part_beats)
        events[:] = sorted(annotated, key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ))
    return result_parts, beats_by_part


def _normalized_fact_key(value):
    value = re.sub(r"[^\w\s]", " ", str(value or "").casefold())
    return " ".join(value.split())


def event_narration_foci(event):
    """Return distinct, evidence-backed facts that one event can narrate."""
    if event.get('grounded_evidence_version'):
        return list(dict.fromkeys(f['text'] for f in event.get('grounded_facts') or []))
    fields = (
        "state_before", "cause", "action", "event", "visual_detail",
        "change", "result", "consequence", "state_after", "outcome",
    )
    result = []
    token_sets = []
    values = ([str(value) for value in event.get("visual_supported_facts") or []]
              if event.get("visual_review_version") else [event.get(field, "") for field in fields])
    if not values and event.get("visual_review_version"):
        values = [source.get("event", "") for source in event.get("visual_sources") or []]
    for raw_value in values:
        value = re.sub(r"\s+", " ", str(raw_value)).strip()
        key = _normalized_fact_key(value)
        if not key:
            continue
        tokens = set(key.split())
        if any(
            tokens and known and len(tokens & known) / max(1, min(len(tokens), len(known))) >= 0.82
            for known in token_sets
        ):
            continue
        result.append(value)
        token_sets.append(tokens)
    return result


def event_narration_capacity(
    event, min_slot_seconds=MIN_NARRATION_ANCHOR_SECONDS,
):
    """Count evidence-backed sentence units available inside one event.

    A broad event may contain separate setup/cause, action and consequence
    facts. It can support several narration sentences only when those facts
    are distinct and the observed visual window is long enough to give each
    sentence its own footage. Dialogue-only anchors remain one sentence.
    """
    if not isinstance(event, dict):
        return 0
    if str(event.get("source", "")).lower() in {
        "transcript_bridge", "transcript_anchor",
    }:
        return 1
    foci = event_narration_foci(event)
    semantic_capacity = min(4, max(1, len(foci)))
    if event.get('sentence_first'):
        # Facts determine narration capacity. Shot duration/availability is
        # resolved later and must not erase otherwise observed story facts.
        return max(1, len(foci))
    start = float(event.get("start_seconds", 0.0) or 0.0)
    end = float(event.get("end_seconds", start) or start)
    duration = max(0.0, end - start)
    visual_sources = [
        source for source in (event.get("visual_sources") or [])
        if isinstance(source, dict)
        and float(source.get("end_seconds", 0.0) or 0.0)
        > float(source.get("start_seconds", 0.0) or 0.0)
    ]
    if visual_sources:
        # One verified shot grounds the event, not the number of sentences.
        # Separate cause/action/change/outcome facts may become separate
        # narration slots; their additional pictures are selected later from
        # the same bounded event window. Do not collapse four real facts to
        # one sentence merely because Gemini returned one representative shot.
        visual_capacity = semantic_capacity
    else:
        # Compatibility for legacy/unit-test evidence without the new contract.
        visual_capacity = max(
            1, int(duration // max(1.0, float(min_slot_seconds)))
        )
    return min(semantic_capacity, visual_capacity)


def event_required_visual_shot_count(event):
    """Return the soft minimum semantic-beat coverage for one event.

    One observed beat is enough to ground the event.  It does not limit how
    many recap sentences the event may support and is never a demand for a
    fixed number of 2-3 second render shots.
    """
    if not isinstance(event, dict):
        return 0
    return 1


def verified_event_visual_sources(event):
    """Return unique semantic beats explicitly verified by a video model."""
    rows = []
    seen = set()
    for source in (event or {}).get("visual_sources") or []:
        if not isinstance(source, dict) or not source.get("verified_visual_shot"):
            continue
        try:
            start = float(source.get("start_seconds"))
            end = float(source.get("end_seconds"))
        except (TypeError, ValueError):
            continue
        if not (0.05 < end - start <= MAX_GROUNDED_VISUAL_BEAT_SECONDS + 0.05):
            continue
        key = (round(start, 2), round(end, 2))
        if key in seen:
            continue
        seen.add(key)
        rows.append(source)
    return sorted(rows, key=lambda row: (
        float(row.get("start_seconds", 0.0)),
        float(row.get("end_seconds", 0.0)),
    ))


def evidence_narration_capacity(events):
    return sum(event_narration_capacity(event) for event in (events or []) if isinstance(event, dict))


def deduplicate_overlapping_chunk_evidence(chunk_evidence, similarity_threshold=0.72):
    """Remove only obvious duplicates created by overlapping chapter clips."""
    chunks = copy.deepcopy(chunk_evidence or [])
    id_redirect = {}
    removed = 0

    def signature(event):
        text = " ".join(str(event.get(key, "")) for key in (
            "event", "visual_detail", "change", "consequence",
        ))
        return set(_normalized_fact_key(text).split())

    for index in range(1, len(chunks)):
        previous = chunks[index - 1].get("events") or []
        current = chunks[index].get("events") or []
        kept = []
        for event in current:
            duplicate = None
            event_tokens = signature(event)
            for candidate in previous:
                overlap = min(float(event.get("end_seconds", 0)), float(candidate.get("end_seconds", 0))) - max(
                    float(event.get("start_seconds", 0)), float(candidate.get("start_seconds", 0))
                )
                starts_close = abs(float(event.get("start_seconds", 0)) - float(candidate.get("start_seconds", 0))) <= 2.0
                candidate_tokens = signature(candidate)
                similarity = len(event_tokens & candidate_tokens) / max(1, min(len(event_tokens), len(candidate_tokens)))
                if (overlap > 0 or starts_close) and similarity >= float(similarity_threshold):
                    duplicate = candidate
                    break
            if duplicate is None:
                kept.append(event)
                continue
            old_id = str(event.get("evidence_id", "")).strip().upper()
            kept_id = str(duplicate.get("evidence_id", "")).strip().upper()
            if old_id and kept_id:
                id_redirect[old_id] = kept_id
            removed += 1
        chunks[index]["events"] = kept

    for chunk in chunks:
        for event in chunk.get("events") or []:
            event["depends_on"] = list(dict.fromkeys(
                id_redirect.get(str(value).strip().upper(), str(value).strip().upper())
                for value in (event.get("depends_on") or [])
                if id_redirect.get(str(value).strip().upper(), str(value).strip().upper())
                != str(event.get("evidence_id", "")).strip().upper()
            ))
    return chunks, removed


def stitch_cross_chunk_event_continuations(
    chunk_evidence, maximum_boundary_distance=3 * 60.0,
):
    """Link unfinished events across adjacent 15+3 minute windows.

    This never deletes an event.  It only gives fragments a stable parent flow
    and a causal dependency, leaving later compression free to keep every
    observed timestamp as a visual source.
    """
    chunks = copy.deepcopy(list(chunk_evidence or []))
    stitched = 0
    unresolved = []

    def character_ids(event):
        return {
            str(value).strip().upper()
            for value in (event.get("characters") or []) if str(value).strip()
        }

    def compatible(left, right):
        left_characters = character_ids(left)
        right_characters = character_ids(right)
        if left_characters and right_characters and not (
            left_characters & right_characters
        ):
            return False
        left_parent = str(left.get("parent_cluster_id", "")).strip().casefold()
        right_parent = str(right.get("parent_cluster_id", "")).strip().casefold()
        return not left_parent or not right_parent or left_parent == right_parent

    flow_number = 1
    for chunk_index in range(len(chunks) - 1):
        left_events = sorted(
            [row for row in (chunks[chunk_index].get("events") or []) if isinstance(row, dict)],
            key=lambda row: float(row.get("start_seconds", 0.0)),
        )
        right_events = sorted(
            [row for row in (chunks[chunk_index + 1].get("events") or []) if isinstance(row, dict)],
            key=lambda row: float(row.get("start_seconds", 0.0)),
        )
        boundary = float(
            chunks[chunk_index].get(
                "primary_end_seconds", chunks[chunk_index].get("end_seconds", 0.0)
            )
        )
        open_left = [
            row for row in left_events
            if bool(row.get("continues_to_next"))
            or str(row.get("completion_status", "")).lower() == "open"
        ]
        for left in open_left:
            candidates = [
                right for right in right_events
                if compatible(left, right)
                and float(right.get("start_seconds", 0.0))
                <= boundary + float(maximum_boundary_distance)
                and (
                    bool(right.get("continues_from_previous"))
                    or float(right.get("start_seconds", 0.0)) <= boundary + 45.0
                )
            ]
            if not candidates:
                unresolved.append(str(left.get("evidence_id", "")).strip().upper())
                continue
            left_end = float(left.get("end_seconds", boundary))
            right = min(candidates, key=lambda row: (
                0 if bool(row.get("continues_from_previous")) else 1,
                abs(float(row.get("start_seconds", boundary)) - left_end),
            ))
            parent = str(
                left.get("parent_cluster_id") or right.get("parent_cluster_id")
                or f"boundary_flow_{flow_number:04d}"
            ).strip().casefold()
            flow_number += 1
            left["parent_cluster_id"] = parent
            left["continuation_stitched"] = True
            right["parent_cluster_id"] = parent
            right["continues_from_previous"] = True
            right["depends_on"] = list(dict.fromkeys(
                list(right.get("depends_on") or [])
                + [str(left.get("evidence_id", "")).strip().upper()]
            ))
            stitched += 1
    return chunks, stitched, [value for value in unresolved if value]


def deduplicate_semantic_chunk_evidence(
    chunk_evidence, similarity_threshold=0.72, max_gap_seconds=180.0,
):
    """Collapse repeated story facts even when their shots do not overlap.

    Gemini may describe the same order, emotion, or outcome with different
    wording and timestamps.  This pass is deliberately conservative: it keeps
    distinct causal steps and only folds events that share a model fact label,
    a repeated-state label, or a strong structured subject/action/result match.
    Removed shots remain attached to the surviving fact as ``visual_sources``.
    """
    chunks = copy.deepcopy(chunk_evidence or [])
    rows = []
    for chunk_index, chunk in enumerate(chunks):
        for event in chunk.get("events") or []:
            if isinstance(event, dict):
                rows.append((chunk_index, event))
    rows.sort(key=lambda item: (
        float(item[1].get("start_seconds", 0.0)),
        float(item[1].get("end_seconds", 0.0)),
    ))
    removed_ids = set()
    id_redirect = {}
    survivors = []

    def tokens(value):
        return {
            token for token in re.findall(
                r"[^\W_]{2,}", str(value or "").casefold(), flags=re.UNICODE
            ) if token not in REPETITION_STOPWORDS
        }

    def similarity(left, right):
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, min(len(left), len(right)))

    def characters(event):
        return {
            str(value).strip().casefold()
            for value in (event.get("characters") or []) if str(value).strip()
        }

    def compatible_identity(left, right):
        left_ids, right_ids = characters(left), characters(right)
        return not left_ids or not right_ids or bool(left_ids & right_ids)

    def field_tokens(event, field):
        return tokens(event.get(field, ""))

    def prose_tokens(event):
        return tokens(" ".join(str(event.get(field, "")) for field in (
            "event", "change", "consequence", "result", "state_after",
        )))

    structural_functions = {
        "inciting_incident", "turning_point", "false_victory", "reversal",
        "climax", "resolution", "ending",
    }

    def same_fact(left, right):
        if not compatible_identity(left, right):
            return False
        gap = abs(
            float(right.get("start_seconds", 0.0))
            - float(left.get("start_seconds", 0.0))
        )
        if gap > float(max_gap_seconds):
            return False
        left_key = str(left.get("fact_key", "")).strip().casefold()
        right_key = str(right.get("fact_key", "")).strip().casefold()
        left_kind = str(left.get("event_kind", "action")).lower()
        right_kind = str(right.get("event_kind", "action")).lower()
        overall = similarity(prose_tokens(left), prose_tokens(right))
        if (
            left_key and right_key and left_key == right_key
            and left_kind == right_kind and overall >= 0.35
        ):
            return True

        left_repeat = str(left.get("repeat_group", "")).strip().casefold()
        right_repeat = str(right.get("repeat_group", "")).strip().casefold()
        if left_repeat and left_repeat == right_repeat and overall >= 0.45:
            return True

        left_function = str(left.get("narrative_function", "")).lower()
        right_function = str(right.get("narrative_function", "")).lower()
        if left_function in structural_functions or right_function in structural_functions:
            return False
        if left_kind != right_kind:
            return False

        structured_fields = ("subject", "action", "object", "result")
        structured_pairs = [
            (field_tokens(left, field), field_tokens(right, field))
            for field in structured_fields
        ]
        comparable = [(a, b) for a, b in structured_pairs if a and b]
        if len(comparable) >= 2:
            scores = [similarity(a, b) for a, b in comparable]
            action_score = similarity(
                field_tokens(left, "action"), field_tokens(right, "action")
            )
            if action_score >= 0.66 and min(scores) >= 0.50 and overall >= 0.52:
                return True
        return overall >= float(similarity_threshold)

    def source_rows(event):
        existing = [
            copy.deepcopy(row) for row in (event.get("visual_sources") or [])
            if isinstance(row, dict)
        ]
        if existing:
            return existing
        return [{
            "evidence_id": str(event.get("evidence_id", "")).strip().upper(),
            "start_seconds": float(event.get("start_seconds", 0.0)),
            "end_seconds": float(event.get("end_seconds", 0.0)),
            "event": str(event.get("event", "")).strip(),
        }]

    for _chunk_index, event in rows:
        duplicate = None
        for candidate in reversed(survivors):
            if (
                float(event.get("start_seconds", 0.0))
                - float(candidate.get("start_seconds", 0.0))
                > float(max_gap_seconds)
            ):
                break
            if same_fact(candidate, event):
                duplicate = candidate
                break
        if duplicate is None:
            event.setdefault("visual_sources", source_rows(event))
            event.setdefault("merged_evidence_ids", [
                str(event.get("evidence_id", "")).strip().upper()
            ])
            survivors.append(event)
            continue

        old_id = str(event.get("evidence_id", "")).strip().upper()
        kept_id = str(duplicate.get("evidence_id", "")).strip().upper()
        if old_id:
            removed_ids.add(old_id)
        if old_id and kept_id:
            id_redirect[old_id] = kept_id
        duplicate["visual_sources"] = source_rows(duplicate) + source_rows(event)
        duplicate["merged_evidence_ids"] = list(dict.fromkeys(
            list(duplicate.get("merged_evidence_ids") or [kept_id])
            + list(event.get("merged_evidence_ids") or [old_id])
        ))
        duplicate["semantic_duplicates_collapsed"] = int(
            duplicate.get("semantic_duplicates_collapsed", 0)
        ) + 1
        for field in (
            "cause", "change", "consequence", "result", "state_before",
            "state_after", "subject", "action", "object", "fact_key",
        ):
            if not str(duplicate.get(field, "")).strip() and str(event.get(field, "")).strip():
                duplicate[field] = event.get(field)

    for chunk in chunks:
        chunk["events"] = [
            event for event in (chunk.get("events") or [])
            if str(event.get("evidence_id", "")).strip().upper() not in removed_ids
        ]
        for event in chunk["events"]:
            event["depends_on"] = list(dict.fromkeys(
                id_redirect.get(str(value).strip().upper(), str(value).strip().upper())
                for value in (event.get("depends_on") or [])
                if id_redirect.get(str(value).strip().upper(), str(value).strip().upper())
                != str(event.get("evidence_id", "")).strip().upper()
            ))
    return chunks, len(removed_ids)


def assess_story_evidence_coverage(chunk_evidence, target_words, video_duration_sec):
    """Measure usable narration capacity and objective whole-film coverage."""
    duration = max(1.0, float(video_duration_sec))
    events = [
        event for chunk in (chunk_evidence or [])
        for event in (chunk.get("events") or []) if isinstance(event, dict)
    ]
    events.sort(key=lambda row: (float(row.get("start_seconds", 0)), float(row.get("end_seconds", 0))))
    required = required_tts_scene_count(target_words)
    empty_chunks = [
        index for index, chunk in enumerate(chunk_evidence or [])
        if not (chunk.get("events") or [])
    ]
    visual_grounding_chunk_indexes = [
        index for index, chunk in enumerate(chunk_evidence or [])
        if any(
            isinstance(event, dict)
            and len(verified_event_visual_sources(event))
            < event_required_visual_shot_count(event)
            for event in (chunk.get("events") or [])
        )
    ]
    missing_visual_event_ids = [
        str(event.get("evidence_id", "")).strip().upper()
        for event in events
        if len(verified_event_visual_sources(event))
        < event_required_visual_shot_count(event)
    ]
    last = events[-1] if events else {}
    last_end = float(last.get("end_seconds", 0) or 0)
    story_role = str(last.get("story_role", "")).lower()
    narrative_function = str(last.get("narrative_function", "")).lower()
    terminal_state = bool(str(last.get("outcome") or last.get("state_after") or last.get("consequence") or "").strip())
    still_active = (
        story_role == "climax" or narrative_function == "climax"
        or str(last.get("event_kind", "")).lower() in {"attack", "defense"}
    )
    has_ending = (
        last_end >= duration * 0.88 and terminal_state and not still_active
        and bool({story_role, narrative_function} & {"ending", "resolution"})
    )
    action_count = sum(
        str(event.get("story_role", "")).lower() == "climax"
        or str(event.get("narrative_function", "")).lower() in {"climax", "reversal"}
        or str(event.get("event_kind", "")).lower() in {"attack", "defense", "transformation", "reversal"}
        for event in events
    )
    capacity = evidence_narration_capacity(events)
    issues = []
    if empty_chunks:
        issues.append("empty_chapters")
    if capacity < required:
        issues.append("insufficient_fact_capacity")
    if not has_ending:
        issues.append("missing_terminal_ending")
    if missing_visual_event_ids:
        issues.append("missing_visual_grounding")
    return {
        "event_count": len(events), "narration_capacity": capacity,
        "required_capacity": required, "capacity_deficit": max(0, required - capacity),
        "empty_chunk_indexes": empty_chunks, "has_terminal_ending": has_ending,
        "visual_grounding_chunk_indexes": visual_grounding_chunk_indexes,
        "missing_visual_event_ids": missing_visual_event_ids,
        "last_event_end": last_end, "action_event_count": action_count,
        "issues": issues,
    }


def ensure_grounded_terminal_ending(chunk_evidence, video_duration_sec):
    """Promote only a verified late resolution; never call an active fight an ending."""
    chunks = copy.deepcopy(chunk_evidence or [])
    events = [
        event for chunk in chunks for event in (chunk.get("events") or [])
        if isinstance(event, dict)
    ]
    if not events:
        return chunks, False
    events.sort(key=lambda row: (float(row.get("start_seconds", 0)), float(row.get("end_seconds", 0))))
    last = events[-1]
    story_role = str(last.get("story_role", "")).lower()
    narrative_function = str(last.get("narrative_function", "")).lower()
    if "ending" in {story_role, narrative_function}:
        return chunks, False
    near_end = float(last.get("end_seconds", 0)) >= max(1.0, float(video_duration_sec)) * 0.88
    has_outcome = bool(str(last.get("outcome") or last.get("state_after") or last.get("consequence") or "").strip())
    active_action = (
        "climax" in {story_role, narrative_function}
        or str(last.get("event_kind", "")).lower() in {"attack", "defense"}
    )
    if near_end and has_outcome and not active_action and "resolution" in {story_role, narrative_function}:
        last["story_role"] = "ending"
        last["narrative_function"] = "ending"
        last["ending_inferred_by_backend"] = True
        return chunks, True
    return chunks, False


def expand_evidence_to_visual_slots(
    events, required_scene_count,
    min_slot_seconds=MIN_NARRATION_ANCHOR_SECONDS,
    require_verified_visuals=False,
):
    """Allocate the complete event ledger into a locked global sentence plan.

    Adjacent minor events may share one slot when the event ledger is denser
    than the requested sentence count. Rich events may receive several slots,
    but every slot owns a distinct evidence-backed focus and an ordered slice
    of source footage. Allocation finishes before any recap prose is written.
    """
    source_events = [copy.deepcopy(event) for event in (events or []) if isinstance(event, dict)]
    required = max(1, int(required_scene_count))
    if not source_events:
        raise ValueError("EVENT_CAPACITY_INSUFFICIENT: Khong co su kien de phan bo cau.")
    # When the map contains more atomic events than sentences, preserve all
    # events by merging only adjacent causal rows into sentence clusters.
    if len(source_events) > required:
        compressed, _ = compress_story_events_to_scene_budget(
            [source_events], required * MIN_TTS_WORDS,
        )
        source_events = compressed[0]
    total_capacity = evidence_narration_capacity(source_events)
    if total_capacity < required:
        raise ValueError(
            f"EVENT_CAPACITY_INSUFFICIENT: Bang chung chi du suc viet {total_capacity} "
            f"cau khong lap; can {required} cau muc tieu."
        )

    allocations = [1] * len(source_events)

    def allocation_priority(index):
        event = source_events[index]
        role = str(event.get("story_role", "development")).lower()
        function = str(event.get("narrative_function", "development")).lower()
        importance = str(event.get("selection_importance", "")).lower()
        structural = (
            80 if role == "climax" or function in {"climax", "reversal"}
            else 60 if function in {"turning_point", "inciting_incident"}
            else 45 if role == "resolution"
            else 35 if importance == "critical"
            else 20
        )
        duration = max(
            0.0,
            float(event.get("end_seconds", 0.0))
            - float(event.get("start_seconds", 0.0)),
        )
        # Spread extra sentences across the whole film before giving a fourth
        # sentence to one dramatic event.
        return structural + min(30.0, duration / 4.0) - allocations[index] * 24

    while sum(allocations) < required:
        candidates = [
            index for index, event in enumerate(source_events)
            if allocations[index] < event_narration_capacity(
                event, min_slot_seconds=min_slot_seconds
            )
        ]
        if not candidates:
            break
        chosen = max(candidates, key=lambda index: (
            allocation_priority(index), -index
        ))
        allocations[chosen] += 1

    if sum(allocations) != required:
        raise ValueError(
            f"EVENT_CAPACITY_INSUFFICIENT: Chi phan bo duoc {sum(allocations)} "
            f"tren {required} cau muc tieu."
        )

    def source_ranges(event):
        rows = []
        for source in verified_event_visual_sources(event):
            start = float(source.get("start_seconds", 0.0) or 0.0)
            end = float(source.get("end_seconds", start) or start)
            if end > start:
                row = copy.deepcopy(source)
                row["start_seconds"] = start
                row["end_seconds"] = end
                rows.append(row)
        if not rows and not require_verified_visuals and not (event.get("visual_review_version") or event.get("visual_ranges_locked")):
            start = float(event.get("start_seconds", 0.0) or 0.0)
            end = float(event.get("end_seconds", start) or start)
            if end > start:
                rows.append({
                    "evidence_id": str(event.get("evidence_id", "")).strip().upper(),
                    "start_seconds": start,
                    "end_seconds": end,
                    "event": str(event.get("event", "")).strip(),
                    "verified_visual_shot": False,
                    "visual_match_tier": "same_event_window",
                    "visual_fallback_reason": "compatibility_only",
                })
        return sorted(rows, key=lambda row: (
            row["start_seconds"], row["end_seconds"]
        ))

    def partition_ranges(event, count):
        """Retain observed action ranges within each sentence's event band.

        Verified shots rank first. Missing secondary shots are sampled from the
        sentence's chronological slice of the same event, which is the safest
        deterministic "nearest scene" fallback. We never cross into a
        different event merely to fill TTS duration.
        """
        verified = source_ranges(event)
        if event.get("visual_review_version") or event.get("visual_ranges_locked"):
            # Partition the concatenated OBSERVED footage, not the enclosing
            # start/end window: the latter can include credits or other actions
            # between two repaired shots.
            coverage = sum(source['end_seconds'] - source['start_seconds'] for source in verified)
            if coverage <= .05 * count:
                raise ValueError("VISUAL_GROUNDING_MISSING: " + str(event.get('evidence_id', 'UNKNOWN')))
            groups = []
            for index in range(count):
                low, high = coverage * index / count, coverage * (index + 1) / count
                cursor, pieces = 0.0, []
                for source in verified:
                    span = source['end_seconds'] - source['start_seconds']
                    a, b = max(low, cursor), min(high, cursor + span)
                    if b - a > .05:
                        piece = copy.deepcopy(source)
                        piece['start_seconds'] = source['start_seconds'] + a - cursor
                        piece['end_seconds'] = source['start_seconds'] + b - cursor
                        pieces.append(piece)
                    cursor += span
                if not pieces:
                    raise ValueError("VISUAL_GROUNDING_MISSING: " + str(event.get('evidence_id', 'UNKNOWN')))
                groups.append(pieces)
            return groups
        event_start = float(event.get("start_seconds", 0.0) or 0.0)
        event_end = float(event.get("end_seconds", event_start) or event_start)
        if event_end <= event_start:
            return [copy.deepcopy(verified) for _ in range(count)]

        groups = []
        slot_count = max(1, int(count))
        total_duration = event_end - event_start
        for slot_index in range(slot_count):
            band_start = event_start + total_duration * slot_index / slot_count
            band_end = event_start + total_duration * (slot_index + 1) / slot_count
            band_duration = max(0.05, band_end - band_start)
            pieces = []

            # Preserve any shot Gemini explicitly verified inside this slice.
            for source in verified:
                overlap_start = max(band_start, source["start_seconds"])
                overlap_end = min(band_end, source["end_seconds"])
                if overlap_end - overlap_start <= 0.05:
                    continue
                piece = copy.deepcopy(source)
                # A semantic beat may cross several narration slots. Give
                # each slot only its own intersection instead of copying the
                # complete beat into every sentence; complete copies caused a
                # previous line's BEAT to overlap the next line's TIME.
                piece_start = overlap_start
                if pieces:
                    piece_start = max(
                        piece_start,
                        max(float(row["end_seconds"]) for row in pieces),
                    )
                piece_end = overlap_end
                if not source.get("verified_visual_shot"):
                    piece_end = min(
                        piece_end,
                        piece_start + MAX_GROUNDED_VISUAL_BEAT_SECONDS,
                    )
                if piece_end - piece_start <= 0.05:
                    continue
                piece["start_seconds"] = piece_start
                piece["end_seconds"] = piece_end
                piece["visual_match_tier"] = str(
                    source.get("visual_match_tier")
                    or (
                        "verified_exact"
                        if source.get("verified_visual_shot")
                        else "same_event_window"
                    )
                )
                pieces.append(piece)

            if not pieces and not require_verified_visuals:
                shot_end = min(band_end, band_start + MAX_GROUNDED_VISUAL_BEAT_SECONDS)
                if shot_end > band_start + 0.05:
                    pieces = [{
                        "evidence_id": str(event.get("evidence_id", "")).strip().upper(),
                        "start_seconds": band_start,
                        "end_seconds": shot_end,
                        "event": str(event.get("event", "")).strip(),
                        "verified_visual_shot": False,
                        "visual_match_tier": "same_event_nearest",
                        "visual_fallback_reason": "compatibility_only",
                    }]

            if not pieces and verified:
                fallback = copy.deepcopy(verified[min(slot_index, len(verified) - 1)])
                fallback["visual_match_tier"] = "verified_reuse"
                fallback["visual_fallback_reason"] = "no_distinct_same_event_shot"
                pieces = [fallback]
            if not pieces:
                raise ValueError(
                    "VISUAL_GROUNDING_MISSING: "
                    + str(event.get("evidence_id", "UNKNOWN"))
                )
            pieces.sort(key=lambda row: (
                float(row.get("start_seconds", 0.0)),
                float(row.get("end_seconds", 0.0)),
            ))
            groups.append(pieces)
        return groups

    slots = []
    for source_index, (source, slot_count) in enumerate(
        zip(source_events, allocations), start=1
    ):
        original_id = str(source.get("evidence_id") or f"LEGACY_VIS{source_index:05d}").strip().upper()
        foci = event_narration_foci(source) or [str(source.get("event", "")).strip()]
        visual_groups = partition_ranges(source, slot_count)
        used_foci = []
        for slot_index in range(slot_count):
            row = copy.deepcopy(source)
            focus = str(foci[min(slot_index, len(foci) - 1)]).strip()
            row["source_evidence_id"] = original_id
            row["narration_focus"] = focus
            base_fact_id = str(
                source.get("fact_key") or original_id
            ).strip()
            row["fact_id"] = (
                f"{base_fact_id}_FACT{slot_index + 1:02d}"
                if slot_count > 1 else base_fact_id
            )
            row["forbidden_previous_facts"] = list(dict.fromkeys(
                list(source.get("forbidden_previous_facts") or []) + used_foci
            ))
            row["visual_support_only"] = False
            row["derived_narration_slot"] = slot_count > 1
            row["narration_slot_index"] = slot_index + 1
            row["narration_slot_count"] = slot_count
            row["visual_sources"] = visual_groups[slot_index]
            for visual_source in row["visual_sources"]:
                visual_source["narration_focus"] = focus
            if visual_groups[slot_index]:
                row["start_seconds"] = min(
                    item["start_seconds"] for item in visual_groups[slot_index]
                )
                row["end_seconds"] = max(
                    item["end_seconds"] for item in visual_groups[slot_index]
                )
            slots.append(row)
            used_foci.append(focus)
    return sorted(slots, key=lambda event: (
        float(event.get("start_seconds", 0.0)),
        int(event.get("narration_slot_index", 1)),
        float(event.get("end_seconds", 0.0)),
    ))


def coalesce_dense_story_events(
    events_by_part, minimum_anchor_seconds=MIN_NARRATION_ANCHOR_SECONDS,
):
    """Merge narration anchors that are too close for one natural TTS each.

    All grounded visual ranges remain in ``visual_sources``. The merged row is
    one narration fact spanning those shots, so the renderer may play real
    footage instead of freezing a sub-second anchor under a 15-25 word line.
    """
    role_priority = {"development": 1, "resolution": 2, "climax": 3, "ending": 4}
    function_priority = {
        "development": 1, "setup": 2, "inciting_incident": 3,
        "turning_point": 4, "false_victory": 4, "reversal": 5,
        "climax": 6, "resolution": 7, "ending": 8,
    }
    result = []
    collapsed = 0

    def source_row(event):
        return {
            "evidence_id": str(
                event.get("source_evidence_id") or event.get("evidence_id") or ""
            ).strip().upper(),
            "start_seconds": float(event.get("start_seconds", 0.0)),
            "end_seconds": float(event.get("end_seconds", 0.0)),
            "event": str(event.get("event", "")).strip(),
            "narration_focus": str(event.get("narration_focus", "")).strip(),
        }

    for raw_part in list(events_by_part or []):
        ordered = sorted(
            (copy.deepcopy(event) for event in (raw_part or []) if isinstance(event, dict)),
            key=lambda event: (
                float(event.get("start_seconds", 0.0)),
                float(event.get("end_seconds", 0.0)),
            ),
        )
        part = []
        for event in ordered:
            event.setdefault("visual_sources", [source_row(event)])
            event.setdefault("merged_evidence_ids", [
                str(event.get("evidence_id", "")).strip().upper()
            ])
            if not part:
                part.append(event)
                continue
            previous = part[-1]
            anchor_gap = (
                float(event.get("start_seconds", 0.0))
                - float(previous.get("start_seconds", 0.0))
            )
            if anchor_gap >= float(minimum_anchor_seconds):
                part.append(event)
                continue

            collapsed += 1
            previous["visual_sources"].extend(event.get("visual_sources") or [source_row(event)])
            previous["merged_evidence_ids"] = list(dict.fromkeys(
                list(previous.get("merged_evidence_ids") or [])
                + list(event.get("merged_evidence_ids") or [])
            ))
            previous["end_seconds"] = max(
                float(previous.get("end_seconds", 0.0)),
                float(event.get("end_seconds", 0.0)),
            )
            facts = []
            for value in (
                previous.get("narration_focus"), event.get("narration_focus"),
                previous.get("event"), event.get("event"),
                previous.get("consequence"), event.get("consequence"),
            ):
                value = re.sub(r"\s+", " ", str(value or "")).strip()
                if value and _normalized_fact_key(value) not in {
                    _normalized_fact_key(known) for known in facts
                }:
                    facts.append(value)
            previous["supporting_facts"] = facts
            previous["forbidden_previous_facts"] = list(dict.fromkeys(
                list(previous.get("forbidden_previous_facts") or [])
                + list(event.get("forbidden_previous_facts") or [])
            ))
            previous["narration_focus"] = "; ".join(facts[:2])
            if role_priority.get(str(event.get("story_role", "development")).lower(), 0) > role_priority.get(
                str(previous.get("story_role", "development")).lower(), 0
            ):
                previous["story_role"] = event.get("story_role")
            if function_priority.get(str(event.get("narrative_function", "development")).lower(), 0) > function_priority.get(
                str(previous.get("narrative_function", "development")).lower(), 0
            ):
                previous["narrative_function"] = event.get("narrative_function")
                previous["story_beat_id"] = event.get("story_beat_id")
                previous["story_beat_summary"] = event.get("story_beat_summary")
                previous["story_beat_outcome"] = event.get("story_beat_outcome")
        result.append(part)
    return result, collapsed


def compress_story_events_to_scene_budget(
    events_by_part, target_words, maximum_events_per_scene=4,
):
    """Compress narration slots without discarding any grounded event.

    The word budget determines the maximum number of 15-word TTS scenes. When
    evidence is denser, adjacent causal events become one narration cluster and
    every source timestamp remains available through ``visual_sources`` and
    ``covered_events``.
    """
    parts = [
        sorted(
            (copy.deepcopy(event) for event in (part or []) if isinstance(event, dict)),
            key=lambda event: float(event.get("start_seconds", 0.0)),
        )
        for part in list(events_by_part or [])
    ]
    maximum_scenes = max(1, int(target_words) // MIN_TTS_WORDS)
    total_events = sum(len(part) for part in parts)
    target_scenes = min(total_events, maximum_scenes)
    if total_events <= target_scenes:
        return parts, 0

    structural = {
        "inciting_incident", "turning_point", "false_victory", "reversal",
        "climax", "resolution", "ending",
    }

    def source_row(event):
        return {
            "evidence_id": str(
                event.get("source_evidence_id") or event.get("evidence_id") or ""
            ).strip().upper(),
            "start_seconds": float(event.get("start_seconds", 0.0)),
            "end_seconds": float(event.get("end_seconds", 0.0)),
            "event": str(event.get("event", "")).strip(),
            "cause": str(event.get("cause", "")).strip(),
            "change": str(event.get("change", "")).strip(),
            "consequence": str(event.get("consequence", "")).strip(),
        }

    for part in parts:
        for event in part:
            event.setdefault("visual_sources", [source_row(event)])
            event.setdefault("covered_events", [source_row(event)])
            event.setdefault("merged_evidence_ids", [
                str(event.get("evidence_id", "")).strip().upper()
            ])

    def group_size(event):
        return max(1, len(event.get("covered_events") or []))

    def merge_cost(left, right, enforce_limit=True):
        if enforce_limit and group_size(left) + group_size(right) > int(maximum_events_per_scene):
            return None
        gap = max(
            0.0,
            float(right.get("start_seconds", 0.0))
            - float(left.get("end_seconds", 0.0)),
        )
        left_parent = str(left.get("parent_cluster_id", "")).strip().casefold()
        right_parent = str(right.get("parent_cluster_id", "")).strip().casefold()
        left_cluster = str(left.get("event_cluster", "")).strip().casefold()
        right_cluster = str(right.get("event_cluster", "")).strip().casefold()
        same_flow = bool(
            (left_parent and left_parent == right_parent)
            or (left_cluster and left_cluster == right_cluster)
        )
        left_function = str(left.get("narrative_function", "")).lower()
        right_function = str(right.get("narrative_function", "")).lower()
        structural_penalty = 600.0 if (
            left_function in structural and right_function in structural
            and left_function != right_function
        ) else 0.0
        return gap + structural_penalty - (1000.0 if same_flow else 0.0)

    def merge_pair(left, right):
        merged = copy.deepcopy(left)
        merged["visual_sources"] = list(left.get("visual_sources") or [source_row(left)]) + list(
            right.get("visual_sources") or [source_row(right)]
        )
        merged["covered_events"] = list(left.get("covered_events") or [source_row(left)]) + list(
            right.get("covered_events") or [source_row(right)]
        )
        merged["merged_evidence_ids"] = list(dict.fromkeys(
            list(left.get("merged_evidence_ids") or [])
            + list(right.get("merged_evidence_ids") or [])
        ))
        merged["start_seconds"] = min(
            float(left.get("start_seconds", 0.0)), float(right.get("start_seconds", 0.0))
        )
        merged["end_seconds"] = max(
            float(left.get("end_seconds", 0.0)), float(right.get("end_seconds", 0.0))
        )
        facts = []
        for row in merged["covered_events"]:
            value = re.sub(r"\s+", " ", str(
                row.get("consequence") or row.get("change") or row.get("event") or ""
            )).strip()
            if value and _normalized_fact_key(value) not in {
                _normalized_fact_key(known) for known in facts
            }:
                facts.append(value)
        merged["supporting_facts"] = facts
        merged["narration_focus"] = "; ".join(facts[:4])
        merged["event"] = " → ".join(
            str(row.get("event", "")).strip()
            for row in merged["covered_events"] if str(row.get("event", "")).strip()
        )
        merged["coverage_cluster"] = True
        # The later structural result owns the cluster's final narrative role.
        if str(right.get("narrative_function", "")).lower() in structural:
            for key in (
                "narrative_function", "story_role", "story_beat_id",
                "story_beat_summary", "story_beat_outcome",
            ):
                if right.get(key):
                    merged[key] = copy.deepcopy(right[key])
        return merged

    collapsed = 0
    while sum(len(part) for part in parts) > target_scenes:
        candidates = []
        for part_index, part in enumerate(parts):
            for event_index in range(len(part) - 1):
                cost = merge_cost(part[event_index], part[event_index + 1], True)
                if cost is not None:
                    candidates.append((cost, part_index, event_index))
        if not candidates:
            for part_index, part in enumerate(parts):
                for event_index in range(len(part) - 1):
                    cost = merge_cost(part[event_index], part[event_index + 1], False)
                    if cost is not None:
                        candidates.append((cost, part_index, event_index))
        if not candidates:
            break
        _, part_index, event_index = min(candidates)
        part = parts[part_index]
        part[event_index:event_index + 2] = [
            merge_pair(part[event_index], part[event_index + 1])
        ]
        collapsed += 1
    return parts, collapsed


def attach_fact_history(events_by_part, maximum=16):
    """Give every writer scene a bounded ledger of already assigned facts."""
    result = copy.deepcopy(list(events_by_part or []))
    history = []
    limit = max(1, int(maximum))
    for part in result:
        for event in sorted(
            part, key=lambda row: float(row.get("start_seconds", 0.0))
        ):
            inherited = [
                re.sub(r"\s+", " ", str(value)).strip()
                for value in (event.get("forbidden_previous_facts") or [])
                if str(value).strip()
            ]
            event["forbidden_previous_facts"] = list(dict.fromkeys(
                inherited + history[-limit:]
            ))[-limit:]
            focus = re.sub(r"\s+", " ", str(
                event.get("narration_focus") or event.get("result")
                or event.get("change") or event.get("event") or ""
            )).strip()
            if focus and _normalized_fact_key(focus) not in {
                _normalized_fact_key(value) for value in history
            }:
                history.append(focus)
    return result


def select_causal_story_evidence(
    events, plan_payload, target_words, boundaries=None,
):
    """Select causal events while keeping the narration spread over the film.

    Critical plot dependencies remain immutable.  Remaining slots balance
    story importance with distance from already selected anchors, preventing a
    climax-heavy score from leaving long unexplained holes in the timeline.
    """
    ordered = sorted(
        (copy.deepcopy(event) for event in (events or []) if isinstance(event, dict)),
        key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ),
    )
    event_by_id = {
        str(event.get("evidence_id", "")).strip().upper(): event
        for event in ordered if str(event.get("evidence_id", "")).strip()
    }
    if not event_by_id:
        return []

    referenced_by = {}
    for event in ordered:
        for dependency in event.get("depends_on") or []:
            dependency = str(dependency).strip().upper()
            if dependency in event_by_id:
                referenced_by[dependency] = referenced_by.get(dependency, 0) + 1

    function_score = {
        "climax": 130, "reversal": 120, "turning_point": 110,
        "inciting_incident": 105, "false_victory": 95, "resolution": 80,
        "ending": 68, "setup": 52, "development": 25,
    }
    kind_score = {
        "transformation": 55, "attack": 52, "defense": 48,
        "reversal": 55, "reveal": 48, "discovery": 42,
        "consequence": 40, "decision": 32, "preparation": 28,
        "action": 26, "resolution": 38, "explanation": 14,
        "travel": -18, "reaction": -12,
    }
    cluster_bonus = {"critical": 80, "main": 35, "support": 5}

    cluster_by_event = {}
    protected_ids = set()
    critical_cluster_ids = set()
    anchor_role_by_event = {}
    editorial_anchors = plan_payload.get("editorial_anchors") or {}
    if isinstance(editorial_anchors, dict):
        for anchor_role, values in editorial_anchors.items():
            if not isinstance(values, list):
                values = [values]
            for value in values:
                evidence_id = str(value).strip().upper()
                if evidence_id not in event_by_id:
                    continue
                protected_ids.add(evidence_id)
                anchor_role_by_event[evidence_id] = str(anchor_role).strip().lower()
    clusters = plan_payload.get("selected_event_clusters") or []
    for cluster_index, cluster in enumerate(clusters, start=1):
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or f"C{cluster_index:03d}").strip()
        importance = str(cluster.get("importance", "main")).strip().lower()
        if importance not in {"critical", "main", "support"}:
            importance = "main"
        ids = []
        for key in ("event_ids", "required_context_ids"):
            values = cluster.get(key) or []
            if not isinstance(values, list):
                values = [values]
            for value in values:
                evidence_id = str(value).strip().upper()
                if evidence_id in event_by_id and evidence_id not in ids:
                    ids.append(evidence_id)
        for evidence_id in ids:
            previous = cluster_by_event.get(evidence_id)
            if previous is None or cluster_bonus[importance] > cluster_bonus[previous[1]]:
                cluster_by_event[evidence_id] = (cluster_id, importance)
        # Critical clusters are immutable causal anchors. Main clusters receive
        # a strong score bonus but are not all forced into the script: when the
        # sentence budget is smaller than the event map, low-value main facts
        # must be allowed to become visual-only instead of inflating narration.
        if importance == "critical":
            protected_ids.update(ids)
        if importance == "critical":
            critical_cluster_ids.update(ids)

    # Guard against an under-specified critical cluster. Pull only a small,
    # nearby action neighborhood so an apparition, transformation or victory
    # cannot survive while its directly preceding setup was discarded.
    index_by_id = {
        str(event.get("evidence_id", "")).strip().upper(): index
        for index, event in enumerate(ordered)
    }
    relevant_kinds = {
        "discovery", "decision", "preparation", "action", "transformation",
        "attack", "defense", "reversal", "reveal", "consequence", "resolution",
    }
    for evidence_id in list(critical_cluster_ids):
        center_index = index_by_id.get(evidence_id)
        if center_index is None:
            continue
        center_start = float(ordered[center_index].get("start_seconds", 0.0))
        before_added = 0
        for neighbor_index in range(center_index - 1, max(-1, center_index - 6), -1):
            neighbor = ordered[neighbor_index]
            if center_start - float(neighbor.get("end_seconds", 0.0)) > 180.0:
                break
            if str(neighbor.get("event_kind", "action")).lower() not in relevant_kinds:
                continue
            protected_ids.add(str(neighbor.get("evidence_id", "")).strip().upper())
            before_added += 1
            if before_added >= 3:
                break
        after_added = 0
        center_end = float(ordered[center_index].get("end_seconds", center_start))
        for neighbor_index in range(center_index + 1, min(len(ordered), center_index + 5)):
            neighbor = ordered[neighbor_index]
            if float(neighbor.get("start_seconds", 0.0)) - center_end > 150.0:
                break
            if str(neighbor.get("event_kind", "action")).lower() not in relevant_kinds:
                continue
            protected_ids.add(str(neighbor.get("evidence_id", "")).strip().upper())
            after_added += 1
            if after_added >= 2:
                break

    # Independently protect explicit structural changes. This is a safety net
    # when Gemini forgets to mention an ID in selected_event_clusters.
    structural_functions = {
        "inciting_incident", "turning_point", "false_victory", "reversal",
        "climax",
    }
    for evidence_id, event in event_by_id.items():
        if (
            str(event.get("narrative_function", "")).lower() in structural_functions
            or str(event.get("required_for", "")).strip()
        ):
            protected_ids.add(evidence_id)

    for terminal_role in ("resolution", "ending"):
        terminal_events = [
            event for event in ordered
            if str(event.get("story_role", "")).lower() == terminal_role
            or str(event.get("narrative_function", "")).lower() == terminal_role
        ]
        if terminal_events:
            protected_ids.add(str(
                terminal_events[-1].get("evidence_id", "")
            ).strip().upper())


    # Keep one real opening and the final real outcome, even if both were
    # conservatively labelled as development by the vision model.
    protected_ids.add(str(ordered[0].get("evidence_id", "")).upper())
    protected_ids.add(str(ordered[-1].get("evidence_id", "")).upper())

    def close_dependencies(seed_ids):
        closed = set(seed_ids)
        pending = list(seed_ids)
        while pending:
            evidence_id = pending.pop()
            event = event_by_id.get(evidence_id)
            if not event:
                continue
            for dependency in event.get("depends_on") or []:
                dependency = str(dependency).strip().upper()
                if dependency in event_by_id and dependency not in closed:
                    closed.add(dependency)
                    pending.append(dependency)
        return closed

    protected_ids = close_dependencies(protected_ids)

    # If one member of a locally observed climax chain is required, keep the
    # whole chain. This prevents a result card from surviving after its
    # transformation, attack or reversal was removed by the word-budget pass.
    local_cluster_members = {}
    for evidence_id, event in event_by_id.items():
        cluster_id = str(event.get("event_cluster", "")).strip().casefold()
        if cluster_id:
            local_cluster_members.setdefault(cluster_id, []).append(evidence_id)
    locked_local_clusters = {
        cluster_id for cluster_id, member_ids in local_cluster_members.items()
        if len(member_ids) >= 2 and any(
            str(event_by_id[member_id].get("narrative_function", "")).lower()
            in {
                "inciting_incident", "turning_point", "false_victory",
                "reversal", "climax", "resolution",
            }
            or str(event_by_id[member_id].get("story_role", "")).lower()
            in {"climax", "resolution"}
            for member_id in member_ids
        )
    }

    def close_locked_clusters(seed_ids):
        closed = set(seed_ids)
        for cluster_id in locked_local_clusters:
            members = set(local_cluster_members[cluster_id])
            if closed & members:
                closed.update(members)
        return close_dependencies(closed)

    protected_ids = close_locked_clusters(protected_ids)

    def score(event):
        evidence_id = str(event.get("evidence_id", "")).strip().upper()
        cluster = cluster_by_event.get(evidence_id)
        value = function_score.get(
            str(event.get("narrative_function", "development")).lower(), 25
        )
        value += kind_score.get(str(event.get("event_kind", "action")).lower(), 0)
        value += 22 if str(event.get("story_role", "")).lower() == "climax" else 0
        value += 14 if str(event.get("importance", "main")).lower() == "main" else -8
        value += min(45, referenced_by.get(evidence_id, 0) * 15)
        value += 12 if str(event.get("cause", "")).strip() else 0
        value += 12 if str(event.get("consequence") or event.get("change") or "").strip() else 0
        value += 10 if str(event.get("state_before", "")).strip() and str(event.get("state_after", "")).strip() else 0
        if cluster:
            value += cluster_bonus[cluster[1]]
        if event.get("derived_visual_slot"):
            value -= 100
        if str(event.get("source", "")).lower() in {
            "transcript_bridge", "transcript_anchor",
        }:
            # Dialogue-only cards repair true coverage gaps but must not beat a
            # grounded visual event when both describe the same region.
            value -= 28
        return value

    # Duration grows only while selected facts still have grounded narration
    # capacity. A long recap therefore selects more real events before it
    # expands a dramatic event into its distinct cause/action/result details.
    required_capacity = required_tts_scene_count(target_words)
    selected_ids = set(protected_ids)

    # Repeated states compete as one candidate. Structural/cluster events are
    # never removed merely because Gemini assigned an over-broad repeat label.
    protected_repeat_groups = {
        str(event_by_id[evidence_id].get("repeat_group", "")).strip().casefold()
        for evidence_id in selected_ids if evidence_id in event_by_id
        and str(event_by_id[evidence_id].get("repeat_group", "")).strip()
    }
    best_repeat = {}
    candidates = []
    for event in ordered:
        evidence_id = str(event.get("evidence_id", "")).strip().upper()
        if evidence_id in selected_ids:
            continue
        repeat_group = str(event.get("repeat_group", "")).strip().casefold()
        if repeat_group:
            if repeat_group in protected_repeat_groups:
                continue
            current = best_repeat.get(repeat_group)
            if current is None or score(event) > score(current):
                best_repeat[repeat_group] = event
        else:
            candidates.append(event)
    candidates.extend(best_repeat.values())
    # Fill the budget iteratively.  Importance still matters, but an event far
    # from all selected anchors receives a coverage bonus.  This replaces the
    # old global score sort that clustered narration around action scenes.
    duration = max(
        float(boundaries[-1]) if boundaries else 0.0,
        max((float(event.get("end_seconds", 0.0)) for event in ordered), default=1.0),
    )
    target_spacing = max(20.0, duration / max(2, required_capacity - 1))

    def midpoint(event):
        return (
            float(event.get("start_seconds", 0.0))
            + float(event.get("end_seconds", 0.0))
        ) / 2.0

    while candidates:
        selected_capacity = evidence_narration_capacity(
            event_by_id[evidence_id] for evidence_id in selected_ids
            if evidence_id in event_by_id
        )
        if selected_capacity >= required_capacity:
            break
        selected_midpoints = [
            midpoint(event_by_id[evidence_id]) for evidence_id in selected_ids
            if evidence_id in event_by_id
        ]

        def balanced_value(event):
            point = midpoint(event)
            if selected_midpoints:
                nearest = min(abs(point - known) for known in selected_midpoints)
            else:
                nearest = duration
            coverage_bonus = min(300.0, (nearest / target_spacing) * 125.0)
            # Prefer the centre of a currently uncovered region over two
            # nearly adjacent candidates with otherwise equal value.
            edge_bonus = min(point, max(0.0, duration - point)) / max(duration, 1.0)
            return score(event) + coverage_bonus + edge_bonus

        event = max(candidates, key=lambda row: (
            balanced_value(row), -float(row.get("start_seconds", 0.0))
        ))
        candidates.remove(event)
        selected_ids.add(str(event.get("evidence_id", "")).strip().upper())
    selected_ids = close_locked_clusters(selected_ids)

    # A valid three-part plan must retain at least one grounded event in each
    # part. This is coverage, not uniform sampling: only the best event is used
    # when a part otherwise has no selected cluster.
    if boundaries and len(boundaries) == 4:
        for part_index in range(3):
            left, right = float(boundaries[part_index]), float(boundaries[part_index + 1])
            part_events = [
                event for event in ordered
                if left <= (
                    float(event.get("start_seconds", 0.0))
                    + float(event.get("end_seconds", 0.0))
                ) / 2.0 < right
            ]
            if part_events and not any(
                str(event.get("evidence_id", "")).strip().upper() in selected_ids
                for event in part_events
            ):
                selected_ids.add(str(max(part_events, key=score).get("evidence_id", "")).upper())

    selected = []
    for event in ordered:
        evidence_id = str(event.get("evidence_id", "")).strip().upper()
        if evidence_id not in selected_ids:
            continue
        cluster = cluster_by_event.get(evidence_id)
        if cluster:
            event["selection_cluster_id"] = cluster[0]
            event["selection_importance"] = cluster[1]
        event["causal_anchor"] = evidence_id in protected_ids
        editorial_role = anchor_role_by_event.get(evidence_id, "")
        if editorial_role:
            event["editorial_anchor_role"] = editorial_role
            if editorial_role == "climax":
                event["story_role"] = "climax"
                event["narrative_function"] = "climax"
            elif editorial_role in {"resolution", "ending"}:
                event["story_role"] = editorial_role
                event["narrative_function"] = editorial_role
            elif editorial_role == "inciting_incident":
                event["narrative_function"] = "inciting_incident"
            elif editorial_role == "opening":
                event["narrative_function"] = "setup"
        selected.append(event)
    return selected


def normalize_story_plan_from_evidence(
    text, chunk_evidence, video_duration_sec, target_words,
    require_editorial_contract=False, require_verified_visuals=False,
):
    """Combine a compact text plan with backend-locked chunk events."""
    payload = _json_object(text)
    duration = max(1.0, float(video_duration_sec))
    try:
        first = float(payload.get("boundary_1_seconds"))
        second = float(payload.get("boundary_2_seconds"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Ke hoach thieu hai ranh gioi theo giay.") from exc
    minimum_part = min(60.0, duration * 0.10)
    if not (minimum_part <= first < second <= duration - minimum_part):
        raise ValueError("Hai ranh gioi ke hoach khong hop le hoac tao phan qua ngan.")

    raw_parts = payload.get("parts")
    if not isinstance(raw_parts, list) or len(raw_parts) != 3:
        raise ValueError("Ke hoach text phai chua dung ba phan.")
    events = []
    for chunk in chunk_evidence or []:
        for event in chunk.get("events") or []:
            if isinstance(event, dict):
                events.append(copy.deepcopy(event))
    events.sort(key=lambda item: (
        float(item.get("start_seconds", 0.0)),
        float(item.get("end_seconds", 0.0)),
    ))
    for evidence_index, event in enumerate(events, start=1):
        if not str(event.get("evidence_id", "")).strip():
            event["evidence_id"] = f"LEGACY_EV{evidence_index:05d}"
    if not events:
        raise ValueError("EVENT_COVERAGE_INSUFFICIENT: Khong co bang chung cot truyen.")
    event_by_id = {
        str(event.get("evidence_id", "")).strip().upper(): event
        for event in events if str(event.get("evidence_id", "")).strip()
    }
    required_anchor_roles = (
        "opening", "inciting_incident", "protagonist_goal",
        "climax", "resolution", "ending",
    )
    raw_anchors = payload.get("editorial_anchors")
    if require_editorial_contract:
        try:
            contract_version = int(payload.get("editorial_contract_version", 0))
        except (TypeError, ValueError):
            contract_version = 0
        if contract_version not in {1, 2} or not isinstance(raw_anchors, dict):
            raise ValueError(
                "EDITORIAL_ANCHORS_MISSING: Ke hoach phai co editorial_contract_version=1 "
                "va du 6 nhom anchor."
            )
        anchors = {}
        for role in required_anchor_roles:
            values = raw_anchors.get(role) or []
            if not isinstance(values, list):
                values = [values]
            ids = list(dict.fromkeys(
                str(value).strip().upper() for value in values
                if str(value).strip()
            ))
            unknown = [value for value in ids if value not in event_by_id]
            if (not ids and contract_version == 1) or unknown:
                raise ValueError(
                    f"EDITORIAL_ANCHORS_INVALID: {role} rong hoac co ID khong ton tai: "
                    + ",".join(unknown)
                )
            anchors[role] = ids

        def first_start(role):
            return min(float(event_by_id[value].get("start_seconds", 0.0)) for value in anchors[role])

        def last_end(role):
            return max(float(event_by_id[value].get("end_seconds", 0.0)) for value in anchors[role])

        if contract_version == 2:
            from story_planning import validate_event_roles
            validate_event_roles(dict(payload, editorial_anchors=anchors), events)
        if contract_version == 1 and first_start("opening") >= first_start("climax"):
            raise ValueError("EDITORIAL_ANCHORS_ORDER: opening phai nam truoc climax.")
        terminal_anchor_status = str(
            payload.get("terminal_anchor_status", "")
        ).strip().lower()
        terminal_anchor_override = terminal_anchor_status in {
            "video_review_confirmed", "last_verified_complete",
            "last_verified_only",
        }
        if (
            contract_version == 1 and not terminal_anchor_override
            and first_start("resolution") < last_end("climax") - 2.0
        ):
            raise ValueError("EDITORIAL_ANCHORS_ORDER: resolution phai theo sau climax.")
        if contract_version == 1 and first_start("ending") < first_start("resolution"):
            raise ValueError("EDITORIAL_ANCHORS_ORDER: ending phai theo sau resolution.")
        payload["editorial_anchors"] = anchors
    boundaries = [0.0, first, second, duration]
    # Select a duration-aware causal spine. The source ledger remains intact;
    # minor facts may be omitted while major clusters can retain several
    # separate events/sentences.
    selected = select_causal_story_evidence(
        events, payload, target_words, boundaries=boundaries
    )
    if not selected:
        raise ValueError("EVENT_COVERAGE_INSUFFICIENT: Khong chon duoc cum su kien.")
    # The selector marks the causal/editorial spine, but the sentence planner
    # owns the complete deduplicated ledger. Add every remaining real event so
    # minor transitions can be grouped rather than silently dropped.
    selected_by_id = {
        str(event.get("evidence_id", "")).strip().upper(): event
        for event in selected if str(event.get("evidence_id", "")).strip()
    }
    for event in events:
        evidence_id = str(event.get("evidence_id", "")).strip().upper()
        if evidence_id and evidence_id not in selected_by_id:
            selected.append(copy.deepcopy(event))
            selected_by_id[evidence_id] = selected[-1]
    selected.sort(key=lambda event: (
        float(event.get("start_seconds", 0.0)),
        float(event.get("end_seconds", 0.0)),
    ))
    selected_distinct_count = len(selected)
    selected_points = [0.0] + [
        (
            float(event.get("start_seconds", 0.0))
            + float(event.get("end_seconds", 0.0))
        ) / 2.0
        for event in selected
    ] + [duration]
    selected_points = sorted(set(selected_points))
    selected_gaps = [
        later - earlier
        for earlier, later in zip(selected_points, selected_points[1:])
    ]
    maximum_selected_gap = max(selected_gaps, default=duration)
    transcript_bridge_selected = sum(
        str(event.get("source", "")).lower() in {
            "transcript_bridge", "transcript_anchor",
        }
        for event in selected
    )
    # Sentence count is a hard user contract. Never shrink it to whatever the
    # current evidence pass happened to return; missing capacity must be
    # repaired before prose writing instead of producing a much shorter recap.
    required_scene_count = required_tts_scene_count(target_words)
    if payload.get("planning_version") in {1, 2}:
        from story_planning import validate_spine, materialize_sentence_plan
        validate_spine(payload, events, required_scene_count)
        selected = materialize_sentence_plan(payload, selected, require_verified_visuals,
                                             defer_visual_selection=payload.get('visual_selection_mode') == 'sentence_first')
        if len(selected) != required_scene_count:
            raise ValueError("DETAIL_SENTENCE_TOTAL: merged plan does not match global target")
    else:
        selected = expand_evidence_to_visual_slots(
            selected, required_scene_count,
            require_verified_visuals=require_verified_visuals,
        )
    if payload.get("editorial_contract_version") == 2:
        # Roles are editorial decisions made after grouping, not inherited
        # guesses from the source ledger or automatically promoted final rows.
        anchors = payload["editorial_anchors"]
        for event in selected:
            ids = set(event.get("planning_source_event_ids") or
                      [event.get("source_evidence_id") or event.get("evidence_id")])
            role = next((role for role in ("ending", "resolution", "climax")
                         if ids.intersection(anchors.get(role, []))), "development")
            event["story_role"] = role
            event["narrative_function"] = role
            event.pop("ending_inferred_by_backend", None)
    visual_slot_count = sum(
        bool(event.get("derived_visual_slot")) for event in selected
    )
    split_source_event_count = len({
        str(event.get("source_evidence_id", "")).strip().upper()
        for event in selected
        if int(event.get("narration_slot_count", 1) or 1) > 1
    })
    grouped_sentence_count = sum(
        len(event.get("covered_events") or []) > 1 for event in selected
    )

    # Keep exactly one short epilogue anchor. Earlier rows that were broadly
    # labelled as endings still carry useful consequences, but belong to the
    # resolution so the writer cannot spend several sentences re-ending the film.
    ending_indexes = [
        index for index, event in enumerate(selected)
        if str(event.get("story_role", "")).lower() == "ending"
    ]
    for ending_index in ending_indexes[:-1]:
        selected[ending_index]["story_role"] = "resolution"
        if str(selected[ending_index].get("narrative_function", "")).lower() == "ending":
            selected[ending_index]["narrative_function"] = "resolution"

    events_by_part = [[], [], []]
    for index, event in enumerate(selected, start=1):
        midpoint = (
            float(event.get("start_seconds", 0.0))
            + float(event.get("end_seconds", 0.0))
        ) / 2.0
        owner = 0 if midpoint < first else (1 if midpoint < second else 2)
        event["event_id"] = f"E{index:04d}"
        role = str(event.get("story_role", "development")).lower()
        event["story_role"] = (
            role if role in {"development", "climax", "resolution", "ending"}
            else "development"
        )
        event["importance"] = (
            "bridge" if str(event.get("importance", "")).lower() == "bridge"
            else "main"
        )
        events_by_part[owner].append(event)
    if any(not part for part in events_by_part):
        raise ValueError("Hai ranh gioi khong phan bo duoc bang chung cho ca ba phan.")
    inferred_terminal_ending = False
    if payload.get("editorial_contract_version") != 2 and not any(event.get("story_role") == "ending" for event in events_by_part[2]):
        # A missing label is not proof that the video has no ending. Prefer a
        # real resolution near the end, then a completed visual event, and only
        # as a last resort the final grounded event in part 3. This changes the
        # role label only; it never invents a new action or timestamp.
        terminal_candidates = list(reversed(events_by_part[2]))
        inferred_ending = next((
            event for event in terminal_candidates
            if str(event.get("story_role", "")).lower() == "resolution"
            or str(event.get("narrative_function", "")).lower()
            in {"resolution", "ending"}
        ), None)
        if inferred_ending is None:
            inferred_ending = next((
                event for event in terminal_candidates
                if str(event.get("completion_status", "")).lower() == "complete"
                and str(event.get("source", "")).lower()
                not in {"transcript_bridge", "transcript_anchor"}
            ), None)
        if inferred_ending is None:
            inferred_ending = next((
                event for event in terminal_candidates
                if str(event.get("source", "")).lower()
                not in {"transcript_bridge", "transcript_anchor"}
            ), None)
        if inferred_ending is None and terminal_candidates:
            inferred_ending = terminal_candidates[0]
        if inferred_ending is not None:
            inferred_ending["story_role"] = "ending"
            inferred_ending["narrative_function"] = "ending"
            inferred_ending["ending_inferred_by_backend"] = True
            inferred_terminal_ending = True

    events_by_part, _ = assign_story_beats(events_by_part)
    # Do not collapse the sentence pool below the duration contract. Several
    # grounded events in one large story cluster may legitimately receive
    # separate recap sentences.
    dense_events_collapsed = 0
    budget_events_collapsed = 0
    next_event_number = 1
    for part in events_by_part:
        for event in part:
            event["event_id"] = f"E{next_event_number:04d}"
            next_event_number += 1
    events_by_part, story_beats_by_part = assign_story_beats(events_by_part)
    events_by_part = attach_fact_history(events_by_part)

    flat_contract_events = [event for part in events_by_part for event in part]
    for part_index, part in enumerate(events_by_part):
        for event in part:
            event["part_boundary_end_seconds"] = float(boundaries[part_index + 1])
    minimum_supported_words = len(flat_contract_events) * MIN_TTS_WORDS
    maximum_supported_words = sum(
        20 if str(event.get("story_role", "")).lower() == "ending"
        else MAX_TTS_WORDS
        for event in flat_contract_events
    )
    # The target sentence count has already been locked above. Keep the user's
    # requested word budget as soft timing metadata; do not lower it because a
    # model returned too few events.
    effective_target_words = int(target_words)
    scene_contract = lock_story_scene_contract(
        events_by_part, effective_target_words
    )
    scene_contract["ending_inferred"] = inferred_terminal_ending
    effective_target_words = int(scene_contract["target_words"])
    scene_contract["requested_target_words"] = int(target_words)
    scene_contract["requested_scene_count"] = required_tts_scene_count(target_words)
    scene_contract["sentence_tolerance_met"] = (
        abs(len(flat_contract_events) - scene_contract["requested_scene_count"]) <= 3
    )
    scene_contract["capacity_reduced_words"] = max(
        0, int(target_words) - effective_target_words
    )
    scene_contract["dense_events_collapsed"] = dense_events_collapsed
    scene_contract["budget_events_collapsed"] = budget_events_collapsed
    covered_source_ids = list(dict.fromkeys(
        str(evidence_id).strip().upper()
        for event in flat_contract_events
        for evidence_id in (
            event.get("merged_evidence_ids")
            or [event.get("source_evidence_id") or event.get("evidence_id")]
        )
        if str(evidence_id or "").strip()
    ))
    source_event_ids = {
        str(event.get("evidence_id", "")).strip().upper() for event in events
        if str(event.get("evidence_id", "")).strip()
    }
    scene_contract["source_event_count"] = len(source_event_ids)
    scene_contract["covered_source_event_count"] = len(
        source_event_ids & set(covered_source_ids)
    )
    scene_contract["uncovered_source_event_ids"] = sorted(
        source_event_ids - set(covered_source_ids)
    )
    scene_contract["omitted_minor_event_count"] = len(
        scene_contract["uncovered_source_event_ids"]
    )

    weights = payload.get("part_weights")
    if not isinstance(weights, list) or len(weights) != 3:
        weights = [len(part) for part in events_by_part]
    # Character identity is owned by verified video evidence, not by the
    # text-only planning response. The planner may enrich a matching role or
    # description, but it cannot introduce or rename a person.
    planner_characters_by_name = {
        str(item.get("name", "")).strip().casefold(): item
        for item in (payload.get("characters") or []) if isinstance(item, dict)
        and str(item.get("name", "")).strip()
    }
    characters_by_id = {}
    for chunk in chunk_evidence or []:
        for source_character in chunk.get("characters") or []:
            if not isinstance(source_character, dict):
                continue
            character_id = str(
                source_character.get("canonical_id")
                or source_character.get("id") or ""
            ).strip().upper()
            name = re.sub(r"\s+", " ", str(
                source_character.get("name")
                or source_character.get("descriptive_alias")
                or source_character.get("neutral_alias")
                or source_character.get("alias")
                or ""
            )).strip()
            if not character_id or not name:
                continue
            row = characters_by_id.setdefault(character_id, {
                "id": character_id,
                "name": name,
                "description": str(
                    source_character.get("description")
                    or source_character.get("visual_signature")
                    or source_character.get("signature") or ""
                ).strip(),
                "role": str(source_character.get("role", "")).strip(),
            })
            planner_row = planner_characters_by_name.get(name.casefold(), {})
            if planner_row:
                row["description"] = str(
                    planner_row.get("description") or row["description"]
                ).strip()
                row["role"] = str(
                    planner_row.get("role") or row["role"]
                ).strip()
    characters = list(characters_by_id.values())
    story_contract = payload.get("story_contract")
    return {
        "boundaries": boundaries,
        "planning_version": int(payload.get("planning_version", 0) or 0),
        "visual_selection_mode": payload.get('visual_selection_mode', ''),
        "sentence_budget_mode": payload.get('sentence_budget_mode', ''),
        "target_sentence_count": payload.get('target_sentence_count', required_scene_count),
        "planning_clusters": copy.deepcopy(payload.get("selected_event_clusters") or []),
        "weights": weights,
        "summaries": [str(part.get("summary", "")).strip() for part in raw_parts],
        "end_states": [str(part.get("end_state", "")).strip() for part in raw_parts],
        "handoffs": [str(part.get("handoff_to_next", "")).strip() for part in raw_parts],
        "characters": characters,
        "story_contract": story_contract if isinstance(story_contract, dict) else {},
        "editorial_contract_version": int(
            payload.get("editorial_contract_version", 0) or 0
        ),
        "editorial_anchors": copy.deepcopy(payload.get("editorial_anchors") or {}),
        "events_by_part": events_by_part,
        "story_beats_by_part": story_beats_by_part,
        "scene_contract": scene_contract,
        "selection_stats": {
            "candidate_event_count": len(events),
            "selected_distinct_event_count": selected_distinct_count,
            "sentence_slot_count": len(selected),
            "split_source_event_count": split_source_event_count,
            "grouped_sentence_count": grouped_sentence_count,
            "visual_slot_count": visual_slot_count,
            "cluster_count": len(payload.get("selected_event_clusters") or []),
            "dense_events_collapsed": dense_events_collapsed,
            "effective_target_words": effective_target_words,
            "maximum_selected_gap_seconds": round(maximum_selected_gap, 2),
            "selected_transcript_bridge_count": transcript_bridge_selected,
        },
        "timestamp_modes": ["global", "global", "global"],
    }


def terminal_anchor_review_candidates(plan_text, chunk_evidence, neighbor_count=1):
    """Return only resolution/ending IDs and their immediate event neighbors."""
    payload = _json_object(plan_text)
    anchors = payload.get("editorial_anchors") or {}
    requested = []
    for role in ("resolution", "ending"):
        values = anchors.get(role) or []
        if not isinstance(values, list):
            values = [values]
        requested.extend(
            str(value).strip().upper() for value in values if str(value).strip()
        )
    events = sorted(
        [
            event for chunk in (chunk_evidence or [])
            for event in (chunk.get("events") or []) if isinstance(event, dict)
            and str(event.get("evidence_id", "")).strip()
        ],
        key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ),
    )
    positions = {
        str(event.get("evidence_id", "")).strip().upper(): index
        for index, event in enumerate(events)
    }
    selected_positions = set()
    radius = max(0, int(neighbor_count))
    for evidence_id in requested:
        position = positions.get(evidence_id)
        if position is None:
            continue
        selected_positions.update(range(
            max(0, position - radius),
            min(len(events), position + radius + 1),
        ))
    return [copy.deepcopy(events[index]) for index in sorted(selected_positions)]


def build_terminal_anchor_review_prompt(candidate_events):
    cards = []
    for event in candidate_events or []:
        if not isinstance(event, dict):
            continue
        cards.append({
            "evidence_id": str(event.get("evidence_id", "")).strip().upper(),
            "start_seconds": event.get("start_seconds"),
            "end_seconds": event.get("end_seconds"),
            "event": str(event.get("event", "")).strip(),
            "consequence": str(event.get("consequence", "")).strip(),
            "state_after": str(event.get("state_after", "")).strip(),
            "completion_status": str(event.get("completion_status", "")).strip(),
        })
    return f"""Bạn chỉ xác minh các ID cuối phim trong những clip ngắn đính kèm.
Mỗi clip được gửi ngay sau nhãn ID tương ứng và có thêm khoảng 20 giây trước/sau.

Hãy xác định:
- `resolution_ids`: ID cho thấy xung đột/vấn đề chính được giải quyết.
- `ending_ids`: ID cho thấy trạng thái cuối thực tế sau đó.
- Một ID có thể nằm trong cả hai danh sách nếu cùng cảnh vừa giải quyết vừa kết thúc.
- Chỉ dùng ID trong danh sách; không nhận ra thì trả `recognized=false`, không đoán.

DANH SÁCH ID:
{json.dumps(cards, ensure_ascii=False)}

Chỉ trả JSON:
{{"recognized":true,"resolution_ids":["EV..."],"ending_ids":["EV..."],"reason":""}}
"""


def apply_terminal_anchor_review(plan_text, review_text, chunk_evidence):
    """Patch only terminal anchor IDs when the targeted review is decisive."""
    plan = _json_object(plan_text)
    review = _json_object(review_text)
    if review.get("recognized") is not True:
        raise ValueError("TERMINAL_ANCHOR_REVIEW_UNRECOGNIZED")
    known_ids = {
        str(event.get("evidence_id", "")).strip().upper()
        for chunk in (chunk_evidence or [])
        for event in (chunk.get("events") or []) if isinstance(event, dict)
        and str(event.get("evidence_id", "")).strip()
    }

    def valid_ids(key):
        values = review.get(key) or []
        if not isinstance(values, list):
            values = [values]
        result = list(dict.fromkeys(
            str(value).strip().upper() for value in values
            if str(value).strip().upper() in known_ids
        ))
        if not result:
            raise ValueError("TERMINAL_ANCHOR_REVIEW_UNRECOGNIZED")
        return result

    anchors = plan.setdefault("editorial_anchors", {})
    anchors["resolution"] = valid_ids("resolution_ids")
    anchors["ending"] = valid_ids("ending_ids")
    plan["terminal_anchor_status"] = "video_review_confirmed"
    return json.dumps(plan, ensure_ascii=False)


def fallback_terminal_editorial_anchors(plan_text, chunk_evidence):
    """Deterministically keep the last verified fact when review is inconclusive.

    This does not claim an unseen happy ending. It only prevents an optional
    editorial label from killing an otherwise grounded recap.
    """
    plan = _json_object(plan_text)
    events = sorted(
        [
            event for chunk in (chunk_evidence or [])
            for event in (chunk.get("events") or []) if isinstance(event, dict)
            and str(event.get("evidence_id", "")).strip()
            and str(event.get("source", "")).lower()
            not in {"transcript_bridge", "transcript_anchor"}
        ],
        key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ),
    )
    if not events:
        raise ValueError("TERMINAL_ANCHOR_FALLBACK_EMPTY")
    terminal = events[-1]
    terminal_is_complete = (
        str(terminal.get("completion_status", "")).lower() == "complete"
        and any(str(terminal.get(key, "")).strip() for key in (
            "consequence", "result", "state_after", "change",
        ))
    )
    terminal_id = str(terminal.get("evidence_id", "")).strip().upper()
    anchors = plan.setdefault("editorial_anchors", {})
    # The same ID is intentionally legal: one final verified scene can both
    # resolve the visible action and be the last state shown by the source.
    anchors["resolution"] = [terminal_id]
    anchors["ending"] = [terminal_id]
    plan["terminal_anchor_status"] = (
        "last_verified_complete" if terminal_is_complete
        else "last_verified_only"
    )
    plan["terminal_anchor_fallback_id"] = terminal_id
    return json.dumps(plan, ensure_ascii=False)


def build_part_story_context(story_plan, part_number, previous_response=None):
    """Create compact text-only evidence with timestamps local to one part."""
    index = int(part_number) - 1
    boundaries = story_plan["boundaries"]
    offset = float(boundaries[index])
    character_names_by_id = {}
    character_identity_rows = []
    registered_names = set()
    for character in story_plan.get("characters") or []:
        if not isinstance(character, dict):
            continue
        character_id = str(character.get("id", "")).strip().upper()
        aliases = character.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = [aliases]
        names = [
            re.sub(r"\s+", " ", str(value or "")).strip()
            for value in [character.get("name"), *aliases,
                          *([character.get('display_name') or character.get('neutral_alias')]
                            if not character.get('name') else [])]
            if str(value or "").strip()
        ]
        if character_id and names:
            character_names_by_id[character_id] = names
            character_identity_rows.append((character_id, names))
        registered_names.update(names)
    beat_character_names = {}
    for event in story_plan["events_by_part"][index]:
        beat_id = str(event.get("story_beat_id", "")).strip().upper()
        names = beat_character_names.setdefault(beat_id, set())
        for value in event.get("characters") or []:
            raw = str(value).strip()
            names.update(character_names_by_id.get(raw.upper(), []))
            if raw in registered_names:
                names.add(raw)
    all_ordered_events = sorted(
        (
            event for part in (story_plan.get("events_by_part") or [])
            for event in (part or []) if isinstance(event, dict)
        ),
        key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ),
    )
    global_position = {
        str(event.get("event_id", "")).strip().upper(): position
        for position, event in enumerate(all_ordered_events)
        if str(event.get("event_id", "")).strip()
    }

    def explicit_event_characters(event):
        """Recover canonical participants named in the event's own facts.

        Gemini sometimes lists only the actor ID even though the event text
        also names the target/people being rescued. Those names are factual
        event participants and must be legal in TTS.
        """
        text_fields = (
            "event", "visual_detail", "cause", "change", "consequence",
            "subject", "action", "object", "result", "state_before",
            "state_after", "narration_focus", "required_for",
        )
        values = [str(event.get(key, "") or "") for key in text_fields]
        values.append(str(event.get('spoken_text') or ''))
        values.extend(str(f.get('text') or '') for f in event.get('observed_facts') or [])
        for collection_key in ("covered_events", "supporting_facts"):
            for row in event.get(collection_key) or []:
                if isinstance(row, dict):
                    values.extend(str(row.get(key, "") or "") for key in text_fields)
                elif row:
                    values.append(str(row))
        haystack = re.sub(r"\s+", " ", " ".join(values)).strip()
        from character_identity import dialogue_name_mentions
        dialogue_mentions = dialogue_name_mentions(event.get('spoken_text') or '',
            [name for _, identity_names in character_identity_rows for name in identity_names])
        ids = []
        names = set()
        for character_id, identity_names in character_identity_rows:
            if any(character_name_in_text(name, haystack) or name in dialogue_mentions for name in identity_names):
                ids.append(character_id)
                names.update(identity_names)
        return ids, names

    def neighbor_card(event):
        if not isinstance(event, dict):
            return None
        return {
            "event_id": str(event.get("event_id", "")).strip().upper(),
            "event": re.sub(r"\s+", " ", str(event.get("event", ""))).strip(),
            "cause": re.sub(r"\s+", " ", str(event.get("cause", ""))).strip(),
            "outcome": re.sub(r"\s+", " ", str(
                event.get("consequence") or event.get("change")
                or event.get("state_after") or ""
            )).strip(),
            "temporal_relation": str(
                event.get("temporal_relation", "continuation")
            ).strip().lower(),
        }

    local_events = []
    for event in story_plan["events_by_part"][index]:
        local_event = copy.deepcopy(event)
        local_event["start"] = _format_time(float(event["start_seconds"]) - offset)
        local_event["end"] = _format_time(float(event["end_seconds"]) - offset)
        if event.get('visual_selection_pending'):
            # These are part-local narrative windows, NOT selected pictures.
            # Cross-part events must not create negative writer timestamps.
            local_event['start'] = _format_time(max(0.0, float(event['start_seconds']) - offset))
            local_event['end'] = _format_time(min(float(boundaries[index + 1]) - offset,
                                                float(event['end_seconds']) - offset))
        local_event.pop("start_seconds", None)
        local_event.pop("end_seconds", None)
        for source in local_event.get("visual_sources") or []:
            if not isinstance(source, dict):
                continue
            source["start"] = _format_time(
                float(source.get("start_seconds", offset)) - offset
            )
            source["end"] = _format_time(
                float(source.get("end_seconds", offset)) - offset
            )
            source.pop("start_seconds", None)
            source.pop("end_seconds", None)
        beat_id = str(local_event.get("story_beat_id", "")).strip().upper()
        explicit_ids, explicit_names = explicit_event_characters(event)
        local_event["characters"] = list(dict.fromkeys(
            [str(value).strip().upper() for value in event.get("characters") or []
             if str(value).strip()]
            + explicit_ids
        ))
        local_event["allowed_character_names"] = sorted(
            (set(name for cid in local_event['characters']
                 for name in character_names_by_id.get(cid, []))
             if event.get('situation_narration') else set(beat_character_names.get(beat_id, set())))
            | explicit_names
        )
        local_event['visible_character_names'] = sorted({name for cid in event.get('characters') or []
            for name in character_names_by_id.get(cid, [])})
        local_event['mentioned_character_names'] = sorted(explicit_names)
        position = global_position.get(
            str(event.get("event_id", "")).strip().upper()
        )
        if position is not None:
            local_event["previous_event_context"] = neighbor_card(
                all_ordered_events[position - 1] if position > 0 else None
            )
            local_event["next_event_context"] = neighbor_card(
                all_ordered_events[position + 1]
                if position + 1 < len(all_ordered_events) else None
            )
        local_events.append(local_event)

    previous_tail = []
    if previous_response:
        try:
            previous_part = _json_object(previous_response).get("parts", [{}])[0]
            for scene in list(previous_part.get("scenes") or [])[-8:]:
                if isinstance(scene, dict):
                    previous_tail.append({
                        "event": str(scene.get("event", "")),
                        "tts": str(scene.get("tts", "")),
                    })
        except (ValueError, IndexError, TypeError):
            previous_tail = []
    return {
        "characters": story_plan.get("characters", []),
        "all_part_summaries": story_plan.get("summaries", []),
        "current_summary": story_plan.get("summaries", ["", "", ""])[index],
        "required_events": local_events,
        "story_beats": copy.deepcopy(
            (story_plan.get("story_beats_by_part") or [[], [], []])[index]
        ),
        "required_ending_sec": max(
            (float(event["end_seconds"]) - offset for event in story_plan["events_by_part"][index]),
            default=float(boundaries[index + 1]) - offset,
        ),
        "previous_part_tail": previous_tail,
    }


def build_single_part_video_prompt(
    part_number, clip_duration_sec, global_start_sec, global_end_sec,
    target_words, style="dramatic", custom_prompt="", enable_multi_voice=False,
    story_context=None,
):
    """Prompt Gemini to write one part from the grounded full-video event map."""
    part_number = int(part_number)
    target_words = max(1, int(target_words))
    requested_words, _, _ = requested_word_range(target_words)
    minimum_words = requested_word_range(target_words)[1]
    extras = str(custom_prompt or "").strip()
    story_context = story_context or {}
    required_events = list(story_context.get("required_events") or [])
    required_event_ids = [str(item.get("event_id")) for item in required_events]
    # Pass 1 has already locked one visual/TTS scene per required event.
    # Never derive a second, conflicting quota from this part's word budget.
    required_scene_count = (
        len(required_events) if required_events
        else required_tts_scene_count(target_words)
    )
    if required_events and all(e.get('grounded_plan_locked') for e in required_events):
        cards = [{'event_id': e['event_id'], 'focus': e['narration_focus'],
                  'observed_facts': e['observed_facts'],
                  'allowed_character_names': e.get('allowed_character_names') or [],
                  'scene_context': e.get('scene_context') or {},
                  'spoken_text': e.get('spoken_text') or '',
                  'transcript_evidence': e.get('transcript_evidence') or [],
                  'sentence_quota': e.get('sentence_quota'),
                  'tts_target_words': e.get('tts_target_words'),
                  'visuals': visual_catalog(e.get('visual_sources') or [])} for e in required_events]
        if all(e.get('situation_narration') for e in required_events):
            return (
                f'Viet loi recap cuoi cung cho {required_scene_count} TINH HUONG phan {part_number}/3. '
                'Moi event_id la CA TINH HUONG gom nhieu canh, KHONG phai mot cau hay mot shot. '
                'Neu co sentence_quota thi moi tts phai DUNG so cau do, gan tts_target_words; '
                'so cau trong doan linh hoat theo quota, KHONG gioi han 1-4 cau. '
                'Neu khong co quota thi viet doan ngan tu nhien. Doc ca chuong de noi mach. '
                'Giu muc tieu, bien co, hanh dong chinh, ket qua da duoc xac nhan. '
                'Bo dong tac vuon vat, khong doc lai toan bo observed_facts. '
                'Khong ke tung dong tac boi, bam da, bat ca rieng neu chi can noi "san duoc ca". '
                'Chi neu dong co, noi dung hoi thoai, cam xuc va quan he nhan qua khi du kien xac nhan. '
                'Chuyen tuyen/thoi gian phai co cau dan ngan; khong noi hai tuyen bang "dung luc". '
                'DOAN TRUOC la nhung y da ke: khong tuyen bo lai cung quyet dinh, '
                'muc tieu hay viec chuan bi hanh ly o moi doan; chi ke buoc tien moi. '
                'Neu tro lai tuyen cu sau canh chen ngang, dung cau dan nhu "Tro lai voi..." '
                'khi can, khong bien cac canh xen ke thanh hanh dong dong thoi. '
                'Neu hoi thoai khong ro noi dung, bo chi tiet rong thay vi viet "tro chuyen nghiem tuc". '
                'Uu tien ten chinh trong NHAN VAT khi allowed_character_names cho phep; '
                'khong goi Gian la "cau be to lon" hay Suneo la "cau be mo nhon" '
                'neu ID va ten da xac nhan. Khong them tinh tu mau sac vao ten Doraemon. '
                'Khong viet cau rong nhu "tiep tuc tro chuyen"; khong lap ten dau moi cau. '
                'Hinh da chon truoc nam trong visuals. CHON CAC HINH DAI DIEN can thiet trong nhom, '
                'khong bat buoc dung het hay viet loi cho moi hinh. Mot cau co the chay qua nhieu hinh. '
                'Chon du footage cho loi ke, uu tien hanh dong va phan ung cung tinh huong; '
                'khong lay canh khac tuyen de lap cho trong. Loi thoai chi nghe duoc co the '
                'dung toan canh/phan ung cua DUNG tinh huong, khong gia lam hinh chung minh hanh dong. '
                'Khong doi tts de phuc vu hinh; khong tu doi timestamp hay fact_ids. '
                'visual_links chi dung ID trong nhom, theo thu tu nguon. Link dau trigger_text=""; '
                'moi link sau gan cum 2-6 tu duy nhat trong loi ke, khong tao them cau de co trigger. '
                'Khong ke logo, intro, montage gioi thieu hay credits. '
                'Chi dung ten duoc phep; khong suy ra ten tu tri nho phim. '
                'Tra thang loi ke cuoi, khong tao ban nhap de chia lai thanh mo ta shot. '
                + '\nPHONG CACH: ' + str(style) + '\nYEU CAU: ' + str(custom_prompt or '')
                + ('\nMoi scene them voice="Nam" hoac voice="Nu".' if enable_multi_voice else '')
                + '\nNHAN VAT: ' + json.dumps(story_context.get('characters') or [], ensure_ascii=False)
                + '\nMACH TRUYEN: ' + json.dumps(story_context.get('all_part_summaries') or [], ensure_ascii=False)
                + '\nDOAN TRUOC: ' + json.dumps(story_context.get('previous_part_tail') or [], ensure_ascii=False)
                + '\nTIEP THEO (chi de noi mach): ' + json.dumps(story_context.get('next_batch_context') or [], ensure_ascii=False)
                + '\nTINH HUONG: ' + json.dumps(cards, ensure_ascii=False)
                + '\nChi tra JSON {"scenes":[{"event_id":"E0001","tts":"Doan ke...",'
                '"visual_links":[{"visual_id":"V001","trigger_text":""}]}]}.'
            )
        return (
            f'Dien dat DUNG {required_scene_count} o cua phan {part_number}/3 thanh recap tieng Viet. '
            'Moi event_id dung mot muc tts; cac muc ghep lai thanh loi ke lien tuc. '
            'Doc ca cum tinh huong truoc khi phan loi ve cac o. Mot cau co the noi qua hai o ke nhau; '
            'diem cat hinh khong bat buoc la dau cham. Moi muc van phai ro nghia va chi ke fact cua o do. '
            'Chi duoc dien dat lai focus va observed_facts da khoa. '
            'Khong them hanh dong, do vat, bao boi, noi tam, nguyen nhan hay ket qua ngoai bang chung. '
            'Khong them pham chat, thai do hay cam giac nhu dung cam, quyet doan, mat lanh '
            'neu observed_facts khong neu. Khong them chi tiet de keo dai cau. '
            'Khong them logo, credits, loi gioi thieu, ket thuc truoc khi du kien cho thay ket thuc. '
            'Khong tao lai y ke hoac tim hinh; visuals la hinh CO DINH cua o nay. '
            'Cau tu 6-25 tu; co the ngan neu du kien ngan, khong chen tu dem de du muc tieu tu. '
            'Neu visuals co nhieu muc, dung du cac visual_id theo thu tu; link dau trigger_text="", '
            'cac link sau dung cum 2-6 tu xuat hien dung mot lan o dau y can chuyen hinh. '
            'Khong chon hinh ngoai danh sach, khong tu viet timestamp. '
            'Bien tap ca doan sau khi phan o: bo lap chu ngu, lap y va cau chuyen may moc. '
            'Khong ep moi cau co sau do/tiep theo; chi dung quan he nhan qua khi bang chung xac nhan. '
            'Khong viet "tiep tuc tro chuyen" thay cho noi dung su kien. '
            'Noi mach giua cac cau theo tom tat va phan truoc; ngu canh khong duoc tao them du kien. '
            + '\nPHONG CACH (chi doi cach dien dat): ' + str(style)
            + '\nYEU CAU BIEN TAP (khong duoc them fact hay doi quota): ' + str(custom_prompt or '')
            + ('\nMoi scene them voice="Nam" hoac voice="Nu".' if enable_multi_voice else '')
            + '\nMACH TRUYEN: ' + json.dumps(story_context.get('all_part_summaries') or [], ensure_ascii=False)
            + '\nNHAN VAT: ' + json.dumps(story_context.get('characters') or [], ensure_ascii=False)
            + '\nPHAN TRUOC: ' + json.dumps(story_context.get('previous_part_tail') or [], ensure_ascii=False)
            + '\nCAC O DA KHOA: ' + json.dumps(cards, ensure_ascii=False)
            + '\nChi tra JSON {"scenes":[{"event_id":"E0001","tts":"Loi ke",'
            '"visual_links":[{"visual_id":"V001","trigger_text":""}]}]}.'
        )
    if required_events and all(e.get('visual_selection_pending') for e in required_events):
        cards = [{'id': e['event_id'], 'fact': e.get('narration_focus') or e.get('event'),
                  'cause': e.get('cause'), 'result': e.get('consequence') or e.get('change'),
                  'who': e.get('allowed_character_names') or [],
                  'previous': e.get('previous_event_context'), 'next': e.get('next_event_context'),
                  'words': e.get('tts_target_words', 22)} for e in required_events]
        return (
            f'Viet recap tieng Viet tu bang chung cho phan {part_number}/3. '
            f'Dung {required_scene_count} cau, muc tieu {target_words} tu. '
            'Chi viet loi ke; chua chon shot, chua dat timestamp. Moi ID dung mot lan theo thu tu. '
            'Chi ke fact da quan sat, khong bia hanh dong de tao hook. Cau sau phai them thong tin moi; '
            'khong dien dat lai cung quyet dinh/cam xuc de du so tu. Noi mach nguyen nhan-hanh dong-ket qua. '
            'Khong ep co ket thuc neu bang chung chi cho thay xung dot dang do. '
            f'Moi cau {MIN_TTS_WORDS}-{MAX_TTS_WORDS} tu; tranh tu dem. '
            + style_storytelling_prompt(style) + '\n' + extras
            + '\nNHAN VAT: ' + json.dumps(story_context.get('characters') or [], ensure_ascii=False)
            + '\nDUOI PHAN TRUOC: ' + json.dumps(story_context.get('previous_part_tail') or [], ensure_ascii=False)
            + '\nDU KIEN: ' + json.dumps(cards, ensure_ascii=False)
            + '\nChi tra JSON {"scenes":[{"event_id":"E0001","tts":"Loi ke tieng Viet"}]}.'
        )
    # Gemini only writes prose. IDs, timestamps, scene metadata and character
    # identities remain backend-owned and are restored after parsing.
    compact_events = []
    for item in required_events:
        if not isinstance(item, dict):
            continue
        compact_events.append({
            "id": str(item.get("event_id", "")).strip().upper(),
            "source_event": str(
                item.get("source_evidence_id") or item.get("evidence_id") or ""
            ).strip().upper(),
            "slot": [
                int(item.get("narration_slot_index", 1) or 1),
                int(item.get("narration_slot_count", 1) or 1),
            ],
            "time": f"{item.get('start', '')}-{item.get('end', '')}",
            "fact": str(
                item.get("narration_focus") or item.get("event") or ""
            ).strip(),
            "cause": str(item.get("cause") or "").strip(),
            "result": str(
                item.get("consequence") or item.get("change") or ""
            ).strip(),
            "who": list(item.get("allowed_character_names") or []),
            "spoken": str(
                item.get("spoken_text") or item.get("transcript_evidence") or ""
            ).strip(),
            "words": int(item.get("tts_target_words") or 22),
            "visuals": visual_catalog(item.get("visual_sources") or []),
        })
    compact_context = {
        "characters": story_context.get("characters") or [],
        "events": compact_events,
        "previous_tail": story_context.get("previous_part_tail") or [],
    }
    compact_json = json.dumps(
        compact_context, ensure_ascii=False, separators=(",", ":")
    )
    ending_rule = (
        "Phần cuối phải kể xong kết quả/kết thúc có trong event cuối."
        if part_number == 3 else ""
    )
    return f"""Viết recap tiếng Việt cho phần {part_number}/3 từ kế hoạch câu toàn phim đã khóa.
Mục tiêu {target_words} từ (ít nhất {minimum_words}), đúng {required_scene_count} câu.
Mỗi `id` là một ô câu đã được backend phân bổ trước và phải tạo đúng một TTS theo thứ tự.
Nhiều ô có thể cùng `source_event`, nhưng mỗi ô có `fact`, `slot` và vùng hình riêng; không kể lại ý của ô trước.
Một ô có thể chứa cụm event liên tiếp đã được backend gộp; không tự tách, gộp, bỏ hoặc lặp ID.
Mỗi TTS dài {MIN_TTS_WORDS}-{MAX_TTS_WORDS} từ và cố gắng gần trường `words`.
Các câu phải nối nhau bằng nguyên nhân, tiếp diễn, đối lập hoặc kết quả; câu sau phải thêm ý mới.
Chỉ dùng tên trong `characters`/`who`; không chắc tên thì giữ nguyên biệt danh trung lập.
`spoken` ưu tiên câu chữ SRT/Whisper cho lời thoại và tên riêng, nhưng không chứng minh hành động
hay người nói; hình ảnh trong event map mới quyết định hành động.
TOÀN BỘ TTS phải là tiếng Việt tự nhiên, có dấu. Nếu `spoken` là tiếng nước ngoài thì dịch ý sang
tiếng Việt; không chép câu tiếng Anh, quảng cáo kênh hoặc câu phụ đề vỡ nghĩa vào TTS.
Không dùng câu đệm chung chung như “tình thế vô cùng nguy cấp” chỉ để đủ 15 từ.
Mỗi câu chỉ kể ý có trong đúng event. `time` là vùng phim nguồn để dựng SAU khi viết,
không phải thời lượng giọng đọc. Nếu `visuals` có hai hành động rõ ràng, có thể thêm
`visual_links`: liên kết ý kể với visual_id trong `visuals`, theo thứ tự hình và lời.
Phần tử đầu là hình cho đầu câu, trigger_text="". Mỗi phần tử sau dùng cụm 2–6 từ
xuất hiện đúng một lần trong TTS, ngay đầu ý hành động cần đổi hình.
Ví dụ: [{{"visual_id":"V001","trigger_text":""}},{{"visual_id":"V002","trigger_text":"rồi cô dùng đá"}}].
Chọn hình đúng hành động, nhân vật, kết quả; không cần dùng hết visuals, không chọn theo vị trí.
Không tự đặt timestamp, không chia lời theo tỷ lệ số từ, không ép shot 2–3 giây.
Một hành động liên tục có thể dùng một visual_id cho cả câu. Nhiều ý khác hành động thì
liên kết từng ý với hình phù hợp. Nếu thiếu hình cho ý muốn kể, chỉ viết dữ kiện được visuals hỗ trợ.
{ending_rule}
Phong cách: {style_storytelling_prompt(style)}
{('Yêu cầu thêm: ' + extras) if extras else ''}
DỮ LIỆU: {compact_json}
Chỉ trả JSON, không markdown, đúng dạng tối giản:
{{"scenes":[{{"event_id":"E0001","tts":"câu recap tiếng Việt","visual_links":[{{"visual_id":"V001","trigger_text":""}}]}}]}}
Phải có đủ event_id theo thứ tự: {json.dumps(required_event_ids, ensure_ascii=False)}
"""
    evidence_json = json.dumps(story_context, ensure_ascii=False, separators=(",", ":"))
    ending_rules = ""
    if part_number == 3:
        ending_rules = f"""
- Đây là PHẦN CUỐI: bắt buộc kể đủ required_event mang resolution/ending thật.
- Không được dừng ở cao trào nếu sau đó còn giải cứu, đoàn tụ, trở về hoặc kết quả cuối.
- Không cần kéo timestamp tới cuối file; credit, outro, im lặng hoặc cảnh dư sau ending đều được bỏ qua.
"""
    voice_rule = (
        "Có thể đặt voice riêng cho scene khi thật sự cần nhiều giọng."
        if enable_multi_voice else "Không cần tạo trường voice."
    )
    return f"""Bạn là biên tập viên recap video. Lượt này KHÔNG có video; chỉ viết phần {part_number}/3
từ EVENT MAP đã khóa và không bịa thêm sự kiện.
- `target_words={target_words}`; tổng TTS phải đạt ít nhất {minimum_words} từ và hướng tới khoảng {requested_words} từ.
- `required_events` là các CỤM kể đã được backend nén từ bản đồ sự kiện đầy đủ.
- Mỗi required_event bắt buộc tạo đúng một TTS; nếu có `covered_events`, câu phải truyền đạt chuỗi nguyên nhân-hành động-kết quả của
  tất cả event con theo thứ tự, không được chỉ kể event đầu hoặc event cuối.
- không tách một fact thành nhiều câu; cụm nhiều fact chỉ được nén khi backend đã ghi rõ `covered_events`.
- Mỗi scene có đúng một TTS tiếng Việt tự nhiên, kể một ý trọn vẹn. Độ dài linh hoạt theo diễn biến,
  không cố nhồi hoặc rút chữ để đạt một số từ cố định;
  nhưng các scene cùng `story_beat_id`
  phải nối thành một đoạn recap về cùng biến cố, không mở lại ý và không kể lần lượt từng động tác trong hình.
- `narration_focus` là fact riêng đã được backend giao cho scene. TTS bắt buộc phát triển fact đó và không được
  kể lại `narration_focus`, nguyên nhân, hành động hay hậu quả đã dùng ở scene trước, kể cả khi cùng story_beat_id.
- `previous_part_tail` là nội dung đã kể ở phần trước: chỉ dùng để nối mạch, tuyệt đối không diễn đạt lại các ý đó.
- Mỗi scene chỉ được gọi tên người nằm trong `allowed_character_names`. Nếu hình chưa chắc danh tính thì dùng vai trò
  trung lập có trong event; tuyệt đối không gán hành động của phản diện cho đồng minh hoặc ngược lại.
- `visual_sources` chứa toàn bộ shot nguồn được backend giữ lại. Hãy dùng tất cả shot làm `visual_beats` theo
  đúng thứ tự khi câu đổi hành động; không tách mỗi shot thành một TTS mới, không bỏ source và không tạo
  timestamp ngoài các range đã cung cấp.
- `forbidden_previous_facts` là các fact đã được kể hoặc đã cấp cho câu trước; không được diễn đạt lại bằng từ đồng nghĩa.
- Bắt buộc tạo đúng {required_scene_count} scene/TTS.
- Mỗi required_event đã có `scene_id` do backend khóa. Phải chép đúng scene_id. `tts_target_words`
  chỉ là ước lượng nhịp đọc cho backend, không phải giới hạn bắt buộc; tổng số từ không bị ép cứng.
- Số scene là số FACT thật hoặc CỤM FACT đã khóa. Không gộp nhiều event chưa được backend đặt trong
  `covered_events`; không tự gộp thêm required_event/ID, không lặp ID và không tách cụm để đủ số lượng.
- Nếu gặp `visual_support_only=true` từ dữ liệu cũ, vẫn chỉ kể `narration_focus` mới của scene; không diễn đạt
  lại summary/cause/outcome đã xuất hiện ở scene trước.
- Các required_event có cùng `selection_cluster_id` phải đọc thành MỘT chuỗi liên tục. Nếu là cụm critical,
  phải kể rõ các mốc thật có trong event map theo thứ tự; không nhảy từ mối đe dọa sang kết quả và không
  bỏ qua transformation/attack/reversal nằm giữa.
- Hai scene liên tiếp không được chỉ lặp lại cùng một cảm xúc, mục tiêu hoặc trạng thái. Câu sau phải thêm
  hành động, thông tin, đảo chiều hoặc kết quả mới; nếu không thì chỉ tiếp tục ý của story_beat thay vì mở ý mới.
- Ưu tiên thay đổi mục tiêu, nguy hiểm, quan hệ và kết quả. Bỏ mô tả phong cảnh, di chuyển, ăn uống, chờ đợi,
  nét mặt và cảm xúc suy diễn nếu chúng không làm đổi cốt truyện.
- Timestamp dùng thời gian CỤC BỘ 00:00-{_format_time(clip_duration_sec)}.
- START-END và visual beat hướng tới khoảng 3 giây nhưng không bắt buộc đúng 3 giây.
- Khi câu đổi shot, giữ lời kể mượt và thêm visual_beat; Một câu có thể có 0, 1 hoặc nhiều beat.
- Mọi event và TTS phải bằng tiếng Việt tự nhiên, có dấu; không trả lời bằng tiếng Anh.
- {voice_rule}
- HƯỚNG KỂ ĐÃ CHỌN: {style_storytelling_prompt(style)}
{ending_rules}
{('- Yêu cầu thêm: ' + extras) if extras else ''}
- `previous_event_context` va `next_event_context` la hai mat xich lien ke. Moi TTS phai noi tu nhien voi cau
  truoc va mo duong cho cau sau bang nguyen nhan, ket qua, doi lap, chuyen canh hoac tiep dien phu hop. Khong
  ke lai event truoc, khong tu gop them required_event va khong tiet lo truoc event sau. Rieng cac event con
  trong `covered_events` phai duoc ke gon trong cung cau theo thu tu backend da khoa.
- `spoken_text` va `transcript_evidence` la loi Whisper gan theo timestamp: chi dung lam bang chung,
  khong bia cau trich dan, khong tu gan speaker, va khong uu tien loi noi hon hanh dong nhin thay.
- Neu `source=transcript_bridge`, chi recap thong tin duoc noi ro trong `spoken_text`; khong doan ai dang noi
  va khong bia hanh dong nhin thay. Dung no de noi khoang trong giua hai su kien hinh anh.
- Event khong co `spoken_text` thi khong duoc tu tao loi thoai; van recap canh im lang bang visual evidence.
EVENT MAP: {evidence_json}
covered_event_ids phải đủ: {json.dumps(required_event_ids, ensure_ascii=False)}
Chỉ trả JSON hợp lệ với characters và đúng một phần; mỗi scene phải chép đúng `story_beat_id` từ event tương ứng,
ví dụ "story_beat_id":"B001","story_role":"development","visual_beats":[{{"trigger_text":"người bạn hoảng hốt","start":"00:16","end":"00:19"}}].
"""


def build_backend_visual_plan(narration, visual_sources, visual_links=None, legacy_beats=None):
    """Bind spoken action changes to observed ranges without a fixed shot rhythm."""
    return semantic_visual_plan(narration, visual_sources, visual_links, legacy_beats)


def coerce_single_part_response(
    text, part_number, clip_duration_sec, target_words,
    required_event_ids=None, story_beat_by_event_id=None,
    locked_event_by_id=None, locked_characters=None,
):
    """Normalize equivalent Gemini shapes to exactly one locked part wrapper."""
    source = str(text or "").lstrip("\ufeff").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", source, re.I)
    candidate_source = fenced.group(1).strip() if fenced else source
    payload = None
    try:
        decoded = json.loads(candidate_source, strict=False)
        if isinstance(decoded, list) and all(isinstance(row, dict) for row in decoded):
            payload = {"scenes": decoded}
        elif isinstance(decoded, dict):
            payload = decoded
    except (json.JSONDecodeError, TypeError):
        pass
    if payload is None:
        payload = _json_object(text)
    requested_part = int(part_number)
    locked_ids = [
        str(value).strip().upper() for value in (required_event_ids or [])
        if str(value).strip()
    ]
    beat_lookup = {
        str(event_id).strip().upper(): str(beat_id).strip().upper()
        for event_id, beat_id in (story_beat_by_event_id or {}).items()
        if str(event_id).strip() and str(beat_id).strip()
    }
    locked_events = {
        str(event_id).strip().upper(): event
        for event_id, event in (locked_event_by_id or {}).items()
        if str(event_id).strip() and isinstance(event, dict)
    }

    # Models sometimes add a harmless envelope or rename `scenes`. Walk only
    # known wrapper keys so arbitrary nested scene/state objects are not
    # mistaken for the requested part.
    containers = []
    pending = [payload]
    seen_containers = set()
    wrapper_keys = ("result", "data", "output", "response", "json", "payload")
    while pending:
        container = pending.pop(0)
        if not isinstance(container, dict) or id(container) in seen_containers:
            continue
        seen_containers.add(id(container))
        containers.append(container)
        for key in wrapper_keys:
            wrapped = container.get(key)
            if isinstance(wrapped, dict):
                pending.append(wrapped)

    candidates = []
    seen_candidates = set()

    def add_candidate(candidate, owner):
        if not isinstance(candidate, dict) or id(candidate) in seen_candidates:
            return
        seen_candidates.add(id(candidate))
        candidates.append((candidate, owner))

    scene_aliases = ("scenes", "narration_scenes", "narration_scene", "scene")
    for container in containers:
        parts = container.get("parts")
        if isinstance(parts, dict):
            add_candidate(parts, container)
        elif isinstance(parts, list):
            for part in parts:
                add_candidate(part, container)
        part_value = container.get("part")
        if isinstance(part_value, dict):
            add_candidate(part_value, container)
        for key in (f"part_{requested_part}", f"part{requested_part}"):
            add_candidate(container.get(key), container)
        if any(isinstance(container.get(key), list) for key in scene_aliases):
            add_candidate(container, container)

    def candidate_scenes(candidate):
        for key in scene_aliases:
            value = candidate.get(key)
            if isinstance(value, list):
                return value
        return None

    def safe_part_value(candidate):
        try:
            return int(candidate.get("part", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def candidate_score(item):
        candidate, _ = item
        scenes = candidate_scenes(candidate)
        if scenes is None:
            return (-1, -1, -1, -1)
        mapped = {
            str(event_id).strip().upper()
            for scene in scenes if isinstance(scene, dict)
            for event_id in (
                scene.get("covered_event_ids")
                if isinstance(scene.get("covered_event_ids"), list)
                else [scene.get("covered_event_ids")]
            )
            if str(event_id or "").strip()
        }
        exact_part = int(safe_part_value(candidate) == requested_part)
        overlap = len(set(locked_ids) & mapped)
        count_closeness = -abs(len(scenes) - len(locked_ids)) if locked_ids else 0
        # Locked event coverage is stronger evidence than a part number because
        # models occasionally copy the previous part number while preserving
        # the correct requested scenes.
        if locked_ids:
            return (overlap, exact_part, count_closeness, len(scenes))
        return (exact_part, 0, count_closeness, len(scenes))

    viable = [item for item in candidates if candidate_scenes(item[0]) is not None]
    if not viable:
        keys = ", ".join(sorted(str(key) for key in payload.keys())) or "(rong)"
        raise ValueError(
            "JSON Gemini sai schema: khong tim thay danh sach scenes "
            f"trong part; truong goc: {keys}."
        )
    selected, owner = max(viable, key=candidate_score)
    selected = copy.deepcopy(selected)
    raw_scenes = candidate_scenes(selected)
    selected["scenes"] = [
        copy.deepcopy(scene) for scene in (raw_scenes or [])
        if isinstance(scene, dict)
    ]
    for alias in scene_aliases[1:]:
        selected.pop(alias, None)
    selected["part"] = int(part_number)
    selected.setdefault("video_start", "00:00")
    selected.setdefault("video_end", _format_time(clip_duration_sec))
    selected.setdefault("target_words", int(target_words))
    if not isinstance(selected.get("reserve_scenes"), list):
        selected["reserve_scenes"] = []
    scenes = selected["scenes"]
    # Accept the intentionally tiny writer schema before restoring all locked
    # metadata from the backend contract.
    for scene in scenes:
        if not isinstance(scene, dict) or scene.get("covered_event_ids"):
            continue
        mapped = scene.get("event_id") or scene.get("events") or scene.get("id")
        if mapped:
            scene["covered_event_ids"] = (
                mapped if isinstance(mapped, list) else [mapped]
            )
    # Gemini sometimes preserves every scene in order but omits the mapping
    # field entirely. Restoring this metadata is safe only for a complete
    # one-to-one response; event content and timestamps remain untouched.
    if (
        locked_ids
        and len(scenes) == len(locked_ids)
        and all(not (scene.get("covered_event_ids") or []) for scene in scenes)
    ):
        for scene, event_id in zip(scenes, locked_ids):
            scene["covered_event_ids"] = [event_id]
    for scene in scenes:
        mapped_ids = scene.get("covered_event_ids") or []
        if not isinstance(mapped_ids, list):
            mapped_ids = [mapped_ids]
        mapped_beats = {
            beat_lookup.get(str(event_id).strip().upper(), "")
            for event_id in mapped_ids
        }
        mapped_beats.discard("")
        if len(mapped_beats) == 1:
            scene["story_beat_id"] = mapped_beats.pop()
        normalized_ids = [
            str(event_id).strip().upper() for event_id in mapped_ids
            if str(event_id).strip()
        ]
        # ID, timestamp and evidence metadata belong to the backend contract,
        # never to the writing model.
        if len(normalized_ids) == 1 and normalized_ids[0] in locked_events:
            locked_event = locked_events[normalized_ids[0]]
            if locked_event.get("scene_id"):
                scene["scene_id"] = str(locked_event["scene_id"]).strip().upper()
            scene["tts_target_words"] = int(
                locked_event.get("tts_target_words", 22)
            )
            scene["tts_min_words"] = int(
                locked_event.get("tts_min_words", MIN_TTS_WORDS)
            )
            scene["tts_max_words"] = int(
                locked_event.get("tts_max_words", MAX_TTS_WORDS)
            )
            for metadata_key in (
                "narrative_function", "story_beat_summary",
                "story_beat_cause", "story_beat_outcome",
                "story_beat_position", "story_beat_size",
                "climax_spine_role", "visual_support_only",
                "derived_visual_slot", "derived_narration_slot",
                "narration_focus", "selection_cluster_id",
                "selection_importance", "causal_anchor", "event_kind",
                "repeat_group", "fact_id", "forbidden_previous_facts",
                "supporting_facts", "visual_sources", "covered_events",
                "allowed_character_names", "merged_evidence_ids",
                "previous_event_context", "next_event_context",
                "source", "source_confidence", "spoken_text", "transcript_evidence",
                "estimated_tts_seconds", "available_visual_seconds",
                "visual_coverage_risk",
                "visual_selection_pending", "planning_source_event_ids",
                "observed_facts", "planned_fact_ids", "grounded_plan_locked",
                "situation_narration",
                "sentence_quota", "source_link_contract_version", "visible_character_names",
                "mentioned_character_names",
                "sentence_slots", "evidence_sentence_plan_version",
                "paragraph_plan_version",
                "narration_unit_version",
            ):
                if metadata_key in locked_event:
                    scene[metadata_key] = copy.deepcopy(
                        locked_event.get(metadata_key)
                    )
            # The backend, not the prose model, owns the visual EDL. Each TIME
            # and BEAT below comes from a short shot observed in the video pass.
            if locked_event.get('sentence_quota') and not locked_event.get('narration_unit_version'):
                from narrative_contract import constrain_repair_sentences
                scene['tts'] = constrain_repair_sentences(scene.get('tts', ''), locked_event['sentence_quota'])
            if locked_event.get('visual_selection_pending'):
                scene.update(start=locked_event['start'], end=locked_event['end'],
                             visual_selection_pending=True, visual_alignment_version=0,
                             visual_alignment_errors=[], visual_beats=[], visual_links=[], locked_visual_shots=[])
                continue
            visual_plan = build_backend_visual_plan(
                scene.get("tts", ""), locked_event.get("visual_sources") or [],
                ([{"visual_id": visual_catalog(locked_event.get('visual_sources') or [])[0]['visual_id'],
                   "trigger_text": ""}]
                 if len(visual_catalog(locked_event.get('visual_sources') or [])) == 1
                 else scene.get("visual_links")), scene.get("visual_beats"),
            )
            if locked_event.get('grounded_plan_locked') and not locked_event.get('situation_narration'):
                expected_visuals = visual_catalog(locked_event.get('visual_sources') or [])
                if len(visual_plan['locked_visual_shots']) != len(expected_visuals):
                    visual_plan['visual_alignment_errors'].append('grounded_plan_shots_missing')
            scene["start"] = visual_plan["start"]
            scene["end"] = visual_plan["end"]
            scene["visual_beats"] = visual_plan["visual_beats"]
            scene["locked_visual_shots"] = visual_plan["locked_visual_shots"]
            for key in ("visual_links", "visual_alignment_errors", "visual_alignment_version"):
                scene[key] = visual_plan[key]
    if locked_characters is not None:
        characters = copy.deepcopy([
            character for character in locked_characters
            if isinstance(character, dict)
        ])
    else:
        characters = owner.get("characters")
        if not isinstance(characters, list):
            characters = payload.get("characters")
        if not isinstance(characters, list):
            characters = selected.pop("characters", [])
        characters = characters if isinstance(characters, list) else []
    display_labels = _character_display_labels(characters)
    for scene in selected["scenes"]:
        for field in (
            "event", "tts", "transition_text", "state_before", "state_after",
        ):
            if field in scene:
                scene[field] = _replace_internal_character_ids(
                    scene.get(field, ""), display_labels
                )
        for link in list(scene.get("visual_links") or []) + list(scene.get("visual_beats") or []):
            if isinstance(link, dict):
                link["trigger_text"] = _replace_internal_character_ids(
                    link.get("trigger_text", ""), display_labels
                )
    return json.dumps({
        "characters": characters,
        "parts": [selected],
    }, ensure_ascii=False)


def build_single_part_fallback_response(
    story_context, part_number, clip_duration_sec, target_words,
):
    """Build a grounded last-resort part from the already locked event map.

    This is used only when repeated model responses violate the JSON schema.
    It never invents timestamps or events, so a remote formatting failure does
    not discard a long-running analysis job.
    """
    context = story_context if isinstance(story_context, dict) else {}
    events = [
        copy.deepcopy(event) for event in (context.get("required_events") or [])
        if isinstance(event, dict) and str(event.get("event_id", "")).strip()
    ]
    if not events:
        raise ValueError("Khong co event map da khoa de tao kich ban du phong.")

    def valid_range(row):
        try:
            start = _time_seconds(row.get("start"))
            end = _time_seconds(row.get("end"))
        except (AttributeError, TypeError, ValueError):
            return None
        if not 0 <= start < end <= float(clip_duration_sec) + 1.0:
            return None
        return start, end

    def match_tokens(row):
        text = " ".join(str(row.get(key, "") or "") for key in (
            "event", "tts", "visual_detail", "cause", "consequence", "change",
            "story_beat_summary", "story_beat_cause", "story_beat_outcome",
        )).casefold()
        return {
            token for token in re.findall(r"[^\W_]{2,}", text, flags=re.UNICODE)
            if token not in REPETITION_STOPWORDS
        }

    reserve_candidates = [
        copy.deepcopy(row)
        for row in (context.get("fallback_visual_candidates") or [])
        if isinstance(row, dict) and valid_range(row) is not None
    ]
    event_candidates = [
        copy.deepcopy(row) for row in events if valid_range(row) is not None
    ]
    visual_candidates = reserve_candidates + event_candidates
    used_reserve_ids = set()

    def best_visual_fallback(event, event_index, previous_start, next_start):
        target_tokens = match_tokens(event)
        target_characters = {
            str(value).strip().upper() for value in (event.get("characters") or [])
            if str(value).strip()
        }
        target_beat = str(event.get("story_beat_id", "")).strip().upper()
        target_role = str(event.get("story_role", "")).strip().lower()
        scored = []
        for candidate_index, candidate in enumerate(visual_candidates):
            candidate_range = valid_range(candidate)
            if candidate_range is None:
                continue
            start, end = candidate_range
            candidate_id = str(
                candidate.get("scene_id") or candidate.get("event_id") or ""
            ).strip().upper()
            is_reserve = candidate_index < len(reserve_candidates)
            if is_reserve and candidate_id and candidate_id in used_reserve_ids:
                continue
            # Keep the repaired response in chronological order. Equal starts
            # are allowed as a final degraded fallback because repeating one
            # relevant shot is safer than aborting the complete analysis.
            if previous_start is not None and start < previous_start:
                continue
            if next_start is not None and start > next_start:
                continue
            candidate_tokens = match_tokens(candidate)
            overlap = len(target_tokens & candidate_tokens)
            union = len(target_tokens | candidate_tokens) or 1
            candidate_characters = {
                str(value).strip().upper()
                for value in (candidate.get("characters") or [])
                if str(value).strip()
            }
            score = 100.0 * overlap / union
            if target_beat and target_beat == str(
                candidate.get("story_beat_id", "")
            ).strip().upper():
                score += 80.0
            if target_characters and candidate_characters:
                score += 35.0 * len(target_characters & candidate_characters) / len(
                    target_characters | candidate_characters
                )
            if target_role and target_role == str(
                candidate.get("story_role", "")
            ).strip().lower():
                score += 12.0
            if is_reserve:
                score += 20.0
            score -= abs(candidate_index - event_index) * 0.25
            scored.append((score, -abs(candidate_index - event_index), candidate, start, end, is_reserve))
        if not scored:
            return None
        _, _, candidate, start, end, is_reserve = max(
            scored, key=lambda item: (item[0], item[1])
        )
        source_id = str(
            candidate.get("scene_id") or candidate.get("event_id") or ""
        ).strip().upper()
        if is_reserve and source_id:
            used_reserve_ids.add(source_id)
        return start, end, source_id or "VALID_SCENE", is_reserve

    def fallback_tts(event):
        fragments = []
        seen = set()
        covered_rows = [
            row for row in (event.get("covered_events") or [])
            if isinstance(row, dict)
        ]
        for row in covered_rows:
            for key in ("cause", "event", "change", "consequence"):
                fragment = _replace_internal_character_ids(row.get(key, ""))
                fragment = re.sub(r"\s+", " ", fragment).strip(" .,:;!?-")
                if (
                    fragment and not _looks_like_english_prose(fragment)
                    and fragment.casefold() not in seen
                ):
                    seen.add(fragment.casefold())
                    fragments.append(fragment)
        for key in (
            "narration_focus", "event", "story_beat_cause",
            "story_beat_outcome", "consequence", "change", "visual_detail", "cause",
            "story_beat_summary",
        ):
            fragment = _replace_internal_character_ids(event.get(key, ""))
            fragment = re.sub(r"\s+", " ", fragment).strip(" .,:;!?-")
            if _looks_like_english_prose(fragment):
                continue
            if fragment and fragment.casefold() not in seen:
                seen.add(fragment.casefold())
                fragments.append(fragment)
        if not fragments:
            fragments = ["Không đủ bằng chứng cụ thể để mô tả chính xác cảnh này"]
        sentence = fragments[0]
        connectors = (", vì vậy ", ", đồng thời ", ", qua đó ", ", sau khi ")
        for index, fragment in enumerate(fragments[1:]):
            candidate = sentence + connectors[min(index, len(connectors) - 1)] + fragment
            if len(candidate.split()) <= MAX_TTS_WORDS:
                sentence = candidate
        if len(sentence.split()) < MIN_TTS_WORDS:
            concrete_event = _replace_internal_character_ids(event.get("event", ""))
            concrete_result = _replace_internal_character_ids(
                event.get("consequence") or event.get("change") or event.get("result") or ""
            )
            concrete_visual = _replace_internal_character_ids(
                event.get("visual_detail", "")
            )
            if concrete_event and concrete_result:
                candidate = (
                    f"{concrete_event}, khiến {concrete_result} xuất hiện như kết quả "
                    "trực tiếp ngay trong cảnh này"
                )
            elif concrete_event and concrete_visual:
                candidate = (
                    f"{concrete_event}, hình ảnh xác nhận {concrete_visual} xuất hiện "
                    "rõ ràng ngay tại thời điểm đó"
                )
            else:
                candidate = sentence
            if len(candidate.split()) <= MAX_TTS_WORDS:
                sentence = candidate
        if len(sentence.split()) > MAX_TTS_WORDS:
            clauses = re.split(r"(?<=[,;])\s+", sentence)
            kept = []
            for clause in clauses:
                if len(" ".join(kept + [clause]).split()) > MAX_TTS_WORDS:
                    break
                kept.append(clause)
            sentence = " ".join(kept) if kept else " ".join(
                sentence.split()[:MAX_TTS_WORDS]
            )
        return sentence.rstrip(" ,;:.") + "."

    original_ranges = [valid_range(event) for event in events]
    scenes = []
    previous_start = None
    for index, event in enumerate(events, start=1):
        event_range = original_ranges[index - 1]
        fallback_source = None
        fallback_is_reserve = False
        if event_range is None:
            next_start = next(
                (row[0] for row in original_ranges[index:] if row is not None),
                None,
            )
            replacement = best_visual_fallback(
                event, index - 1, previous_start, next_start
            )
            if replacement is not None:
                start_seconds, end_seconds, fallback_source, fallback_is_reserve = replacement
                event_range = (start_seconds, end_seconds)
            else:
                # No valid timestamp exists anywhere in this part. Use a short,
                # position-preserving window so one malformed scene cannot stop
                # the complete job. The metadata below makes this auditable.
                duration = max(1.0, float(clip_duration_sec))
                slot = duration / max(1, len(events))
                start_seconds = min(duration - 0.25, max(0.0, (index - 1) * slot))
                end_seconds = min(duration, max(start_seconds + 0.25, index * slot))
                event_range = (start_seconds, end_seconds)
                fallback_source = "POSITIONAL_WINDOW"
            required_ending = context.get("required_ending_sec")
            if index == len(events) and required_ending is not None:
                try:
                    required_ending = min(
                        float(clip_duration_sec), float(required_ending)
                    )
                except (TypeError, ValueError):
                    required_ending = None
                if required_ending is not None and event_range[1] < required_ending:
                    shot_length = max(0.25, event_range[1] - event_range[0])
                    ending_start = max(
                        float(previous_start or 0.0), required_ending - shot_length
                    )
                    event_range = (ending_start, required_ending)
                    fallback_source = "POSITIONAL_ENDING_WINDOW"
                    fallback_is_reserve = False
        start_seconds, end_seconds = event_range
        previous_start = start_seconds
        scene = {
            "scene_id": str(event.get("scene_id") or f"S{index:04d}").strip().upper(),
            "start": _format_time(start_seconds),
            "end": _format_time(end_seconds),
            "type": (
                "bridge" if str(event.get("importance", "")).lower() == "bridge"
                else "main"
            ),
            "characters": [str(value) for value in (event.get("characters") or [])],
            "event": str(event.get("event", "")).strip(),
            "tts": fallback_tts(event),
            "tts_target_words": int(event.get("tts_target_words", 22)),
            "tts_min_words": int(event.get("tts_min_words", MIN_TTS_WORDS)),
            "tts_max_words": int(event.get("tts_max_words", MAX_TTS_WORDS)),
            "covered_event_ids": [str(event["event_id"]).strip().upper()],
            "story_beat_id": str(event.get("story_beat_id", "")).strip().upper(),
            "selection_cluster_id": str(
                event.get("selection_cluster_id", "")
            ).strip(),
            "selection_importance": str(
                event.get("selection_importance", "")
            ).strip().lower(),
            "causal_anchor": bool(event.get("causal_anchor", False)),
            "event_kind": str(event.get("event_kind", "")).strip().lower(),
            "repeat_group": str(event.get("repeat_group", "")).strip(),
            "story_role": str(event.get("story_role") or "development").lower(),
            "narrative_function": str(
                event.get("narrative_function") or "development"
            ).lower(),
            "story_beat_summary": str(event.get("story_beat_summary", "")).strip(),
            "story_beat_cause": str(event.get("story_beat_cause", "")).strip(),
            "story_beat_outcome": str(event.get("story_beat_outcome", "")).strip(),
            "story_beat_position": event.get("story_beat_position", 0),
            "story_beat_size": event.get("story_beat_size", 0),
            "climax_spine_role": str(
                event.get("climax_spine_role", "")
            ).strip().lower(),
            "visual_support_only": bool(event.get("visual_support_only", False)),
            "derived_visual_slot": bool(event.get("derived_visual_slot", False)),
            "derived_narration_slot": bool(
                event.get("derived_narration_slot", False)
            ),
            "narration_focus": str(event.get("narration_focus", "")).strip(),
            "visual_beats": [],
            "status": "fallback",
            "errors": [],
        }
        if fallback_source is not None:
            scene["timestamp_fallback_from"] = fallback_source
            if fallback_source.startswith("POSITIONAL_"):
                scene["timestamp_fallback_kind"] = "positional_window"
            else:
                scene["timestamp_fallback_kind"] = (
                    "reserve_scene" if fallback_is_reserve else "best_valid_scene"
                )
        scenes.append(scene)
    return json.dumps({
        "characters": context.get("characters") or [],
        "parts": [{
            "part": int(part_number),
            "video_start": "00:00",
            "video_end": _format_time(clip_duration_sec),
            "target_words": int(target_words),
            "scenes": scenes,
            "reserve_scenes": [],
        }],
    }, ensure_ascii=False)


def single_part_missing_requirements(
    text, clip_duration_sec, target_words, required_event_ids=None,
    require_word_floor=False, required_ending_sec=None,
    require_tts_word_range=False, required_scene_count=None,
    required_story_anchor_ids=None, require_narrative_links=False,
    require_timeline_spread=False,
):
    """Measure one part by required events, narration length, and ending coverage."""
    payload = _json_object(text)
    parts = payload.get("parts")
    if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
        raise ValueError("Kết quả một phần phải chứa đúng một phần.")
    part = parts[0]
    scenes = [row for row in (part.get("scenes") or []) if isinstance(row, dict)]
    duration = max(1.0, float(clip_duration_sec))
    valid_ranges = []
    invalid_timestamp_scenes = []
    previous_start = -1.0
    for index, scene in enumerate(scenes, start=1):
        scene_id = str(scene.get("scene_id") or f"SCENE_{index}")
        if scene.get("visual_alignment_errors"):
            invalid_timestamp_scenes.append({
                "scene_id": scene_id,
                "reasons": list(scene["visual_alignment_errors"]),
            })
        try:
            start = _time_seconds(scene.get("start"))
            end = _time_seconds(scene.get("end"))
        except (TypeError, ValueError):
            reasons = []
            if scene.get("start") in (None, ""):
                reasons.append("missing_start")
            if scene.get("end") in (None, ""):
                reasons.append("missing_end")
            invalid_timestamp_scenes.append({
                "scene_id": scene_id,
                "reasons": reasons or ["invalid_format"],
            })
            continue
        reasons = []
        if start < 0 or end > duration + 1.0:
            reasons.append("out_of_range")
        if end <= start:
            reasons.append("nonpositive")
        if start < previous_start:
            reasons.append("out_of_order")
        if reasons:
            invalid_timestamp_scenes.append({"scene_id": scene_id, "reasons": reasons})
            continue
        valid_ranges.append((start, end))
        previous_start = start

    actual_words = sum(len(str(scene.get("tts", "")).split()) for scene in scenes)
    invalid_tts_scenes = []
    tts_length_warnings = []
    if require_tts_word_range:
        previous_tokens = None
        for index, scene in enumerate(scenes, start=1):
            narration = re.sub(r"\s+", " ", str(scene.get("tts", ""))).strip()
            word_count = len(narration.split())
            scene_min_words = max(
                1, int(scene.get("tts_min_words", MIN_TTS_WORDS))
            )
            scene_max_words = max(
                scene_min_words,
                int(scene.get("tts_max_words", MAX_TTS_WORDS)),
            )
            sentence_fragments = [
                value.strip() for value in re.split(r"[.!?;]+", narration)
                if value.strip()
            ]
            fragmented = (
                len(sentence_fragments) >= 3
                and any(len(value.split()) < 5 for value in sentence_fragments)
            )
            outside_word_range = (
                word_count < scene_min_words or word_count > scene_max_words
            )
            # Word length and fragmentation are editorial warnings only. They
            # must never trigger another Gemini request. Vietnamese validity
            # remains a hard requirement.
            language_issues = vietnamese_narration_issues(narration)
            if outside_word_range or fragmented:
                warning = {
                    "scene_id": str(scene.get("scene_id") or f"SCENE_{index}"),
                    "word_count": word_count,
                }
                if outside_word_range:
                    warning["outside_word_range"] = True
                    warning["minimum_words"] = scene_min_words
                    warning["maximum_words"] = scene_max_words
                if fragmented:
                    warning["fragmented"] = True
                tts_length_warnings.append(warning)
            if language_issues:
                issue = {
                    "scene_id": str(scene.get("scene_id") or f"SCENE_{index}"),
                    "word_count": word_count,
                    "language_issues": language_issues,
                }
                invalid_tts_scenes.append(issue)
            narration_tokens = {
                token for token in re.findall(
                    r"[^\W_]{2,}", narration.casefold(), flags=re.UNICODE
                ) if token not in REPETITION_STOPWORDS
            }
            smaller = min(len(previous_tokens or set()), len(narration_tokens))
            if (
                previous_tokens is not None
                and smaller >= 6
                and len(previous_tokens & narration_tokens) / smaller >= 0.88
            ):
                scene_id = str(scene.get("scene_id") or f"SCENE_{index}")
                existing = next((
                    item for item in invalid_tts_scenes
                    if item["scene_id"] == scene_id
                ), None)
                if existing is None:
                    invalid_tts_scenes.append({
                        "scene_id": scene_id,
                        "word_count": word_count,
                        "repeated_previous": True,
                    })
                else:
                    existing["repeated_previous"] = True
            previous_tokens = narration_tokens
    registered_character_names = []
    for character in payload.get("characters") or []:
        if not isinstance(character, dict):
            continue
        aliases = character.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = [aliases]
        for value in [character.get("name"), *aliases]:
            name = re.sub(r"\s+", " ", str(value or "")).strip()
            if name and name.casefold() not in {
                known.casefold() for known in registered_character_names
            }:
                registered_character_names.append(name)
    invalid_character_mentions = []
    for index, scene in enumerate(scenes, start=1):
        allowed = scene.get("allowed_character_names") or []
        if not isinstance(allowed, list):
            allowed = [allowed]
        allowed_keys = {
            re.sub(r"\s+", " ", str(value or "")).strip().casefold()
            for value in allowed if str(value or "").strip()
        }
        if not allowed_keys:
            continue
        narration = str(scene.get("tts", ""))
        unexpected = []
        for name in registered_character_names:
            if name.casefold() in allowed_keys:
                continue
            if character_name_in_text(name, narration):
                unexpected.append(name)
        if unexpected:
            invalid_character_mentions.append({
                "scene_id": str(scene.get("scene_id") or f"SCENE_{index}"),
                "unexpected_names": unexpected,
                "allowed_names": sorted(allowed_keys),
            })
    invalid_narrative_links = []
    if require_narrative_links:
        for index, scene in enumerate(scenes):
            scene_id = str(scene.get("scene_id") or f"SCENE_{index + 1}")
            relation = str(scene.get("relation_to_previous", "")).strip().lower()
            state_before = str(scene.get("state_before", "")).strip()
            state_after = str(scene.get("state_after", "")).strip()
            transition = re.sub(
                r"\s+", " ", str(scene.get("transition_text", ""))
            ).strip()
            narration = re.sub(r"\s+", " ", str(scene.get("tts", ""))).strip()
            issues = []
            if relation not in NARRATIVE_RELATIONS:
                issues.append("invalid_relation")
            if not state_before or not state_after:
                issues.append("missing_state")
            if relation in EXPLICIT_TRANSITION_RELATIONS:
                transition_words = len(re.findall(r"\w+", transition, re.UNICODE))
                if not 1 <= transition_words <= 6:
                    issues.append("missing_transition")
                elif narration.casefold().count(transition.casefold()) != 1:
                    issues.append("transition_not_in_tts")
            if index > 0 and relation == "opening":
                issues.append("reopened_story")
            if issues:
                invalid_narrative_links.append({
                    "scene_id": scene_id, "relation": relation, "issues": issues,
                })

    timeline_clustering = None
    if require_timeline_spread and len(valid_ranges) >= 4 and duration >= 300.0:
        ordered_ranges = sorted(valid_ranges)
        first_start = ordered_ranges[0][0]
        last_end = ordered_ranges[-1][1]
        span_ratio = max(0.0, last_end - first_start) / duration
        leading_ratio = max(0.0, first_start) / duration
        trailing_ratio = max(0.0, duration - last_end) / duration
        internal_gap = max(
            (later[0] - earlier[1]
             for earlier, later in zip(ordered_ranges, ordered_ranges[1:])),
            default=0.0,
        )
        if (span_ratio < 0.45 or leading_ratio > 0.55
                or trailing_ratio > 0.55 or internal_gap > duration * 0.60):
            timeline_clustering = {
                "span_ratio": round(span_ratio, 3),
                "leading_seconds": round(first_start, 2),
                "trailing_seconds": round(max(0.0, duration - last_end), 2),
                "largest_internal_gap_seconds": round(internal_gap, 2),
            }
    requested_words, minimum_words, maximum_words = requested_word_range(target_words)
    scene_event_ids = []
    covered_event_ids = set()
    for index, scene in enumerate(scenes, start=1):
        raw_ids = scene.get("covered_event_ids") or []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        normalized_ids = [
            str(event_id).strip().upper()
            for event_id in raw_ids
            if str(event_id).strip()
        ]
        scene_event_ids.append((
            str(scene.get("scene_id") or f"SCENE_{index}"),
            normalized_ids,
        ))
        covered_event_ids.update(normalized_ids)
    required_event_ids = {
        str(event_id).strip().upper()
        for event_id in (required_event_ids or [])
        if str(event_id).strip()
    }
    missing_event_ids = sorted(required_event_ids - covered_event_ids)
    required_story_anchor_ids = {
        str(event_id).strip().upper()
        for event_id in (required_story_anchor_ids or [])
        if str(event_id).strip()
    }
    covered_story_anchor_ids = {
        str(event_id).strip().upper()
        for scene in scenes
        for event_id in (
            scene.get("covered_story_anchor_ids")
            if isinstance(scene.get("covered_story_anchor_ids"), list)
            else [scene.get("covered_story_anchor_ids")]
        )
        if str(event_id or "").strip()
    }
    missing_story_anchor_ids = sorted(
        required_story_anchor_ids - covered_story_anchor_ids
    )
    unknown_story_anchor_ids = sorted(
        covered_story_anchor_ids - required_story_anchor_ids
    ) if required_story_anchor_ids else []
    invalid_event_mappings = []
    seen_mapped_ids = set()
    if required_event_ids:
        for scene_id, event_ids in scene_event_ids:
            if len(event_ids) != 1:
                invalid_event_mappings.append({
                    "scene_id": scene_id,
                    "covered_event_ids": event_ids,
                    "issue": "missing_id" if not event_ids else "multiple_ids",
                })
                continue
            event_id = event_ids[0]
            if event_id not in required_event_ids:
                invalid_event_mappings.append({
                    "scene_id": scene_id,
                    "covered_event_ids": event_ids,
                    "issue": "unknown_id",
                })
            elif event_id in seen_mapped_ids:
                invalid_event_mappings.append({
                    "scene_id": scene_id,
                    "covered_event_ids": event_ids,
                    "issue": "duplicate_id",
                })
            else:
                seen_mapped_ids.add(event_id)
    last_scene_end = max((end for start, end in valid_ranges), default=0.0)
    # A grounded event map already identifies the real ending. Do not force
    # the narration to reach the end of the physical file: credits, outro,
    # silence, or an early final shot are all valid. Coverage is semantic via
    # required_event_ids; the old timestamp rule remains only for legacy calls
    # that do not have an event map.
    if required_event_ids:
        coverage_target = last_scene_end
        coverage_distance = 0.0
        ending_missing = False
    else:
        coverage_target = (
            max(0.0, min(duration, float(required_ending_sec)))
            if required_ending_sec is not None else duration * 0.90
        )
        coverage_distance = max(0.0, coverage_target - last_scene_end)
        ending_missing = coverage_distance > 0
    word_deficit = max(0, minimum_words - actual_words) if require_word_floor else 0
    word_excess = max(0, actual_words - maximum_words) if require_word_floor else 0
    scene_count = len(scenes)
    scene_count_gap = (
        abs(scene_count - int(required_scene_count))
        if required_scene_count is not None else 0
    )
    coverage_missing = (
        ending_missing or bool(missing_event_ids) or bool(invalid_event_mappings)
        or bool(missing_story_anchor_ids) or bool(unknown_story_anchor_ids)
    )
    if invalid_timestamp_scenes:
        action = "timestamps"
    elif (missing_event_ids or invalid_event_mappings
            or missing_story_anchor_ids or unknown_story_anchor_ids):
        action = "events"
    elif ending_missing:
        action = "coverage"
    elif scene_count_gap:
        action = "scene_count"
    elif invalid_character_mentions:
        action = "characters"
    elif invalid_tts_scenes:
        action = (
            "tts_language"
            if any(item.get("language_issues") for item in invalid_tts_scenes)
            else "tts_quality"
        )
    elif invalid_narrative_links:
        action = "continuity"
    elif timeline_clustering:
        action = "timeline_spread"
    elif word_deficit or word_excess:
        action = "words"
    else:
        action = "ok"
    needs_revision = (
        coverage_missing or scene_count_gap > 0 or bool(invalid_timestamp_scenes)
        or bool(invalid_character_mentions) or bool(invalid_tts_scenes)
        or bool(invalid_narrative_links)
        or bool(timeline_clustering) or word_deficit > 0 or word_excess > 0
    )
    tts_length_distance = 100 * sum(
        1 for item in invalid_tts_scenes if item.get("language_issues")
    )
    distance = coverage_distance
    if required_event_ids or require_word_floor:
        distance = (
            (len(missing_event_ids) + len(invalid_event_mappings)
             + len(missing_story_anchor_ids) + len(unknown_story_anchor_ids)) * 1_000_000_000
            + len(invalid_timestamp_scenes) * 100_000_000
            + coverage_distance * 1_000_000
            + scene_count_gap * 100_000
            + len(invalid_character_mentions) * 50_000
            + tts_length_distance * 1_000
            + len(invalid_narrative_links) * 500
            + (10_000 if timeline_clustering else 0)
            + word_deficit + word_excess
        )
    return {
        "actual_words": actual_words,
        "requested_words": requested_words,
        "minimum_words": minimum_words,
        "word_deficit": word_deficit,
        "word_excess": word_excess,
        "maximum_words": maximum_words,
        "action": action,
        "coverage_missing": coverage_missing,
        "ending_missing": ending_missing,
        "missing_event_ids": missing_event_ids,
        "missing_story_anchor_ids": missing_story_anchor_ids,
        "unknown_story_anchor_ids": unknown_story_anchor_ids,
        "invalid_event_mappings": invalid_event_mappings,
        "invalid_timestamp_scenes": invalid_timestamp_scenes,
        "invalid_character_mentions": invalid_character_mentions,
        "invalid_tts_scenes": invalid_tts_scenes,
        "tts_length_warnings": tts_length_warnings,
        "invalid_narrative_links": invalid_narrative_links,
        "timeline_clustering": timeline_clustering,
        "tts_length_distance": tts_length_distance,
        "scene_count": scene_count,
        "required_scene_count": required_scene_count,
        "scene_count_gap": scene_count_gap,
        "needs_revision": needs_revision,
        "distance": distance,
    }


def single_part_has_hard_failures(requirements):
    """Return whether a candidate is unsafe to keep after writing retries."""
    result = requirements if isinstance(requirements, dict) else {}
    return bool(
        result.get("coverage_missing")
        or result.get("invalid_timestamp_scenes")
        or result.get("scene_count_gap")
    )


def merge_single_part_responses(responses, boundaries, target_words, *, single_timeline=False):
    """Merge three clip-local Gemini responses into one global payload."""
    if single_timeline:
        if len(responses) != 1 or len(boundaries) != 2:
            raise ValueError('GLOBAL_WRITING_LAYOUT_INVALID')
    elif len(responses) != 3 or len(boundaries) != 4:
        raise ValueError("Cần đúng ba kết quả và bốn mốc ranh giới để ghép.")

    merged_characters = []
    seen_character_ids = set()
    merged_parts = []
    active_id = 1
    used_active_ids = set()
    reserve_id = 9001

    for index, response_text in enumerate(responses):
        part_number = index + 1
        payload = _json_object(response_text)
        raw_parts = payload.get("parts")
        if not isinstance(raw_parts, list) or len(raw_parts) != 1:
            raise ValueError(f"Kết quả phần {part_number} phải chứa đúng một phần.")
        raw_part = copy.deepcopy(raw_parts[0])
        if not isinstance(raw_part, dict):
            raise ValueError(f"Kết quả phần {part_number} không phải object.")

        for character in payload.get("characters") or []:
            if not isinstance(character, dict):
                continue
            character_id = str(character.get("id", "")).strip().upper()
            if character_id and character_id not in seen_character_ids:
                seen_character_ids.add(character_id)
                merged_characters.append(copy.deepcopy(character))

        offset = float(boundaries[index])
        raw_part["part"] = part_number
        raw_part["video_start"] = _format_time(boundaries[index])
        raw_part["video_end"] = _format_time(boundaries[index + 1])

        for key in ("scenes", "reserve_scenes"):
            rows = raw_part.get(key)
            if not isinstance(rows, list):
                rows = []
            rows.sort(key=lambda row: _time_seconds(row.get("start")) if isinstance(row, dict) else 0.0)
            normalized_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row = copy.deepcopy(row)
                row["start"] = _format_time(offset + _time_seconds(row.get("start")))
                row["end"] = _format_time(offset + _time_seconds(row.get("end")))
                for beat in row.get("visual_beats") or []:
                    if isinstance(beat, dict):
                        beat["start"] = _format_time(offset + _time_seconds(beat.get("start")))
                        beat["end"] = _format_time(offset + _time_seconds(beat.get("end")))
                for shot in row.get("locked_visual_shots") or []:
                    if isinstance(shot, dict):
                        shot["start"] = _format_time(offset + _time_seconds(shot.get("start")))
                        shot["end"] = _format_time(offset + _time_seconds(shot.get("end")))
                        shot['start_seconds'] = _time_seconds(shot['start'])
                        shot['end_seconds'] = _time_seconds(shot['end'])
                if row.get('grounded_plan_locked'):
                    for source in row.get('visual_sources') or []:
                        source['start'] = _format_time(offset + _time_seconds(source['start']))
                        source['end'] = _format_time(offset + _time_seconds(source['end']))
                        source['start_seconds'] = _time_seconds(source['start'])
                        source['end_seconds'] = _time_seconds(source['end'])
                if key == "scenes":
                    locked_id = str(row.get("scene_id", "")).strip().upper()
                    if not re.fullmatch(r"S\d{4,}", locked_id) or locked_id in used_active_ids:
                        while f"S{active_id:04d}" in used_active_ids:
                            active_id += 1
                        locked_id = f"S{active_id:04d}"
                    row["scene_id"] = locked_id
                    used_active_ids.add(locked_id)
                    active_id += 1
                else:
                    row["scene_id"] = f"S{reserve_id:04d}"
                    reserve_id += 1
                normalized_rows.append(row)
            raw_part[key] = normalized_rows
        merged_parts.append(raw_part)

    if any(scene.get('visual_selection_pending') for part in merged_parts for scene in part.get('scenes') or []):
        # Preserve the narrative order and event IDs. Provisional event windows
        # must never be subdivided/reordered as if they were selected shots.
        return {'characters': merged_characters, 'parts': merged_parts,
                'visual_selection_mode': 'sentence_first'}

    if all(scene.get('grounded_plan_locked') for part in merged_parts for scene in part.get('scenes') or []):
        # These source windows and IDs were locked before prose. Never run the
        # legacy timestamp collision repair or strip their observation proofs.
        return {'characters': merged_characters, 'parts': merged_parts,
                'visual_selection_mode': 'grounded_evidence_first'}

    # Timestamp is the source of truth at the final join. Events from Gemini
    # and Whisper can legitimately share one source timestamp, so first order
    # every active scene globally, split an identical-start window into small
    # chronological ranges, and only then assign the public S0001... IDs.
    ordered_active = [
        (part_index, original_index, scene)
        for part_index, part in enumerate(merged_parts)
        for original_index, scene in enumerate(part.get("scenes") or [])
        if isinstance(scene, dict)
    ]
    ordered_active.sort(key=lambda item: (
        _time_seconds(item[2].get("start")),
        _time_seconds(item[2].get("end")),
        item[0], item[1],
    ))

    cursor = 0
    while cursor < len(ordered_active):
        group_start = _time_seconds(ordered_active[cursor][2].get("start"))
        group_end_index = cursor + 1
        while (
            group_end_index < len(ordered_active)
            and abs(
                _time_seconds(ordered_active[group_end_index][2].get("start"))
                - group_start
            ) < 0.005
        ):
            group_end_index += 1
        group = ordered_active[cursor:group_end_index]
        if len(group) > 1:
            original_ends = [
                _time_seconds(item[2].get("end")) for item in group
            ]
            next_start = (
                _time_seconds(ordered_active[group_end_index][2].get("start"))
                if group_end_index < len(ordered_active) else max(original_ends)
            )
            usable_end = max(
                group_start + 0.02 * len(group),
                min(next_start, max(original_ends)),
            )
            step = max(0.02, (usable_end - group_start) / len(group))
            for group_index, (_, _, scene) in enumerate(group):
                repaired_start = group_start + step * group_index
                repaired_end = (
                    group_start + step * (group_index + 1)
                    if group_index + 1 < len(group)
                    else max(repaired_start + 0.02, min(max(original_ends), next_start))
                )
                scene["start"] = _format_time(repaired_start)
                scene["end"] = _format_time(max(repaired_start + 0.02, repaired_end))
                scene["timestamp_collision_repaired"] = True
        cursor = group_end_index

    # A timestamp repair can change ordering within an equal-start group.
    ordered_active.sort(key=lambda item: (
        _time_seconds(item[2].get("start")),
        _time_seconds(item[2].get("end")),
        item[0], item[1],
    ))
    for scene_number, (_, _, scene) in enumerate(ordered_active, start=1):
        scene["scene_id"] = f"S{scene_number:04d}"
    for part in merged_parts:
        part["scenes"] = sorted(
            part.get("scenes") or [],
            key=lambda scene: (
                _time_seconds(scene.get("start")),
                _time_seconds(scene.get("end")),
            ),
        )

    merged = {"characters": merged_characters, "parts": merged_parts}
    return normalize_three_part_payload(
        json.dumps(merged, ensure_ascii=False), boundaries[-1], target_words
    )


def validate_single_part_response(
    text, part_number, target_words, require_coverage=True,
    required_event_ids=None, require_word_floor=False, required_ending_sec=None,
    require_tts_word_range=False, required_scene_count=None,
):
    """Validate one part by structure and coverage, never by word or scene count."""
    payload = _json_object(text)
    parts = payload.get("parts")
    if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
        raise ValueError(f"Kết quả phần {part_number} phải chứa đúng một phần.")
    part = parts[0]
    scenes = part.get("scenes") if isinstance(part.get("scenes"), list) else []
    valid_ends = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        try:
            valid_ends.append(_time_seconds(scene.get("end")))
        except (TypeError, ValueError):
            pass
    clip_duration = max(valid_ends + [1.0])
    try:
        declared_end = _time_seconds(part.get("video_end"))
    except (TypeError, ValueError):
        declared_end = clip_duration
    if declared_end > clip_duration:
        clip_duration = declared_end
    requirements = single_part_missing_requirements(
        text,
        clip_duration,
        target_words,
        required_event_ids=required_event_ids,
        require_word_floor=require_word_floor,
        required_ending_sec=required_ending_sec,
        require_tts_word_range=require_tts_word_range,
        required_scene_count=required_scene_count,
    )
    if requirements["invalid_timestamp_scenes"]:
        details = ", ".join(
            f"{item['scene_id']} ({'/'.join(item['reasons'])})"
            for item in requirements["invalid_timestamp_scenes"]
        )
        raise ValueError(
            f"Phan {part_number} co timestamp khong hop le: {details}."
        )
    if require_coverage and requirements["coverage_missing"]:
        missing = requirements.get("missing_event_ids") or []
        if missing:
            raise ValueError(
                f"Phần {part_number} còn thiếu sự kiện bắt buộc: {', '.join(missing)}."
            )
        invalid_mappings = requirements.get("invalid_event_mappings") or []
        if invalid_mappings:
            raise ValueError(
                f"Phần {part_number} ánh xạ event/TTS không hợp lệ; "
                "mỗi event phải thuộc đúng một scene và mỗi scene chỉ có một event."
            )
        raise ValueError(f"Phần {part_number} chưa kể đến diễn biến cuối clip.")
    if require_word_floor and requirements["word_deficit"]:
        raise ValueError(
            f"Phần {part_number} còn thiếu {requirements['word_deficit']} từ TTS."
        )
    if require_tts_word_range and requirements["invalid_tts_scenes"]:
        details = []
        for item in requirements["invalid_tts_scenes"]:
            reasons = []
            if item.get("outside_word_range"):
                reasons.append(
                    f"ngoài {item.get('minimum_words', MIN_TTS_WORDS)}-"
                    f"{item.get('maximum_words', MAX_TTS_WORDS)} từ"
                )
            if item.get("fragmented"):
                reasons.append("câu bị vụn")
            if item.get("repeated_previous"):
                reasons.append("lặp mạnh câu trước")
            details.append(
                f"{item['scene_id']}={item['word_count']} từ"
                f" ({', '.join(reasons) or 'chưa đạt chất lượng'})"
            )
        raise ValueError(
            f"Phần {part_number} có TTS chưa đạt yêu cầu: "
            + ", ".join(details) + "."
        )
    if required_scene_count is not None and requirements["scene_count_gap"]:
        raise ValueError(
            f"Phần {part_number} có {requirements['scene_count']} scene; "
            f"cần đúng {int(required_scene_count)} scene/TTS."
        )
    return payload


def _normalize_visual_beats(raw_beats, narration, scene_start, scene_end=None):
    """Keep only contextual beats that can be locked to this exact narration."""
    visual_beats = []
    narration = re.sub(r"\s+", " ", str(narration or "")).strip()
    previous_trigger_end = 0
    previous_beat_start = float(scene_start)
    for raw_beat in list(raw_beats or []):
        if not isinstance(raw_beat, dict):
            continue
        trigger = re.sub(
            r"\s+", " ", str(raw_beat.get("trigger_text", ""))
        ).strip()
        trigger_word_count = len(re.findall(r"\w+", trigger, flags=re.UNICODE))
        max_trigger_words = 12 if raw_beat.get('backend_locked') else 6
        if not trigger or not 2 <= trigger_word_count <= max_trigger_words:
            continue
        matches = list(re.finditer(re.escape(trigger), narration, flags=re.IGNORECASE))
        if len(matches) != 1 or matches[0].start() < previous_trigger_end:
            continue
        # START-END already illustrates the opening action. A beat is only a
        # later visual change inside the same narration, never an opening noun
        # or prop selected merely to add another cut.
        prefix_word_count = len(re.findall(
            r"\w+", narration[:matches[0].start()], flags=re.UNICODE
        ))
        semantic_action = bool(raw_beat.get("semantic_action"))
        if prefix_word_count < MIN_BEAT_PREFIX_WORDS and not semantic_action:
            continue
        prefix = narration[:matches[0].start()].rstrip()
        if not semantic_action and prefix_word_count == MIN_BEAT_PREFIX_WORDS and not re.search(
            r"[,;:.!?…]$", prefix
        ):
            continue
        try:
            beat_start = _time_seconds(raw_beat.get("start"))
            beat_end = _time_seconds(raw_beat.get("end"))
        except ValueError:
            continue
        if beat_end <= beat_start or beat_start <= previous_beat_start:
            continue
        if scene_end is not None and beat_end > float(scene_end) + 0.05:
            continue
        # Three seconds is an editorial target, not a hard validator. Preserve
        # the grounded source range when the real action needs a shorter or
        # longer shot instead of silently changing Gemini's timestamp.
        confidence = raw_beat.get("confidence", 0.0)
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < MIN_VISUAL_BEAT_CONFIDENCE:
            continue
        match_type = str(raw_beat.get("match_type", "literal")).strip().lower()
        if match_type not in {"literal", "reaction", "context"}:
            match_type = "literal"
        visual_beats.append({
            "trigger_text": trigger,
            "start": _format_time(beat_start),
            "end": _format_time(beat_end),
            "match_type": match_type,
            "confidence": confidence,
            "semantic_action": semantic_action,
            **({'backend_locked': True} if raw_beat.get('backend_locked') else {}),
        })
        previous_trigger_end = matches[0].start() + 1 if raw_beat.get('backend_locked') else matches[0].end()
        previous_beat_start = beat_start
    return visual_beats


def normalize_three_part_payload(text, video_duration_sec, target_words):
    """Parse and normalize the pass-1 JSON while preserving model evidence."""
    payload = _json_object(text)
    characters = payload.get("characters")
    parts = payload.get("parts")
    if not isinstance(characters, list):
        characters = []
    if not isinstance(parts, list) or len(parts) != 3:
        raise ValueError("Lượt 1 phải trả đúng ba phần.")

    expected_parts = {1, 2, 3}
    seen_parts = set()
    seen_scene_ids = set()
    normalized_parts = []
    previous_boundary = 0.0
    previous_active_start = -1.0
    previous_active_scene = None
    part_targets = proportional_part_targets(
        [part.get("target_words", 0) if isinstance(part, dict) else 0 for part in parts],
        target_words,
    )
    for raw_part in parts:
        if not isinstance(raw_part, dict):
            raise ValueError("Mỗi phần phải là một object.")
        number = int(raw_part.get("part", 0))
        if number not in expected_parts or number in seen_parts:
            raise ValueError("Số phần bị thiếu hoặc trùng.")
        if number != len(normalized_parts) + 1:
            raise ValueError("Ba phần phải được trả theo thứ tự 1, 2, 3.")
        seen_parts.add(number)
        start = _time_seconds(raw_part.get("video_start"))
        end = _time_seconds(raw_part.get("video_end"))
        if end <= start:
            raise ValueError(f"Phần {number} có ranh giới không hợp lệ.")
        if number == 1 and abs(start) > 2.0:
            raise ValueError("Phần 1 phải bắt đầu từ đầu video.")
        if number > 1 and abs(start - previous_boundary) > 2.0:
            raise ValueError("Ba phần phải nối tiếp nhau, không có khoảng trống.")
        previous_boundary = end

        normalized = {
            "part": number,
            "video_start": _format_time(start),
            "video_end": _format_time(end),
            "target_words": part_targets[number - 1],
            "scenes": [],
            "reserve_scenes": [],
        }
        for key in ("scenes", "reserve_scenes"):
            rows = raw_part.get(key)
            if not isinstance(rows, list):
                rows = []
            for raw_scene in rows:
                if not isinstance(raw_scene, dict):
                    continue
                scene_id = str(raw_scene.get("scene_id", "")).strip().upper()
                if not re.fullmatch(r"S\d{4,}", scene_id):
                    raise ValueError(f"scene_id không hợp lệ: {scene_id or 'rỗng'}")
                if scene_id in seen_scene_ids:
                    raise ValueError(f"scene_id bị trùng: {scene_id}")
                seen_scene_ids.add(scene_id)
                scene_start = _time_seconds(raw_scene.get("start"))
                scene_end = _time_seconds(raw_scene.get("end"))
                if scene_end <= scene_start:
                    raise ValueError(f"{scene_id} có START-END không hợp lệ.")
                if key == "scenes" and scene_start <= previous_active_start:
                    raise ValueError("Các scene_id đang dùng phải tăng theo thời gian.")
                if key == "scenes":
                    if (
                        previous_active_scene is not None
                        and _time_seconds(previous_active_scene["end"]) > scene_start
                    ):
                        previous_active_scene["end"] = _format_time(scene_start)
                    previous_active_start = scene_start
                scene_type = str(raw_scene.get("type", "")).strip().lower()
                if key == "reserve_scenes":
                    scene_type = "reserve"
                elif scene_type not in {"main", "bridge"}:
                    scene_type = "main"
                scene = {
                    "scene_id": scene_id,
                    "start": _format_time(scene_start),
                    "end": _format_time(scene_end),
                    "type": scene_type,
                    "characters": list(raw_scene.get("characters") or []),
                    "event": re.sub(r"\s+", " ", str(raw_scene.get("event", ""))).strip(),
                    "tts": re.sub(r"\s+", " ", str(raw_scene.get("tts", ""))).strip(),
                    "story_role": str(raw_scene.get("story_role", "")).strip().lower(),
                    "status": str(raw_scene.get("status", "ok")).strip().lower(),
                    "errors": list(raw_scene.get("errors") or []),
                    "covered_event_ids": [
                        str(value).strip().upper()
                        for value in (raw_scene.get("covered_event_ids") or [])
                        if str(value).strip()
                    ],
                    "covered_story_anchor_ids": [
                        str(value).strip().upper()
                        for value in (raw_scene.get("covered_story_anchor_ids") or [])
                        if str(value).strip()
                    ],
                    "relation_to_previous": str(
                        raw_scene.get("relation_to_previous", "")
                    ).strip().lower(),
                    "transition_text": re.sub(
                        r"\s+", " ", str(raw_scene.get("transition_text", ""))
                    ).strip(),
                    "state_before": re.sub(
                        r"\s+", " ", str(raw_scene.get("state_before", ""))
                    ).strip(),
                    "state_after": re.sub(
                        r"\s+", " ", str(raw_scene.get("state_after", ""))
                    ).strip(),
                    "narrative_function": str(
                        raw_scene.get("narrative_function", "development")
                    ).strip().lower(),
                    "selection_cluster_id": str(
                        raw_scene.get("selection_cluster_id", "")
                    ).strip(),
                    "selection_importance": str(
                        raw_scene.get("selection_importance", "")
                    ).strip().lower(),
                    "causal_anchor": bool(raw_scene.get("causal_anchor", False)),
                    "event_kind": str(
                        raw_scene.get("event_kind", "")
                    ).strip().lower(),
                    "repeat_group": str(
                        raw_scene.get("repeat_group", "")
                    ).strip(),
                    "story_beat_id": str(
                        raw_scene.get("story_beat_id", "")
                    ).strip().upper(),
                    "story_beat_summary": re.sub(
                        r"\s+", " ", str(raw_scene.get("story_beat_summary", ""))
                    ).strip(),
                    "story_beat_cause": re.sub(
                        r"\s+", " ", str(raw_scene.get("story_beat_cause", ""))
                    ).strip(),
                    "story_beat_outcome": re.sub(
                        r"\s+", " ", str(raw_scene.get("story_beat_outcome", ""))
                    ).strip(),
                    "story_beat_position": raw_scene.get("story_beat_position", 0),
                    "story_beat_size": raw_scene.get("story_beat_size", 0),
                    "climax_spine_role": str(
                        raw_scene.get("climax_spine_role", "")
                    ).strip().lower(),
                    "visual_support_only": bool(
                        raw_scene.get("visual_support_only", False)
                    ),
                    "derived_visual_slot": bool(
                        raw_scene.get("derived_visual_slot", False)
                    ),
                    "derived_narration_slot": bool(
                        raw_scene.get("derived_narration_slot", False)
                    ),
                    "narration_focus": str(
                        raw_scene.get("narration_focus", "")
                    ).strip(),
                    "locked_visual_shots": copy.deepcopy([
                        shot for shot in (raw_scene.get("locked_visual_shots") or [])
                        if isinstance(shot, dict)
                    ]),
                    "visual_links": copy.deepcopy(raw_scene.get("visual_links") or []),
                    "visual_alignment_version": raw_scene.get("visual_alignment_version", 0),
                    "visual_alignment_errors": list(raw_scene.get("visual_alignment_errors") or []),
                }
                if scene["story_role"] not in {
                    "development", "climax", "resolution", "ending"
                }:
                    scene["story_role"] = ""
                # Optional word-locked shot changes inside the same natural
                # narration sentence. Invalid beats are ignored instead of
                # making an otherwise usable analysis fail completely.
                scene["visual_beats"] = _normalize_visual_beats(
                    raw_scene.get("visual_beats"), scene["tts"], scene_start
                )
                normalized[key].append(scene)
                if key == "scenes":
                    previous_active_scene = scene
        normalized_parts.append(normalized)

    normalized_parts.sort(key=lambda item: item["part"])
    # Gemini occasionally puts a valid reserve clip under the neighboring
    # part. Reserve clips are not active yet, so safely re-home them by their
    # midpoint instead of rejecting the complete analysis.
    all_reserves = [
        scene
        for part in normalized_parts
        for scene in part["reserve_scenes"]
    ]
    for part in normalized_parts:
        part["reserve_scenes"] = []
    for scene in all_reserves:
        midpoint = (_time_seconds(scene["start"]) + _time_seconds(scene["end"])) / 2.0
        owner = next(
            (
                part for part in normalized_parts
                if _time_seconds(part["video_start"]) <= midpoint
                <= _time_seconds(part["video_end"])
            ),
            None,
        )
        if owner is not None:
            owner["reserve_scenes"].append(scene)

    for part in normalized_parts:
        part["reserve_scenes"].sort(key=lambda item: _time_seconds(item["start"]))
    if seen_parts != expected_parts:
        raise ValueError("Lượt 1 thiếu một trong ba phần.")
    if abs(previous_boundary - float(video_duration_sec)) > max(5.0, float(video_duration_sec) * 0.03):
        raise ValueError("Phần 3 chưa bao phủ đến cuối video.")
    if not any(part["scenes"] for part in normalized_parts):
        raise ValueError("Lượt 1 không tạo cảnh recap.")
    return {"characters": characters, "parts": normalized_parts}


def inspect_three_part_payload(payload):
    """Return deterministic scene/part defects for the selective second pass."""
    issues = []
    ordered_scene_ids = [
        scene.get("scene_id")
        for part in payload.get("parts", [])
        for scene in part.get("scenes", [])
        if isinstance(scene, dict)
    ]
    terminal_allowed_ids = set(ordered_scene_ids[-3:])
    previous_scene_start = -1.0
    opening_phrases = {}
    phrase_owners = {}
    previous_content_tokens = None
    previous_opening = None
    for part in payload["parts"]:
        for scene in part["scenes"]:
            scene_id = scene["scene_id"]
            start = _time_seconds(scene["start"])
            end = _time_seconds(scene["end"])
            narration = scene["tts"]
            codes = []
            if not narration:
                codes.append("missing_tts")
            if re.search(r"(?<!\w)[CP]\d+(?!\w)", narration, re.IGNORECASE):
                codes.append("internal_character_id")
            if _looks_like_english_prose(narration):
                codes.append("non_vietnamese_tts")
            relation = str(scene.get("relation_to_previous", "")).strip().lower()
            transition = str(scene.get("transition_text", "")).strip()
            if relation and relation not in NARRATIVE_RELATIONS:
                codes.append("invalid_narrative_relation")
            if relation in EXPLICIT_TRANSITION_RELATIONS:
                if not transition:
                    codes.append("missing_transition")
                elif narration.casefold().count(transition.casefold()) != 1:
                    codes.append("transition_not_in_tts")
            if start < previous_scene_start:
                codes.append("timestamp_out_of_order")
            if end <= start:
                codes.append("invalid_timestamp")
            previous_scene_start = max(previous_scene_start, start)
            lowered = narration.lower()
            if any(phrase in lowered for phrase in ABSTRACT_PHRASES):
                codes.append("abstract_wording")
            if (
                scene_id not in terminal_allowed_ids
                and any(phrase in lowered for phrase in TERMINAL_STORY_PHRASES)
            ):
                codes.append("premature_ending")
            opening = " ".join(re.findall(r"\w+", lowered, re.UNICODE)[:4])
            if opening:
                opening_phrases.setdefault(opening, []).append(scene_id)
            tokens = re.findall(r"\w+", lowered, re.UNICODE)
            current_opening = tuple(tokens[:2])
            if len(current_opening) == 2 and current_opening == previous_opening:
                codes.append("repeated_opening")
            previous_opening = current_opening
            content_tokens = [
                token for token in tokens
                if len(token) >= 3 and token not in REPETITION_STOPWORDS
            ]
            token_counts = {}
            for token in content_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            if any(count >= 3 for count in token_counts.values()):
                codes.append("repeated_word")
            current_content_set = set(content_tokens)
            if previous_content_tokens:
                common = current_content_set & previous_content_tokens
                smaller = min(len(current_content_set), len(previous_content_tokens))
                if smaller >= 4 and len(common) >= 3 and len(common) / smaller >= 0.55:
                    codes.append("repeated_vocabulary")
            previous_content_tokens = current_content_set
            for index in range(max(0, len(tokens) - 3)):
                phrase = " ".join(tokens[index:index + 4])
                phrase_owners.setdefault(phrase, []).append(scene_id)
            if scene.get("status") == "error":
                codes.extend(str(item).strip() for item in scene.get("errors", []) if str(item).strip())
                if not scene.get("errors"):
                    codes.append("model_uncertain")
            for code in dict.fromkeys(codes):
                issues.append({"scope": "scene", "target_id": scene_id, "code": code})

    for scene_ids in opening_phrases.values():
        if len(scene_ids) > 2:
            for scene_id in scene_ids[2:]:
                issues.append({"scope": "scene", "target_id": scene_id, "code": "repeated_opening"})
    repeated_ids = set()
    for owners in phrase_owners.values():
        unique_owners = list(dict.fromkeys(owners))
        if len(unique_owners) > 2:
            repeated_ids.update(unique_owners[2:])
    for scene_id in sorted(repeated_ids):
        issues.append({"scope": "scene", "target_id": scene_id, "code": "repeated_phrase"})

    # Ending is semantic, not proximity to the physical EOF. Credits, outro,
    # silence, or unused footage after the final story event must not create a
    # false missing-ending issue.
    final_part = payload["parts"][-1]
    final_scenes = final_part.get("scenes") or []
    if final_scenes:
        video_end = _time_seconds(final_part["video_end"])
        if video_end >= LONG_STORY_SECONDS:
            has_ending = any(scene.get("story_role") == "ending" for scene in final_scenes)
            if not has_ending:
                issues.append({
                    "scope": "part",
                    "target_id": f"PART_{final_part['part']}",
                    "code": "story_role_ending_missing",
                    "has_ending": has_ending,
                })
    return issues


def payload_to_timeline(payload):
    rows = []
    for part in payload["parts"]:
        for scene in part["scenes"]:
            if scene.get('visual_selection_pending'):
                raise ValueError('SENTENCE_SHOTS_PENDING: ' + str(scene.get('scene_id', '')))
            if scene.get("visual_alignment_version") == 1:
                links = scene.get("visual_links") or []
                if (scene.get("visual_alignment_errors")
                        or len(scene.get("visual_beats") or []) != max(0, len(links) - 1)):
                    raise ValueError(
                        f"VISUAL_ALIGNMENT_MISSING: {scene.get('scene_id', '')}; "
                        "can sua lien ket y ke-hinh cua dung cau nay."
                    )
            narration = str(scene.get("tts", "")).strip()
            if narration:
                # Encode word-locked changes as inline BEAT markers. The TTS
                # parser removes them from speech and later resolves the exact
                # cut from Edge word-boundary timing.
                insertions = []
                for beat in scene.get("visual_beats", []):
                    trigger = str(beat.get("trigger_text", "")).strip()
                    matches = list(re.finditer(re.escape(trigger), narration, flags=re.IGNORECASE)) if trigger else []
                    if len(matches) == 1:
                        marker = f"[BEAT={beat['start']}-{beat['end']}] "
                        insertions.append((matches[0].start(), marker))
                for position, marker in sorted(insertions, reverse=True):
                    narration = narration[:position] + marker + narration[position:]
                rows.append(f"[TIME={scene['start']}-{scene['end']}] {narration}")
    if not rows:
        raise ValueError("Không có câu TTS hợp lệ để dựng video.")
    return "\n".join(rows)


def inspect_narration_quality(payload, maximum=40):
    """Find local repetition and padded prose without another model call."""
    issues = []
    recent = []
    limit = max(1, int(maximum))
    character_tokens = {
        token
        for character in (payload.get("characters") or [])
        if isinstance(character, dict)
        for label in [character.get("name"), *(character.get("aliases") or [])]
        for token in re.findall(
            r"[^\W_]{2,}", str(label or "").casefold(), flags=re.UNICODE
        )
    }

    def content_tokens(value):
        return {
            token for token in re.findall(
                r"[^\W_]{2,}", str(value or "").casefold(), flags=re.UNICODE
            ) if token not in REPETITION_STOPWORDS and token not in character_tokens
        }

    def focus_key(scene):
        return _normalized_fact_key(scene.get("narration_focus", ""))

    for part in payload.get("parts") or []:
        for scene in part.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id", "")).strip().upper()
            narration = re.sub(r"\s+", " ", str(scene.get("tts", ""))).strip()
            if not scene_id or not narration:
                continue
            lowered = narration.casefold()
            padded = [
                phrase for phrase in NARRATION_PADDING_PHRASES
                if phrase.casefold() in lowered
            ]
            if padded:
                issues.append({
                    "code": "padded_ai_prose",
                    "scene_id": scene_id,
                    "message": (
                        "Rút tính từ/cụm đệm, giữ sự kiện cụ thể: "
                        + ", ".join(padded)
                    ),
                })
            tokens = content_tokens(narration)
            # Look far enough back to catch a command, goal, or outcome that
            # gets paraphrased again later in the same part. Three sentences
            # was too short for common recap repetition loops.
            for previous in reversed(recent[-20:]):
                previous_tokens = previous["tokens"]
                smaller = min(len(tokens), len(previous_tokens))
                overlap = len(tokens & previous_tokens) / smaller if smaller else 0.0
                same_beat = bool(
                    scene.get("story_beat_id")
                    and scene.get("story_beat_id") == previous["story_beat_id"]
                )
                current_focus = focus_key(scene)
                previous_focus = previous["focus_key"]
                current_focus_tokens = content_tokens(current_focus)
                previous_focus_tokens = content_tokens(previous_focus)
                focus_smaller = min(
                    len(current_focus_tokens), len(previous_focus_tokens)
                )
                focus_overlap = (
                    len(current_focus_tokens & previous_focus_tokens) / focus_smaller
                    if focus_smaller else 0.0
                )
                distinct_locked_foci = bool(
                    current_focus and previous_focus
                    and current_focus != previous_focus and focus_overlap < 0.60
                )
                same_fact = bool(
                    scene.get("fact_id") and previous.get("fact_id")
                    and str(scene.get("fact_id")).casefold()
                    == str(previous.get("fact_id")).casefold()
                )
                repeated = (
                    same_fact
                    or overlap >= 0.72
                    or (
                        same_beat and not distinct_locked_foci
                        and overlap >= 0.58
                    )
                    or (
                        current_focus and previous_focus
                        and current_focus == previous_focus and overlap >= 0.45
                    )
                )
                if smaller >= 5 and repeated:
                    issues.append({
                        "code": "repeated_narration_idea",
                        "scene_id": scene_id,
                        "related_scene_id": previous["scene_id"],
                        "overlap": round(overlap, 3),
                        "narration_focus": str(
                            scene.get("narration_focus", "")
                        ).strip(),
                        "message": (
                            "Câu này gần lặp ý câu trước; chỉ giữ thông tin mới "
                            "về nguyên nhân, hành động hoặc hậu quả."
                        ),
                    })
                    break
            recent.append({
                "scene_id": scene_id,
                "story_beat_id": scene.get("story_beat_id"),
                "focus_key": focus_key(scene),
                "fact_id": scene.get("fact_id"),
                "tokens": tokens,
            })
            if len(issues) >= limit:
                return issues[:limit]
    return issues[:limit]


def narration_candidate_rank(requirements, repeated_issue_count):
    """Rank writer candidates: hard safety, repetition, then soft distance."""
    result = requirements if isinstance(requirements, dict) else {}
    return (
        1 if single_part_has_hard_failures(result) else 0,
        len(result.get("invalid_timestamp_scenes") or []),
        len(result.get("invalid_character_mentions") or []),
        max(0, int(repeated_issue_count)),
        float(result.get("distance", float("inf"))),
    )


def narration_scene_issue_rank(requirements, scene_id):
    """Rank one TTS row so a retry can never replace it with a worse version."""
    result = requirements if isinstance(requirements, dict) else {}
    wanted = str(scene_id or "").strip().upper()
    character_error = any(
        str(item.get("scene_id", "")).strip().upper() == wanted
        for item in result.get("invalid_character_mentions") or []
    )
    if any(str(item.get("scene_id", "")).strip().upper() == wanted
           for item in result.get("invalid_timestamp_scenes") or []):
        return (2, 1, 0, 0)
    issue = next((
        item for item in result.get("invalid_tts_scenes") or []
        if str(item.get("scene_id", "")).strip().upper() == wanted
    ), None)
    if issue is None:
        return (int(character_error), 0, 0, 0)
    language_error = bool(issue.get("language_issues"))
    minimum = int(issue.get("minimum_words", MIN_TTS_WORDS))
    maximum = int(issue.get("maximum_words", MAX_TTS_WORDS))
    words = max(0, int(issue.get("word_count", 0)))
    distance = max(minimum - words, words - maximum, 0)
    secondary = int(bool(
        issue.get("fragmented") or issue.get("repeated_previous")
    ))
    return (
        int(character_error or language_error),
        1,
        distance,
        secondary,
    )


def build_targeted_narration_repair_prompt(
    payload, requirements, locked_event_by_id,
):
    """Request only missing or defective TTS rows from the locked event map."""
    source = payload if isinstance(payload, dict) else _json_object(payload)
    locked = {
        str(event_id).strip().upper(): event
        for event_id, event in (locked_event_by_id or {}).items()
        if str(event_id).strip() and isinstance(event, dict)
    }
    scene_by_id = {}
    event_by_scene_id = {}
    for part in source.get("parts") or []:
        for scene in part.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id", "")).strip().upper()
            mapped = scene.get("covered_event_ids") or []
            if not isinstance(mapped, list):
                mapped = [mapped]
            event_id = next((
                str(value).strip().upper() for value in mapped
                if str(value).strip().upper() in locked
            ), "")
            if scene_id:
                scene_by_id[scene_id] = scene
                if event_id:
                    event_by_scene_id[scene_id] = event_id

    reasons = {}
    for item in (requirements or {}).get("invalid_timestamp_scenes") or []:
        scene_id = str(item.get("scene_id", "")).strip().upper()
        if scene_id:
            reasons.setdefault(scene_id, []).append({"visual_alignment": item.get("reasons")})
    for item in (requirements or {}).get("invalid_tts_scenes") or []:
        scene_id = str(item.get("scene_id", "")).strip().upper()
        if scene_id:
            reasons.setdefault(scene_id, []).append({
                key: copy.deepcopy(value)
                for key, value in item.items() if key != "scene_id"
            })
    for item in (requirements or {}).get("invalid_character_mentions") or []:
        scene_id = str(item.get("scene_id", "")).strip().upper()
        if scene_id:
            reasons.setdefault(scene_id, []).append({
                "unexpected_names": list(item.get("unexpected_names") or []),
                "allowed_names": list(item.get("allowed_names") or []),
            })

    requested_event_ids = []
    for scene_id in reasons:
        event_id = event_by_scene_id.get(scene_id)
        if event_id and event_id not in requested_event_ids:
            requested_event_ids.append(event_id)
    for event_id in (requirements or {}).get("missing_event_ids") or []:
        normalized = str(event_id).strip().upper()
        if normalized in locked and normalized not in requested_event_ids:
            requested_event_ids.append(normalized)

    rows = []
    for event_id in requested_event_ids:
        event = locked[event_id]
        scene_id = str(event.get("scene_id", "")).strip().upper()
        existing = scene_by_id.get(scene_id, {})
        rows.append({
            "scene_id": scene_id,
            "event_id": event_id,
            "current_tts": re.sub(
                r"\s+", " ", str(existing.get("tts", ""))
            ).strip(),
            "event": str(event.get("event", "")).strip(),
            "visuals": visual_catalog(event.get("visual_sources") or []),
            "situation_narration": bool(event.get('situation_narration')),
            "sentence_quota": event.get('sentence_quota'),
            "narration_focus": str(event.get("narration_focus", "")).strip(),
            "observed_facts": copy.deepcopy(event.get("observed_facts") or []),
            "allowed_character_names": list(
                event.get("allowed_character_names") or []
            ),
            "minimum_words": int(event.get("tts_min_words", MIN_TTS_WORDS)),
            "maximum_words": int(event.get("tts_max_words", MAX_TTS_WORDS)),
            "reason": reasons.get(scene_id, [{"missing_sentence": True}]),
        })
    if not rows:
        raise ValueError("TARGETED_REPAIR_EMPTY: khong co cau cu the de sua.")
    return f"""Ban la bien tap vien recap phim tieng Viet. Chi viet lai cac cau co trong danh sach.
- Tra dung va du {len(rows)} event_id; khong tra bat ky event nao khac.
- Giu nguyen scene_id va event_id. Tra `tts` va `visual_links`.
- visual_links lien ket y ke voi visual_id trong visuals. Phan tu dau co trigger_text="";
  cac phan tu sau dung cum 2-6 tu xuat hien dung mot lan trong TTS, ngay dau y doi hanh dong.
- Khong tu tao timestamp, khong chon canh theo ti le so tu. Chi dung ID hinh da cho.
- Neu chi loi lien ket hinh, giu current_tts va chi sua visual_links cho dung.
- TTS la mot cau tieng Viet co dau, tu nhien. minimum_words/maximum_words chi la tham khao
  nhip doc; khong nhoi hoac cat y de chay theo cac con so nay.
- Neu situation_narration=true, GIU TTS la doan ke 1-4 cau lien mach, khong be thanh mo ta shot;
  chi can cac hinh dai dien trong nhom theo thu tu, khong bat buoc dung het visuals.
- Neu co sentence_quota, bat buoc giu DUNG so cau do (ghi de gioi han 1-4);
  khong them/bot cau khi sua hinh hay chi tiet. Moi cau van phai them du kien moi.
- Giu dung su kien; khong them hanh dong, loi thoai hoac ten nhan vat moi.
- Neu co observed_facts, chi dien dat cac du kien do. Loi unsupported_narration
  bat buoc sua TTS, khong chi visual_links. Voi o cu khong phai situation_narration,
  dung TAT CA visuals da khoa theo thu tu.
- Khi sua unsupported_narration, bo dung chi tiet bi bac bo; giu hanh dong da duoc
  observed_facts chung minh. Khong them tinh tu cam giac (mat lanh), pham chat
  (dung cam), thai do (quyet doan) neu facts khong neu. Khong them y de du so tu.
- Chi goi ten trong allowed_character_names. Neu chua chac ten, dung vai tro trung tinh.
- Khong viet timestamp, ghi chu, ma scene hay loi giai thich trong TTS.

CAC CAU CAN VIET LAI:
{json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}

Chi tra JSON: {{"scenes":[{{"scene_id":"S0001","event_id":"EV001","tts":"...","visual_links":[{{"visual_id":"V001","trigger_text":""}}]}}]}}
"""


def apply_targeted_narration_repairs(
    payload, text, requested_event_ids, locked_event_by_id,
):
    """Merge exact repaired rows; missing scenes are restored from locked events."""
    source = copy.deepcopy(
        payload if isinstance(payload, dict) else _json_object(payload)
    )
    raw_source = str(text or "").lstrip("\ufeff").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_source, re.I)
    raw_source = fenced.group(1).strip() if fenced else raw_source
    candidate = None
    try:
        decoded = json.loads(raw_source, strict=False)
        if isinstance(decoded, list):
            candidate = {"scenes": decoded}
        elif isinstance(decoded, dict):
            candidate = decoded
    except (json.JSONDecodeError, TypeError):
        pass
    if candidate is None:
        candidate = _json_object(text)
    rows = candidate.get("scenes")
    if not isinstance(rows, list):
        raise ValueError("TARGETED_REPAIR_SCHEMA_INVALID: thieu scenes.")
    expected = {
        str(value).strip().upper() for value in (requested_event_ids or [])
        if str(value).strip()
    }
    locked = {
        str(event_id).strip().upper(): event
        for event_id, event in (locked_event_by_id or {}).items()
        if str(event_id).strip() and isinstance(event, dict)
    }
    replacements = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        event_id = str(row.get("event_id", "")).strip().upper()
        tts = re.sub(r"\s+", " ", str(row.get("tts", ""))).strip()
        if event_id in replacements:
            raise ValueError(f"TARGETED_REPAIR_DUPLICATE: {event_id}.")
        if event_id and tts:
            if locked.get(event_id, {}).get('sentence_quota') and not locked.get(event_id, {}).get('narration_unit_version'):
                from narrative_contract import constrain_repair_sentences
                tts = constrain_repair_sentences(tts, locked[event_id]['sentence_quota'])
            replacements[event_id] = {"tts": tts, "visual_links": copy.deepcopy(row.get("visual_links") or [])}
            for field in ('source_grounded_writer_version', 'source_grounded_signature', 'sentence_proofs', 'sentence_slots', 'evidence_sentence_plan_version', 'paragraph_plan_version', 'narration_unit_version'):
                if field in row:
                    replacements[event_id][field] = copy.deepcopy(row[field])
    if set(replacements) != expected:
        raise ValueError(
            "TARGETED_REPAIR_IDS_INVALID: thieu "
            + ",".join(sorted(expected - set(replacements)))
            + "; thua " + ",".join(sorted(set(replacements) - expected))
        )
    parts = source.get("parts") or []
    if len(parts) != 1 or not isinstance(parts[0], dict):
        raise ValueError("TARGETED_REPAIR_SOURCE_INVALID: phai co dung mot part.")
    scenes = [
        scene for scene in (parts[0].get("scenes") or [])
        if isinstance(scene, dict)
    ]
    scene_by_event = {}
    for scene in scenes:
        mapped = scene.get("covered_event_ids") or []
        if not isinstance(mapped, list):
            mapped = [mapped]
        for value in mapped:
            event_id = str(value).strip().upper()
            if event_id in locked:
                scene_by_event[event_id] = scene
    for event_id, replacement in replacements.items():
        scene = scene_by_event.get(event_id)
        if scene is None:
            event = locked.get(event_id)
            if event is None:
                raise ValueError(f"TARGETED_REPAIR_UNKNOWN_EVENT: {event_id}.")
            scene = {
                "scene_id": str(event.get("scene_id", "")).strip().upper(),
                "covered_event_ids": [event_id],
            }
            scenes.append(scene)
            scene_by_event[event_id] = scene
        scene["tts"] = replacement["tts"]
        scene["visual_links"] = replacement["visual_links"]
        for field in ('source_grounded_writer_version', 'source_grounded_signature', 'sentence_proofs', 'sentence_slots', 'evidence_sentence_plan_version', 'paragraph_plan_version'):
            if field in replacement:
                scene[field] = copy.deepcopy(replacement[field])
        scene["visual_beats"] = []
    order = {
        event_id: index for index, event_id in enumerate(locked.keys())
    }
    scenes.sort(key=lambda scene: min((
        order.get(str(value).strip().upper(), len(order))
        for value in (
            scene.get("covered_event_ids")
            if isinstance(scene.get("covered_event_ids"), list)
            else [scene.get("covered_event_ids")]
        )
    ), default=len(order)))
    parts[0]["scenes"] = scenes
    return source


def build_narration_repetition_feedback(payload, issues, maximum=12):
    """Build precise retry feedback without spending another model request."""
    scene_by_id = {
        str(scene.get("scene_id", "")).strip().upper(): scene
        for part in (payload.get("parts") or []) if isinstance(part, dict)
        for scene in (part.get("scenes") or []) if isinstance(scene, dict)
    }
    rows = []
    for issue in list(issues or [])[:max(1, int(maximum))]:
        if issue.get("code") != "repeated_narration_idea":
            continue
        scene_id = str(issue.get("scene_id", "")).strip().upper()
        related_id = str(issue.get("related_scene_id", "")).strip().upper()
        scene = scene_by_id.get(scene_id, {})
        related = scene_by_id.get(related_id, {})
        focus = str(scene.get("narration_focus", "")).strip()
        current_tts = re.sub(r"\s+", " ", str(scene.get("tts", ""))).strip()
        related_tts = re.sub(r"\s+", " ", str(related.get("tts", ""))).strip()
        instruction = (
            f"{scene_id} lặp {related_id}; không kể lại ý của {related_id}. "
            + (f"Chỉ phát triển fact mới: {focus}. " if focus else "")
            + f"Câu lặp: {current_tts[:180]}. "
            + f"Câu đã kể: {related_tts[:180]}."
        )
        rows.append(instruction)
    if not rows:
        return ""
    return (
        "\nLOI LAP Y CAN SUA CHINH XAC:\n- "
        + "\n- ".join(rows)
        + "\nGiữ nguyên scene_id/event/timestamp; mỗi câu phải thêm một "
          "nguyên nhân, hành động, thay đổi hoặc hậu quả chưa được kể."
    )


def build_continuous_recap_draft_prompt(events, context):
    """Write situation-level prose before assigning words to locked visuals."""
    cards = [{
        'event_id': e['event_id'],
        'focus': e.get('narration_focus') or e.get('event'),
        'observed_facts': e.get('observed_facts') or [],
        'allowed_character_names': e.get('allowed_character_names') or [],
        'situation': e.get('story_beat_id') or e.get('selection_cluster_id'),
    } for e in events]
    return (
        'CHI DIEN DAT BANG CHUNG DA KHOA. BUOC 1: VIET DOAN TRUYEN LIEN MACH, CHUA CHIA O HINH. '
        'Doc toan bo du kien truoc khi viet. Gom cac hanh dong lien quan thanh cum tinh huong: '
        'boi canh/muc tieu, tro ngai, hanh dong, ket qua. Moi cum viet thanh mot doan recap tu nhien. '
        'Khong viet mot cau mo ta cho tung shot; khong liet ke moi cu chi, di chuyen hay canh phong canh. '
        'Gom thao tac vuon vat thanh hanh dong co y nghia, giu dien bien chinh va thu tu su kien. '
        'Chi noi nguyen nhan, dong co, ket qua hoac noi dung hoi thoai khi bang chung neu ro; '
        'neu chua ro thi chuyen canh trung tinh, khong bia quan he vi-the, dung-luc hay bat-ngo. '
        'Khong viet cau rong nhu "tiep tuc tro chuyen", "bat dau cau chuyen"; '
        'chi ke noi dung trao doi neu da duoc xac nhan. Khong lap ten nhan vat o dau moi cau. '
        'Chuyen tuyen thoi gian/khong gian can cau dan ngan dua tren ngu canh da xac nhan. '
        'Khong ke logo, opening, nhac mo dau, montage gioi thieu nhan vat, credits hay outro. '
        'Du kien va ten rieng phai thuoc cac the duoi day; ngu canh chi de noi mach, khong them su kien. '
        'Van ban se duoc phan ve cac o hinh CO DINH o buoc sau; giu thu tu va du y chinh cua moi the, '
        'nhung KHONG ep mot the thanh mot cau doc lap. '
        '\nPHONG CACH: ' + str(context.get('style') or '')
        + '\nYEU CAU: ' + str(context.get('custom_prompt') or '')
        + '\nMACH TRUYEN: ' + json.dumps(context.get('all_part_summaries') or [], ensure_ascii=False)
        + '\nDOAN TRUOC: ' + json.dumps(context.get('previous_part_tail') or [], ensure_ascii=False)
        + '\nSU KIEN TIEP THEO (chi de chuyen mach, chua ke): '
        + json.dumps(context.get('next_batch_context') or [], ensure_ascii=False)
        + '\nDU KIEN THEO THU TU: ' + json.dumps(cards, ensure_ascii=False)
        + '\nChi tra JSON {"paragraphs":["Doan recap lien mach..."]}.'
    )


def parse_continuous_recap_draft(text):
    paragraphs = _json_object(text).get('paragraphs')
    if (not isinstance(paragraphs, list) or not paragraphs
            or any(not isinstance(p, str) or not p.strip() for p in paragraphs)):
        raise ValueError('CONTINUOUS_RECAP_DRAFT_INVALID')
    return '\n\n'.join(p.strip() for p in paragraphs)


def build_final_narration_smoothing_prompt(payload):
    """Build one text-only continuity pass with an immutable scene contract."""
    rows = []
    for part in payload.get("parts") or []:
        for scene in part.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            rows.append({
                "scene_id": str(scene.get("scene_id", "")).strip().upper(),
                "event": str(scene.get("event", "")).strip(),
                "narration_focus": str(scene.get("narration_focus", "")).strip(),
                "tts": str(scene.get("tts", "")).strip(),
                "allowed_character_names": list(
                    scene.get("allowed_character_names") or []
                ),
            })
    return f"""Ban la bien tap vien lam muot loi recap. Khong co video trong luot nay.
Chi sua cach noi cua `tts`; scene_id, thu tu, su kien, nhan vat, y nghia va timestamp da bi khoa.
- Moi scene phai tra dung mot dong, dung scene_id va dung thu tu dau vao.
- Moi TTS la mot cau tieng Viet tu nhien; do dai linh hoat theo y, khong ep so tu co dinh.
- Cau sau phai noi voi cau truoc bang quan he nguyen nhan, ket qua, doi lap, tiep dien hoac chuyen canh.
- Khong mo lai chu ngu/cam xuc cua cau truoc neu khong can; khong lap y bang tu dong nghia.
- Khong gop hai event, khong bo fact, khong ke truoc event sau va khong them loi thoai.
- Chi goi ten trong allowed_character_names; neu danh sach rong thi khong tu them ten rieng.

CAC CAU DA KHOA:
{json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}

Chi tra JSON: {{"scenes":[{{"scene_id":"S0001","tts":"..."}}]}}
"""


def apply_final_narration_smoothing(payload, text):
    """Apply smoothing only when every immutable scene ID is returned once."""
    candidate = _json_object(text)
    rows = candidate.get("scenes")
    if not isinstance(rows, list):
        raise ValueError("SMOOTHING_SCHEMA_INVALID: thieu scenes.")
    source_scenes = [
        scene for part in (payload.get("parts") or [])
        for scene in (part.get("scenes") or []) if isinstance(scene, dict)
    ]
    expected_ids = [
        str(scene.get("scene_id", "")).strip().upper()
        for scene in source_scenes
    ]
    returned = {}
    returned_order = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        scene_id = str(row.get("scene_id", "")).strip().upper()
        tts = re.sub(r"\s+", " ", str(row.get("tts", ""))).strip()
        if not scene_id or scene_id in returned or not tts:
            raise ValueError("SMOOTHING_SCHEMA_INVALID: ID trung/rong hoac thieu TTS.")
        word_count = len(re.findall(r"\w+", tts, re.UNICODE))
        if not MIN_TTS_WORDS <= word_count <= MAX_TTS_WORDS:
            raise ValueError("SMOOTHING_TTS_RANGE_INVALID")
        returned[scene_id] = tts
        returned_order.append(scene_id)
    if returned_order != expected_ids:
        raise ValueError("SMOOTHING_SCENE_CONTRACT_CHANGED")
    result = copy.deepcopy(payload)
    for part in result.get("parts") or []:
        for scene in part.get("scenes") or []:
            scene_id = str(scene.get("scene_id", "")).strip().upper()
            scene["tts"] = returned[scene_id]
    return result


def _narration_text_normalizer(characters):
    """Build the same text substitutions for proof text and edit cues."""
    display_labels = _character_display_labels(characters or [])
    alias_replacements = []
    for character in characters or []:
        if not isinstance(character, dict):
            continue
        canonical_name = re.sub(
            r"\s+", " ", str(character.get("name", ""))
        ).strip()
        if not canonical_name:
            continue
        aliases = character.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = [aliases]
        else:
            aliases = list(aliases)
        aliases.extend((
            character.get("proposed_name", ""),
            character.get("descriptive_alias", ""),
        ))
        for alias in aliases:
            alias = re.sub(r"\s+", " ", str(alias or "")).strip()
            if (
                len(alias) >= 3
                and alias.casefold() != canonical_name.casefold()
                and not _looks_like_english_prose(alias)
            ):
                alias_replacements.append((alias, canonical_name))
    alias_replacements.sort(key=lambda item: len(item[0]), reverse=True)
    concise_phrases = {
        "quyết tâm cao độ": "quyết tâm",
        "vô cùng nguy hiểm": "nguy hiểm",
        "tàn ác khủng khiếp": "tàn ác",
        "niềm vui sướng tột độ": "vui mừng",
        "an toàn tuyệt đối": "an toàn",
        "đầy rẫy hiểm nguy": "nguy hiểm",
        "sức mạnh tà ác khủng khiếp": "sức mạnh áp đảo",
    }
    def normalize(text):
        text = _replace_internal_character_ids(text, display_labels)
        from character_identity import safe_name_mentions
        text = safe_name_mentions(text, dict(alias_replacements))
        for padded, concise in concise_phrases.items():
            text = re.sub(re.escape(padded), concise, text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip()
    return normalize


def clamp_payload_narrations(payload):
    """Normalize unsealed narration; preserve validated source writer locks."""
    result = copy.deepcopy(payload)
    display_labels = _character_display_labels(result.get("characters") or [])
    normalize = _narration_text_normalizer(result.get("characters") or [])
    scenes = []
    for part in result["parts"]:
        for scene in part["scenes"]:
            if scene.get('source_grounded_writer_version'):
                from source_grounded_writer import is_source_grounded
                if not is_source_grounded(scene):
                    raise ValueError('SOURCE_WRITER_LOCK_CHANGED: loi/hinh da doi sau khi chot')
                scenes.append(scene)
                continue
            narration = normalize(scene.get("tts", ""))
            if _looks_like_english_prose(narration):
                grounded_rows = [
                    row for row in (scene.get("covered_events") or [])
                    if isinstance(row, dict)
                ]
                event_text = " ".join(
                    str(row.get("consequence") or row.get("change") or row.get("event") or "").strip()
                    for row in grounded_rows
                    if str(row.get("consequence") or row.get("change") or row.get("event") or "").strip()
                ) or _replace_internal_character_ids(
                    scene.get("event", ""), display_labels
                )
                if event_text and not _looks_like_english_prose(event_text):
                    words = re.sub(r"\s+", " ", event_text).strip(" .").split()
                    narration = " ".join(words[:25]).rstrip(" ,;:") + "."
                else:
                    narration = (
                        "Không đủ bằng chứng tiếng Việt cụ thể để mô tả chính xác cảnh này."
                    )
            if scene.get('sentence_quota') and not scene.get('narration_unit_version'):
                from narrative_contract import constrain_repair_sentences
                narration = constrain_repair_sentences(narration, scene['sentence_quota'])
            scene["tts"] = narration
            # Canonicalizing character names or tightening prose must update
            # the same literal anchors, otherwise final normalization drops
            # a correctly mapped action solely because its words changed.
            for beat in list(scene.get("visual_beats") or []) + list(scene.get("visual_links") or []):
                if not isinstance(beat, dict):
                    continue
                beat["trigger_text"] = normalize(beat.get("trigger_text", ""))
            scenes.append(scene)

    for scene in scenes:
        if scene.get("source_grounded_writer_version"):
            continue
        scene["visual_beats"] = _normalize_visual_beats(
            scene.get("visual_beats"),
            scene["tts"],
            _time_seconds(scene["start"]),
            # BEAT ranges are independent, non-contiguous source shots. The
            # scene END belongs only to the opening TIME shot, so using it as
            # an upper bound erased every later verified shot before render.
            None,
        )

    # The renderer consumes rows in narration order.  A BEAT attached to one
    # row must therefore finish before the next row's primary TIME begins.
    # Flexible semantic beats can span several sentence bands; keeping the
    # whole beat on the first sentence duplicated footage already assigned to
    # the next sentence and made the render validator reject the timeline.
    primary_ranges = [
        (
            _time_seconds(scene.get("start")),
            _time_seconds(scene.get("end")),
        )
        for scene in scenes
    ]
    for scene_index, scene in enumerate(scenes):
        if scene.get("source_grounded_writer_version"):
            continue
        primary_start, primary_end = primary_ranges[scene_index]
        next_start = (
            primary_ranges[scene_index + 1][0]
            if scene_index + 1 < len(primary_ranges) else float("inf")
        )
        cursor_end = primary_end
        kept_beats = []
        for beat in scene.get("visual_beats") or []:
            beat_start = _time_seconds(beat.get("start"))
            beat_end = _time_seconds(beat.get("end"))
            if beat_start < cursor_end - 0.05:
                continue
            if beat_end > next_start + 0.05:
                continue
            kept_beats.append(beat)
            cursor_end = beat_end
        scene["visual_beats"] = kept_beats

        # Keep the internal EDL consistent with the public TIME/BEAT markers.
        allowed_ranges = {
            (round(primary_start, 2), round(primary_end, 2)),
            *((
                round(_time_seconds(beat.get("start")), 2),
                round(_time_seconds(beat.get("end")), 2),
            ) for beat in kept_beats),
        }
        if isinstance(scene.get("locked_visual_shots"), list):
            scene["locked_visual_shots"] = [
                shot for shot in scene["locked_visual_shots"]
                if isinstance(shot, dict) and (
                    round(_time_seconds(shot.get("start")), 2),
                    round(_time_seconds(shot.get("end")), 2),
                ) in allowed_ranges
            ]
    return result
