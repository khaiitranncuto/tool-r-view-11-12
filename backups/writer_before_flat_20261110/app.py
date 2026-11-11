import os
import uuid
import copy
import threading
import shutil
import subprocess
import json
import time
import re
import traceback
import faulthandler
import sys
import math
import tempfile
import unicodedata
from recap_budget import DEFAULT_VOICE_RATE, normalize_voice_rate, narration_budget
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template, send_from_directory
from google import genai
import asyncio
import edge_tts
from tts_engine import (
    extract_time_range,
    format_timeline_rows,
    parse_sentences_and_visual_beats,
    remove_non_story_timeline_rows,
    split_text_into_sentences,
)
from video_renderer import render_review_video_ffmpeg, export_capcut_package, generate_all_images, render_ai_image_video_ffmpeg
from task_store import (
    load_task_state,
    save_task_payload,
    save_task_state,
)
from event_storyboard import (
    clamp_final_timeline_end,
    drop_beats_overlapping_next_time,
    repair_nonpositive_timeline_ranges,
    validate_render_visual_grounding,
    validate_timeline_ranges,
)
from recap_pipeline import (
    build_full_video_outline_prompt,
    build_evidence_review_prompt,
    build_evidence_replenishment_prompt,
    build_identity_comparison_prompt,
    build_chunk_evidence_prompt,
    build_part_story_context,
    build_single_part_video_prompt,
    build_story_plan_from_evidence_prompt,
    build_story_evidence_digest_prompt,
    build_story_plan_from_digests_prompt,
    build_story_gap_supplement_prompt,
    build_terminal_anchor_review_prompt,
    build_targeted_visual_shot_repair_prompt,
    clamp_payload_narrations,
    coerce_single_part_response,
    compact_story_evidence,
    expanded_word_target,
    extract_complete_recap_scenes,
    inspect_three_part_payload,
    inspect_narration_quality,
    build_narration_repetition_feedback,
    build_targeted_narration_repair_prompt,
    apply_targeted_narration_repairs,
    narration_scene_issue_rank,
    merge_single_part_responses,
    merge_story_plan_supplement,
    apply_terminal_anchor_review,
    apply_targeted_visual_shot_repairs,
    fallback_terminal_editorial_anchors,
    normalize_global_evidence_audit,
    normalize_full_video_outline,
    add_objective_evidence_gap_windows,
    normalize_chunk_evidence,
    recover_chunk_evidence,
    merge_evidence_replenishment,
    assess_story_evidence_coverage,
    deduplicate_overlapping_chunk_evidence,
    deduplicate_semantic_chunk_evidence,
    drop_unrepairable_visual_events,
    stitch_cross_chunk_event_continuations,
    terminal_anchor_review_candidates,
    ensure_grounded_terminal_ending,
    apply_verified_evidence_results,
    normalize_story_plan_from_evidence,
    parse_json_object,
    payload_to_timeline,
    plan_adaptive_story_boundaries,
    plan_evidence_replenishment,
    plan_targeted_evidence_gap_windows,
    proportional_part_targets,
    requested_word_range,
    required_tts_scene_count,
    single_part_missing_requirements,
    single_part_has_hard_failures,
    narration_candidate_rank,
    story_outline_for_window,
    split_compact_story_evidence,
    three_part_targets,
    validate_single_part_response,
    vietnamese_narration_issues,
)
from transcript_pipeline import (
    attach_transcript_to_evidence,
    format_transcript_window,
    merge_preferred_subtitles,
    parse_subtitle_text,
    transcribe_video,
)

# =====================================================================
# Cấu hình Môi trường
# =====================================================================
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(WORKSPACE_DIR, "app_settings.json")
LOG_DIR = os.path.join(WORKSPACE_DIR, "logs")
TRANSCRIPT_CACHE_DIR = os.path.join(WORKSPACE_DIR, ".cache", "transcripts")
GEMINI_MODEL_CACHE_PATH = os.path.join(WORKSPACE_DIR, ".cache", "gemini_models.json")
DEFAULT_GEMINI_MODEL_ROWS = [
    {"id": model_id, "selectable": selectable}
    for model_id, selectable in (
        ("gemini-2.5-flash", True), ("gemini-2.5-pro", True),
        ("gemini-2.5-flash-preview-tts", False),
        ("gemini-2.5-pro-preview-tts", False),
        ("gemini-flash-latest", True), ("gemini-flash-lite-latest", True),
        ("gemini-pro-latest", True), ("gemini-2.5-flash-lite", True),
        ("gemini-2.5-flash-image", False), ("gemini-3-flash-preview", True),
        ("gemini-3.1-pro-preview", True),
        ("gemini-3.1-pro-preview-customtools", True),
        ("gemini-3.1-flash-lite-preview", True),
        ("gemini-3.1-flash-lite", True),
        ("gemini-3-pro-image-preview", False), ("gemini-3-pro-image", False),
        ("gemini-3.1-flash-image-preview", False),
        ("gemini-3.1-flash-image", False),
        ("gemini-3.1-flash-lite-image", False),
        ("gemini-nano-banana-2.1", True), ("gemini-3.5-flash", True),
        ("gemini-3.5-flash-lite", True), ("gemini-omni-flash-preview", True),
        ("gemini-omni-1.1-flash", True), ("gemini-3.5-transcribe", False),
        ("gemini-3.6-flash", True), ("gemini-3.7-flash", True),
        ("gemini-3.8-flash", True), ("gemini-4-argon", True),
        ("gemini-3.1-flash-tts-preview", False),
        ("gemini-3.8-flash-tts", False),
        ("gemini-3.8-flash-lite-tts", False),
        ("gemini-robotics-er-2-preview", False),
        ("gemini-2.5-computer-use-preview-10-2025", False),
        ("gemini-embedding-001", False),
        ("gemini-embedding-2-preview", False), ("gemini-embedding-2", False),
        ("gemini-3.5-transcribe-live", False),
        ("gemini-2.5-flash-native-audio-latest", False),
        ("gemini-2.5-flash-native-audio-preview-09-2025", False),
        ("gemini-2.5-flash-native-audio-preview-12-2025", False),
        ("gemini-3.1-flash-live-preview", False), ("gemini-3.8-live", False),
        ("gemini-3.8-live-extended-thinking", False),
        ("gemini-robotics-er-2-streaming-preview", False),
        ("gemini-3.5-live-translate-preview", False),
    )
]

# Tạo các thư mục cần thiết
os.makedirs(os.path.join(WORKSPACE_DIR, "templates"), exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "static", "css"), exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "static", "js"), exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "static", "bgm"), exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "static", "output"), exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "static", "temp_preview"), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)

# Keep native Python crash information even when the web process disappears.
_crash_log_handle = open(os.path.join(LOG_DIR, "python_crash.log"), "a", encoding="utf-8", buffering=1)
faulthandler.enable(file=_crash_log_handle, all_threads=True)

def _load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as settings_file:
            return json.load(settings_file)
    except (OSError, json.JSONDecodeError):
        return {}

def _save_settings(settings):
    temp_path = SETTINGS_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8", newline="\n") as settings_file:
        json.dump(settings, settings_file, ensure_ascii=False, indent=2)
        settings_file.flush()
        os.fsync(settings_file.fileno())
    os.replace(temp_path, SETTINGS_PATH)


def get_configured_export_directory():
    export_dir = _load_settings().get("export_directory", "")
    if not isinstance(export_dir, str):
        return ""
    export_dir = export_dir.strip().strip('"')
    return os.path.abspath(os.path.normpath(export_dir)) if export_dir else ""

def get_export_directory():
    export_dir = get_configured_export_directory()
    return export_dir if export_dir and os.path.isdir(export_dir) else ""

def check_export_directory_writable(export_dir):
    """Verify an export folder with a real create/flush/delete operation."""
    if not export_dir or not os.path.isdir(export_dir):
        return False, "Thu muc xuat khong ton tai."
    probe_path = os.path.join(export_dir, f".review_tool_write_test_{uuid.uuid4().hex}.tmp")
    try:
        with open(probe_path, "xb") as probe:
            probe.write(b"ok")
            probe.flush()
            os.fsync(probe.fileno())
        os.remove(probe_path)
        return True, ""
    except OSError as exc:
        try:
            if os.path.exists(probe_path):
                os.remove(probe_path)
        except OSError:
            pass
        return False, f"Khong co quyen ghi vao thu muc: {exc}"

export_copy_lock = threading.Lock()

def copy_result_to_export_directory(result_url, export_directory=""):
    """Atomically copy a completed MP4/ZIP and verify its size."""
    configured_dir = os.path.abspath(os.path.normpath(export_directory)) if export_directory else get_configured_export_directory()
    if not configured_dir:
        return "", "Chua chon thu muc xuat video."
    if not os.path.isdir(configured_dir):
        return "", f"Thu muc xuat khong ton tai: {configured_dir}. Vui long chon lai."

    filename = os.path.basename(result_url.split("?", 1)[0].rstrip("/\\"))
    source_path = os.path.join(WORKSPACE_DIR, "static", "output", filename)
    if not os.path.isfile(source_path):
        return "", "Khong tim thay file ket qua de sao chep."

    with export_copy_lock:
        target_path = os.path.join(configured_dir, filename)
        stem, extension = os.path.splitext(filename)
        counter = 1
        while os.path.exists(target_path):
            target_path = os.path.join(configured_dir, f"{stem}_{counter}{extension}")
            counter += 1

        partial_path = target_path + f".{uuid.uuid4().hex}.partial"
        last_error = None
        for attempt in range(3):
            try:
                shutil.copy2(source_path, partial_path)
                if os.path.getsize(partial_path) != os.path.getsize(source_path):
                    raise OSError("Kich thuoc file sau khi chep khong khop file goc.")
                os.replace(partial_path, target_path)
                if not os.path.isfile(target_path):
                    raise OSError("File dich khong ton tai sau khi chep.")
                return target_path, ""
            except OSError as exc:
                last_error = exc
                try:
                    if os.path.exists(partial_path):
                        os.remove(partial_path)
                except OSError:
                    pass
                if attempt < 2:
                    time.sleep(1)
        return "", f"Khong chep duoc video vao '{configured_dir}': {last_error}"

app = Flask(__name__, static_folder=os.path.join(WORKSPACE_DIR, "static"), template_folder=os.path.join(WORKSPACE_DIR, "templates"))
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB cho video dai
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.before_request
def validate_api_json():
    if not request.path.startswith('/api/') or not request.is_json:
        return None
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify(success=False, message='Du lieu JSON phai la mot object.'), 400
    for field in ('video_name', 'bgm_name', 'api_key', 'groq_api_key', 'model_name',
                  'custom_prompt', 'script_text', 'subtitle_text', 'project_id',
                  'text', 'voice', 'voice_name', 'style', 'style_prompt',
                  'render_engine', 'aspect_ratio', 'export_mode'):
        if field in data and not isinstance(data[field], str):
            return jsonify(success=False, message=f'{field} phai la chuoi van ban.'), 400
    for field in ('bgm_volume', 'zoom_factor', 'speed_factor', 'contrast', 'brightness',
                  'font_size', 'bg_opacity'):
        if field not in data:
            continue
        try:
            valid = not isinstance(data[field], bool) and math.isfinite(float(data[field]))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            return jsonify(success=False, message=f'{field} phai la so huu han.'), 400
        number = float(data[field])
        if field in ('font_size', 'bg_opacity') and not number.is_integer():
            return jsonify(success=False, message=f'{field} phai la so nguyen.'), 400
        if field in ('font_size', 'bg_opacity'):
            data[field] = int(number)
        if field in ('font_size', 'zoom_factor', 'speed_factor') and number <= 0:
            return jsonify(success=False, message=f'{field} phai lon hon 0.'), 400
    return None


@app.errorhandler(400)
@app.errorhandler(415)
def api_bad_request(error):
    if request.path.startswith('/api/'):
        return jsonify(success=False, message='Yeu cau JSON khong hop le; hay kiem tra du lieu gui len.'), error.code
    return error


@app.after_request
def disable_local_ui_cache(response):
    """The local tool must not keep an old HTML/JS build after code updates."""
    if request.path == "/" or request.path.endswith((".js", ".css")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

# Khóa bảo mật mặc định
DEFAULT_API_KEY = ""
STORY_PLAN_MAX_ATTEMPTS = 2
GEMINI_GENERATION_MAX_ATTEMPTS = 3
GEMINI_REQUEST_TIMEOUT_SECONDS = 6 * 60
GEMINI_503_MAX_RETRIES = 2
GEMINI_503_RETRY_DELAYS = (3 * 60, 3 * 60)
STORY_PLAN_REQUEST_ATTEMPTS = 1
STORY_PLAN_PRIMARY_MODEL = "gemini-3.8-flash"
STORY_PLAN_FALLBACK_MODEL = "gemini-3.7-flash"
STORY_PLAN_503_RETRY_DELAY_SECONDS = 60
# One primary request followed by five retries, each on the next model.
STORY_PLAN_MODEL_SEQUENCE = (
    STORY_PLAN_PRIMARY_MODEL,
    STORY_PLAN_FALLBACK_MODEL,
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
STORY_DIGEST_MAX_EVENTS = 60
BATCH_REQUEST_COOLDOWN_SECONDS = 15
GEMINI_CAPACITY_RESUME_DELAY_SECONDS = 120
GEMINI_CAPACITY_CYCLES = 1
GEMINI_INPUT_TOKENS_PER_MINUTE = 250000
# Keep a margin for prompt text and small differences in Google's video
# tokenization. The limiter uses actual usage returned by Gemini after every
# successful request and waits before admitting the next large request.
GEMINI_INPUT_TOKENS_SAFE_PER_MINUTE = 220000
GEMINI_QUOTA_WINDOW_SECONDS = 60
GEMINI_QUOTA_GUARD_SECONDS = 2
GEMINI_MINUTE_QUOTA_MAX_RETRIES = 12
EVIDENCE_CONTRACT_VERSION = 20
# The whole-film pass is only a cheap story map, not an evidence extractor.
# The global director pass receives Whisper text and a silent visual proxy, so
# one frame every ten seconds is enough to map chapters without paying for the
# source audio twice. Detailed silent chapter passes use two FPS.
FULL_VIDEO_ANALYSIS_FPS = 0.1
DETAIL_VIDEO_ANALYSIS_FPS = 2.0
GEMINI_PRIMARY_PART_COUNT = 3
GEMINI_EVENT_LOOKAHEAD_SECONDS = 90
MAX_TARGETED_VISUAL_REVIEWS = 2
GEMINI_TASK_SOFT_TOKEN_BUDGET = 700000
TWO_PASS_ANALYSIS_MODEL = "gemini-3.1-flash-lite"
# The whole-video outline is useful as a map, but it is too coarse to be the
# sole source of recap scenes. Always collect high-recall evidence from short
# chunks before the global selector is allowed to discard anything.
USE_ANCHOR_VISUAL_REVIEW = True

# =====================================================================
# Quản lý Tiến Trình Chạy Ngầm (Thread-safe)
# =====================================================================
TASKS = {}
PROJECTS = {}
task_lock = threading.Lock()
task_log_write_lock = threading.Lock()

# Whitelist dinh dang file duoc phep upload
ALLOWED_VIDEO_EXT = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
ALLOWED_AUDIO_EXT = {'.mp3', '.wav', '.m4a', '.ogg'}

def create_task(project_id="", video_name="", persistent=False, task_kind="generic"):
    task_id = str(uuid.uuid4())
    with task_lock:
        TASKS[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "Dang khoi tao...",
            "result": None,
            "project_id": project_id,
            "video_name": video_name,
            "created_at": __import__('time').time(),
            "logs": [],
            "persistent": bool(persistent),
            "task_kind": str(task_kind or "generic"),
        }
        if task_kind == 'analysis':
            TASKS[task_id]['analysis_pid'] = os.getpid()
        snapshot = copy.deepcopy(TASKS[task_id])
    if persistent:
        save_task_state(task_id, snapshot)
    return task_id


_task_log_checkpoint_at = {}


def _persist_task_if_needed(task_id):
    with task_lock:
        now = time.monotonic()
        task = TASKS.get(task_id)
        terminal = task and task.get('status') in ('success', 'failed')
        if not terminal and now - _task_log_checkpoint_at.get(task_id, -float('inf')) < 15:
            return
        _task_log_checkpoint_at[task_id] = now
        task = TASKS.get(task_id)
        snapshot = copy.deepcopy(task) if task and task.get("persistent") else None
    if snapshot is not None:
        try:
            save_task_state(task_id, snapshot)
        except OSError as exc:
            # A transient Windows file lock must never abort video/TTS work.
            # A later heartbeat or log entry will checkpoint the same RAM state.
            print(f"[TASK STATE WARNING] Chua luu duoc checkpoint {task_id[:8]}: {exc}", flush=True)


def update_task_fields(task_id, **fields):
    """Update RAM state and atomically checkpoint persistent render state."""
    with task_lock:
        task = TASKS.setdefault(task_id, {"logs": [], "persistent": True})
        task.update(fields)
        # Progress and diagnostic counters are recoverable; durable artifacts,
        # status and results still checkpoint immediately.
        transient = {'progress', 'message', 'visual_fallback_reports', 'visual_fallback_count',
                     'heartbeat_at', 'narration_audit_checkpoint', 'grounded_narration_audit'}
        if set(fields).issubset(transient):
            now = time.monotonic()
            if now - _task_log_checkpoint_at.get(task_id, -float('inf')) < 15:
                return None
            _task_log_checkpoint_at[task_id] = now
        snapshot = copy.deepcopy(task) if task.get("persistent") else None
    if snapshot is not None:
        try:
            save_task_state(task_id, snapshot)
        except OSError as exc:
            print(f"[TASK STATE WARNING] Chua luu duoc checkpoint {task_id[:8]}: {exc}", flush=True)
    return snapshot


def restore_persistent_task(task_id, state):
    restored = copy.deepcopy(state or {})
    restored["persistent"] = True
    restored.setdefault("logs", [])
    with task_lock:
        TASKS[task_id] = restored
    try:
        save_task_state(task_id, restored)
    except OSError as exc:
        print(f"[TASK STATE WARNING] Chua luu duoc checkpoint {task_id[:8]}: {exc}", flush=True)

def _task_log_path(task_id):
    return os.path.join(LOG_DIR, f"task_{task_id}.log")

def append_task_log(task_id, level, message, details=""):
    """Store diagnostic events so the browser can show the real backend log."""
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "level": str(level or "INFO").upper(),
        "message": str(message),
    }
    if details:
        entry["details"] = str(details)
    with task_lock:
        task = TASKS.get(task_id)
        if task is not None:
            # Full diagnostics belong to the append-only file, not every snapshot.
            ui_entry = dict(entry)
            if len(ui_entry.get('details', '')) > 4000:
                ui_entry['details'] = ui_entry['details'][:4000] + '\n[Full details in task log file]'
            task.setdefault("logs", []).append(ui_entry)
            # Prevent an unusually long run from growing memory forever.
            if len(task["logs"]) > 2000:
                task["logs"] = task["logs"][-2000:]
    _persist_task_if_needed(task_id)
    # File log is deliberately independent from TASKS, which only lives in RAM.
    # It therefore survives a Flask/VS Code crash and can be inspected afterward.
    try:
        with task_log_write_lock, open(_task_log_path(task_id), "a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(f"[{entry['time']}] [{entry['level']}] {entry['message']}\n")
            if details:
                log_file.write(str(details).rstrip() + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())
    except OSError as log_error:
        print(f"[LOG ERROR] Khong ghi duoc task log: {log_error}", flush=True)
    print(f"[{entry['time']}] [{entry['level']}] [task {task_id[:8]}] {entry['message']}", flush=True)
    if details:
        print(details, flush=True)

def calculate_script_budget(video_duration_min, review_minutes="auto", style="dramatic",
                            voice_rate=DEFAULT_VOICE_RATE):
    """Return one predictable recap budget for every source duration."""
    video_duration_min = max(1.0, float(video_duration_min))
    if review_minutes == "auto":
        # Auto mode targets one fifth of the source duration.
        target_review_min = max(1.5, video_duration_min * 0.20)
    else:
        target_review_min = min(float(review_minutes), video_duration_min)

    target_word_count = narration_budget(target_review_min, voice_rate)['target_words']
    _, min_word_count, max_word_count = requested_word_range(target_word_count)
    return target_review_min, target_word_count, min_word_count, max_word_count


def recap_depth_instruction(video_duration_min, target_review_min):
    """Choose story depth relative to the suggested one-fifth duration."""
    source_minutes = max(0.1, float(video_duration_min))
    target_minutes = max(0.1, float(target_review_min))
    ratio = target_minutes / max(0.1, source_minutes * 0.20)
    if ratio < 0.80:
        mode = "CO_DONG"
        rule = (
            "Chi giu tuyen truyen chinh: mo dau can thiet, bien co khoi phat, "
            "cac buoc ngoat, cao trao va ket cuc that. Gop hoac bo tuyen phu "
            "khong lam thay doi muc tieu, nguy hiem hay ket qua."
        )
    elif ratio <= 1.20:
        mode = "CAN_BANG"
        rule = (
            "Ke du chuoi nguyen nhan -> hanh dong -> hau qua cua tuyen chinh; "
            "chi giu tuyen phu khi no tac dong truc tiep den buoc ngoat hoac ket cuc."
        )
    else:
        mode = "MO_RONG"
        rule = (
            "Bo sung dong co, quan he, cam xuc co bang chung va tuyen phu co anh huong; "
            "khong lap y, ke tung shot, suy dien noi tam hay bia tinh tiet de keo dai."
        )
    return (
        f"CHE_DO_DO_SAU={mode}; recap duoc chon bang {ratio:.2f} lan moc 1/5. "
        f"{rule} Neu da ke het cot truyen thi uu tien ban ke gon, khong chen noi dung rong."
    )


def _format_review_time(seconds):
    value = max(0.0, float(seconds))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = value % 60
    seconds_text = f"{secs:05.2f}".rstrip("0").rstrip(".")
    if len(seconds_text.split(".", 1)[0]) < 2:
        seconds_text = "0" + seconds_text
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_text}"
    return f"{minutes:02d}:{seconds_text}"


def measure_script_narration(script_text):
    """Count spoken words and valid timeline rows, excluding TIME metadata."""
    word_count = 0
    event_count = 0
    for row in format_timeline_rows(script_text).splitlines():
        row = row.strip()
        if not row:
            continue
        start, end, narration = extract_time_range(row)
        if start is None or end is None:
            continue
        event_count += 1
        from narrative_contract import spoken_text
        narration = spoken_text(narration)
        word_count += len(re.findall(r"\b\w+\b", narration, flags=re.UNICODE))
    return word_count, event_count


def script_with_error(partial_script, error_message):
    """Keep usable partial narration and append one visible terminal error row."""
    partial = str(partial_script or "").strip()
    error = re.sub(r"\s+", " ", str(error_message or "Lỗi không xác định.")).strip()
    error = re.sub(r"^(?:lỗi|loi)\s*:\s*", "", error, flags=re.IGNORECASE)
    error_row = f"[LỖI: {error}]"
    return f"{partial}\n\n{error_row}" if partial else error_row


def gemini_response_diagnostic(response_text, limit=12000):
    """Keep enough local evidence to diagnose schema failures without huge logs."""
    text = str(response_text or "").strip()
    if not text:
        return "(Gemini khong tra text)"
    limit = max(500, int(limit))
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n... [da rut gon {len(text) - limit} ky tu] ...\n"
        + text[-half:]
    )


def trim_timeline_ends_to_locked_starts(script_text):
    """Resolve ordinary overlaps by trimming END only; never move a START."""
    rows = [
        row.strip() for row in format_timeline_rows(script_text).splitlines()
        if row.strip()
    ]
    parsed = []
    for row in rows:
        start, end, narration = extract_time_range(row)
        if start is None or end is None:
            parsed.append([start, end, narration, row])
        else:
            parsed.append([float(start), float(end), narration, row])

    for index in range(1, len(parsed)):
        previous = parsed[index - 1]
        current = parsed[index]
        if previous[0] is None or current[0] is None:
            continue
        if current[0] <= previous[0]:
            # Repeated/backwards START values are evidence errors. Preserve
            # them so validation reports the bad Gemini anchor explicitly.
            continue
        if previous[1] is not None and previous[1] > current[0]:
            previous[1] = current[0]

    normalized = []
    for start, end, narration, original in parsed:
        if start is None or end is None:
            normalized.append(original)
            continue
        normalized.append(
            f"[TIME={_format_review_time(start)}-{_format_review_time(end)}] {narration}"
        )
    return "\n".join(normalized)


def analysis_proxy_fps(video_duration_sec):
    """Keep enough temporal detail for Gemini without making long uploads excessive."""
    duration_min = max(0.0, float(video_duration_sec)) / 60.0
    if duration_min <= 10:
        return 12
    if duration_min <= 30:
        return 8
    if duration_min <= 60:
        return 6
    return 3


def build_model_candidates(selected_model, available_models):
    """Use the selected compatible model first, then known Flash fallbacks."""
    blocked_markers = (
        "-tts", "-image", "-live", "transcribe", "embedding", "aqa",
        "robotics", "bidi", "computer-use", "deep-research", "native-audio",
    )
    usable = []
    for model in available_models:
        raw_name = getattr(model, "name", model)
        name = str(raw_name).replace("models/", "")
        if "gemini" not in name.lower() or any(marker in name.lower() for marker in blocked_markers):
            continue
        supported_actions = getattr(model, "supported_actions", None)
        if supported_actions:
            action_names = [
                str(getattr(action, "value", action)).lower().replace("_", "")
                for action in supported_actions
            ]
            if not any(
                "generatecontent" in action and "bidi" not in action
                for action in action_names
            ):
                continue
        usable.append(name)

    preferred_names = [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3-flash-preview",
    ]
    selected_model = str(selected_model or "").replace("models/", "")
    selected = [selected_model] if selected_model in usable else []
    preferred = [name for name in preferred_names if name in usable]
    ordered = selected + preferred + [m for m in usable if "flash" in m.lower()] + usable
    seen = set()
    return [m for m in ordered if m and m not in seen and not seen.add(m)]


def describe_gemini_models(available_models):
    """List every Gemini model while marking safe text-recap choices."""
    blocked_markers = (
        "-tts", "-image", "-live", "transcribe", "embedding", "aqa",
        "robotics", "bidi", "computer-use", "deep-research", "native-audio",
    )
    rows = []
    seen = set()
    for model in available_models or []:
        name = str(getattr(model, "name", model)).replace("models/", "").strip()
        if not name or "gemini" not in name.lower() or name in seen:
            continue
        seen.add(name)
        actions = getattr(model, "supported_actions", None) or []
        action_names = [
            str(getattr(action, "value", action)).lower().replace("_", "")
            for action in actions
        ]
        supports_generate = not action_names or any(
            "generatecontent" in action and "bidi" not in action
            for action in action_names
        )
        blocked = any(marker in name.lower() for marker in blocked_markers)
        rows.append({"id": name, "selectable": bool(supports_generate and not blocked)})
    return rows


def _save_gemini_model_cache(rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(GEMINI_MODEL_CACHE_PATH), exist_ok=True)
    temporary = GEMINI_MODEL_CACHE_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, GEMINI_MODEL_CACHE_PATH)


def _load_gemini_model_cache():
    try:
        with open(GEMINI_MODEL_CACHE_PATH, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        valid_rows = [
            row for row in rows if isinstance(row, dict) and row.get("id")
        ]
        return valid_rows or copy.deepcopy(DEFAULT_GEMINI_MODEL_ROWS)
    except (OSError, ValueError, TypeError):
        return copy.deepcopy(DEFAULT_GEMINI_MODEL_ROWS)


def resolve_selected_gemini_model(model_name):
    """Use the UI-selected model verbatim, with a default only when empty."""
    selected = str(model_name or "").replace("models/", "").strip()
    return selected or "gemini-3.1-flash-lite"


def thinking_config_for_effort(genai_types, model_name, effort="low"):
    """Translate the UI effort to the reasoning control supported by a model."""
    effort = str(effort or "low").strip().lower()
    if effort not in {"low", "medium", "high"}:
        effort = "low"
    normalized = str(model_name or "").replace("models/", "").lower()
    match = re.search(r"gemini-(\d+)", normalized)
    major = int(match.group(1)) if match else 3
    if major >= 3:
        level = {
            "low": genai_types.ThinkingLevel.LOW,
            "medium": genai_types.ThinkingLevel.MEDIUM,
            "high": genai_types.ThinkingLevel.HIGH,
        }[effort]
        return genai_types.ThinkingConfig(
            thinking_level=level, include_thoughts=False
        )
    budget = {"low": 1024, "medium": 4096, "high": 8192}[effort]
    return genai_types.ThinkingConfig(
        thinking_budget=budget, include_thoughts=False
    )


def build_full_video_gemini_part(
    genai_types, uploaded_file, fps=FULL_VIDEO_ANALYSIS_FPS,
):
    """Create the one-pass full-video input at the token-safe analysis quality."""
    file_uri = str(getattr(uploaded_file, "uri", "") or "").strip()
    if not file_uri:
        return uploaded_file
    mime_type = str(getattr(uploaded_file, "mime_type", "") or "video/mp4")
    content = genai_types.Part.from_uri(
        file_uri=file_uri,
        mime_type=mime_type,
        media_resolution=(
            genai_types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
        ),
    )
    content.video_metadata = genai_types.VideoMetadata(
        fps=float(fps),
    )
    return content


def gemini_error_chain_text(error):
    """Return messages from an exception and every wrapped cause/context."""
    messages = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current or "").strip()
        if message:
            messages.append(message)
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return " | caused by: ".join(messages)


def is_gemini_transport_error(error):
    """Recognize disconnects even when exception text omits its class name."""
    current, seen = error, set()
    transport_types = {
        'RemoteProtocolError', 'ReadError', 'WriteError', 'ConnectError',
        'ReadTimeout', 'WriteTimeout', 'ConnectTimeout', 'PoolTimeout',
        'ConnectionResetError', 'ConnectionAbortedError', 'BrokenPipeError',
    }
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        if any(cls.__name__ in transport_types for cls in type(current).__mro__):
            return True
        current = current.__cause__ or current.__context__
    message = gemini_error_chain_text(error).lower()
    return any(marker in message for marker in (
        'peer closed connection', 'incomplete chunked read',
        'server disconnected', 'connection reset', 'connection aborted',
        'remoteprotocolerror', 'readerror', 'connecterror', 'broken pipe',
    ))


def is_transient_gemini_error(error):
    """Return True only for temporary server/capacity failures, never quota errors."""
    message = gemini_error_chain_text(error).lower()
    if any(marker in message for marker in ("429", "resource_exhausted", "quota")):
        return False
    if is_gemini_auth_error(error):
        return False
    if is_gemini_transport_error(error):
        return True
    return any(marker in message for marker in (
        "503", "unavailable", "high demand", "temporarily", "internal",
        "500", "502", "504", "deadline exceeded", "remoteprotocolerror",
        "server disconnected", "connection reset", "connection aborted",
        "readerror", "connecterror", "timeout", "timed out",
    ))


def is_gemini_transport_timeout(error):
    message = gemini_error_chain_text(error).lower()
    return any(marker in message for marker in ('readtimeout', 'connecttimeout', 'timeout', 'timed out'))


def is_gemini_capacity_503_error(error):
    """Identify model-capacity 503s separately from timeouts/large requests."""
    message = gemini_error_chain_text(error).lower()
    return "503" in message and any(marker in message for marker in (
        "unavailable", "high demand", "temporarily",
    ))


def is_per_minute_gemini_quota_error(error):
    """Retry only rolling per-minute quota, never daily/project exhaustion."""
    message = gemini_error_chain_text(error).lower().replace("_", "")
    if any(marker in message for marker in (
        "perdayperproject", "requestsperdayperproject", "per day per project",
    )):
        return False
    return (
        ("429" in message or "resourceexhausted" in message)
        and any(marker in message for marker in (
            "perminute", "per minute", "retrydelay", "inputtokencount",
        ))
    )


def is_daily_project_gemini_quota_error(error):
    """Detect only daily per-project exhaustion, which can use the next key."""
    message = gemini_error_chain_text(error).lower().replace("_", "")
    if not ("429" in message or "resourceexhausted" in message):
        return False
    return any(marker in message for marker in (
        "perdayperproject",
        "requestsperdayperproject",
        "per day per project",
        "generatedrequestsperdayperprojectpermodel",
        "generaterequestsperdayperprojectpermodel",
    ))


def gemini_retry_delay_seconds(error, attempt):
    """Honor Google's RetryInfo when present, otherwise use bounded backoff."""
    message = str(error or "")
    matches = re.findall(
        r"(?:retry(?:Delay| in)?[^0-9]{0,20})(\d+(?:\.\d+)?)\s*s",
        message,
        flags=re.IGNORECASE,
    )
    if matches:
        return min(90.0, max(1.0, float(matches[-1]) + 1.0))
    lowered = message.lower()
    if any(marker in lowered for marker in ("503", "unavailable", "high demand")):
        overload_delays = (10.0, 20.0, 40.0, 60.0, 90.0)
        return overload_delays[min(max(0, int(attempt)), len(overload_delays) - 1)]
    return min(45.0, 5.0 * (2 ** max(0, int(attempt))))


def upload_gemini_file_with_recovery(
    client, file_path, label="video", update_progress=None,
    max_attempts=5, sleep_fn=time.sleep,
):
    """Upload a Gemini file with the same transport recovery as generation."""
    last_error = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            return client.files.upload(file=file_path)
        except Exception as exc:
            last_error = exc
            retryable = (
                is_transient_gemini_error(exc)
                or is_per_minute_gemini_quota_error(exc)
            )
            if not retryable or attempt + 1 >= max_attempts:
                raise
            delay = gemini_retry_delay_seconds(exc, attempt)
            if update_progress is not None:
                update_progress(
                    f"Ket noi upload {label} bi ngat; thu lai "
                    f"{attempt + 2}/{max_attempts} sau {int(math.ceil(delay))} giay..."
                )
            sleep_fn(delay)
    raise last_error or RuntimeError(f"Upload {label} that bai.")


def gemini_usage_dict(usage_metadata):
    """Normalize SDK usage metadata without depending on one SDK version."""
    if usage_metadata is None:
        return {}
    fields = {
        "input_tokens": ("prompt_token_count", "input_token_count"),
        "output_tokens": ("candidates_token_count", "output_token_count"),
        "thought_tokens": ("thoughts_token_count", "thought_token_count"),
        "tool_tokens": ("tool_use_prompt_token_count", "tool_use_token_count"),
        "total_tokens": ("total_token_count",),
    }
    result = {}
    for target, candidates in fields.items():
        value = 0
        for name in candidates:
            raw = getattr(usage_metadata, name, None)
            if raw is None and isinstance(usage_metadata, dict):
                raw = usage_metadata.get(name)
            try:
                value = int(raw or 0)
            except (TypeError, ValueError):
                value = 0
            if value:
                break
        result[target] = max(0, value)
    if not result["total_tokens"]:
        result["total_tokens"] = (
            result["input_tokens"] + result["output_tokens"]
            + result["thought_tokens"] + result["tool_tokens"]
        )
    return result


class GeminiMinuteQuotaPacer:
    """Pace input tokens in Google's rolling per-minute quota window."""

    def __init__(
        self,
        safe_limit=GEMINI_INPUT_TOKENS_SAFE_PER_MINUTE,
        window_seconds=GEMINI_QUOTA_WINDOW_SECONDS,
        guard_seconds=GEMINI_QUOTA_GUARD_SECONDS,
        clock_fn=time.monotonic,
    ):
        self.safe_limit = max(1, int(safe_limit))
        self.window_seconds = max(1.0, float(window_seconds))
        self.guard_seconds = max(0.0, float(guard_seconds))
        self.clock_fn = clock_fn
        self._usage = []
        self._lock = threading.RLock()
        self._inflight = {}

    def _active_usage(self, now=None):
        current = self.clock_fn() if now is None else float(now)
        cutoff = current - self.window_seconds
        self._usage = [
            (timestamp, tokens)
            for timestamp, tokens in self._usage
            if timestamp > cutoff and tokens > 0
        ]
        return current, list(self._usage)

    def record(self, input_tokens, now=None):
        tokens = max(0, int(input_tokens or 0))
        if not tokens:
            return
        with self._lock:
            current, _ = self._active_usage(now)
            self._usage.append((current, tokens))
            self._usage.sort(key=lambda item: item[0])

    def required_wait(self, estimated_input_tokens, now=None):
        """Return seconds until an estimated request fits the safe window."""
        with self._lock:
            return self._required_wait(estimated_input_tokens, now)

    def _required_wait(self, estimated_input_tokens, now=None):
        estimate = min(
            self.safe_limit,
            max(0, int(estimated_input_tokens or 0)),
        )
        if not estimate:
            return 0.0
        current, active = self._active_usage(now)
        reserved = sum(self._inflight.values())
        active_tokens = sum(tokens for _, tokens in active) + reserved
        if active_tokens + estimate <= self.safe_limit:
            return 0.0
        remaining = active_tokens
        for timestamp, tokens in active:
            remaining -= tokens
            if remaining + estimate <= self.safe_limit:
                return max(
                    0.0,
                    timestamp + self.window_seconds + self.guard_seconds - current,
                )
        # In-flight requests remain reserved even when they last over a minute.
        # Poll until their returned usage can replace the reservation.
        return 2.0 if reserved else 0.0

    def reserve_request(self, label, estimated_input_tokens, update_progress=None, sleep_fn=time.sleep):
        estimate = min(self.safe_limit, max(1, int(estimated_input_tokens or 1)))
        while True:
            with self._lock:
                delay = self._required_wait(estimate)
                if delay <= 0:
                    ticket = object()
                    self._inflight[ticket] = estimate
                    return ticket
            wait_with_progress(delay, f'{label}: dang cho quota token/phut',
                               update_progress=update_progress, sleep_fn=sleep_fn)

    def complete_request(self, ticket, input_tokens=None):
        with self._lock:
            estimate = self._inflight.pop(ticket, 0)
            # A timeout may already have consumed tokens. Keep the conservative
            # estimate when the server did not return usage metadata.
            self.record(estimate if input_tokens is None else input_tokens)

    def wait_before_request(
        self, label, estimated_input_tokens, update_progress=None,
        sleep_fn=time.sleep,
    ):
        delay = self.required_wait(estimated_input_tokens)
        if delay <= 0:
            return 0.0
        wait_with_progress(
            delay,
            f"{label}: dang cho cua so quota token/phut hoi phuc",
            update_progress=update_progress,
            sleep_fn=sleep_fn,
        )
        return delay


def estimate_gemini_video_input_tokens(duration_seconds, fps):
    """Conservative request estimate calibrated from recorded task usage."""
    duration = max(0.0, float(duration_seconds or 0.0))
    frames_per_second = max(0.01, float(fps or 0.01))
    tokens_per_second = 85.0 * frames_per_second if frames_per_second >= 0.9 else 13.0
    return min(
        GEMINI_INPUT_TOKENS_SAFE_PER_MINUTE,
        max(10000, int(math.ceil(duration * tokens_per_second + 10000))),
    )


def generate_gemini_text_with_recovery(
    client, model, contents, config, label, update_progress=None,
    stream=False, max_attempts=GEMINI_GENERATION_MAX_ATTEMPTS, sleep_fn=time.sleep,
    resume_503_forever=False, capacity_retry_delay_seconds=60,
    usage_callback=None, quota_pacer=None, estimated_input_tokens=0,
    capacity_max_retries=None, retry_timeouts=True,
    minute_quota_max_retries=None, retry_transport_errors=True,
    diagnostic_callback=None,
):
    """Generate text with streaming and bounded recovery for transport/TPM errors."""
    last_error = None
    attempt = 0
    minute_quota_attempt = 0
    timeout_retries = 0
    capacity_wait_mode = False
    fast_attempt_limit = max(1, int(max_attempts))
    from request_diagnostics import RequestTrace
    trace = RequestTrace(diagnostic_callback or getattr(usage_callback, 'diagnostic_callback', None),
                         label, model, contents, config)
    while True:
        quota_ticket = None
        trace.begin()
        try:
            if quota_pacer is not None:
                quota_ticket = quota_pacer.reserve_request(
                    label,
                    estimated_input_tokens,
                    update_progress=update_progress,
                    sleep_fn=sleep_fn,
                )
            trace.send()
            if stream:
                parts = []
                usage = {}
                for chunk in client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    chunk_usage = gemini_usage_dict(
                        getattr(chunk, "usage_metadata", None)
                    )
                    if chunk_usage.get("total_tokens"):
                        usage = chunk_usage
                    chunk_text = str(getattr(chunk, "text", "") or "")
                    trace.chunk(chunk_text)
                    if chunk_text:
                        parts.append(chunk_text)
                result_text = "".join(parts).strip()
            else:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                usage = gemini_usage_dict(
                    getattr(response, "usage_metadata", None)
                )
                result_text = str(getattr(response, "text", "") or "").strip()
            if not result_text:
                raise RuntimeError("Gemini tra ve noi dung rong.")
            trace.success(result_text)
            if usage_callback is not None and usage:
                usage_callback(label, usage)
            if quota_pacer is not None:
                quota_pacer.complete_request(quota_ticket, usage.get('input_tokens') if usage else None)
                quota_ticket = None
            return result_text
        except Exception as exc:
            trace.failure(exc)
            if quota_ticket is not None:
                quota_pacer.complete_request(quota_ticket)
            last_error = exc
            is_minute_quota = is_per_minute_gemini_quota_error(exc)
            if is_minute_quota:
                minute_limit = GEMINI_MINUTE_QUOTA_MAX_RETRIES if minute_quota_max_retries is None else max(0, int(minute_quota_max_retries))
                if minute_quota_attempt >= minute_limit:
                    trace.emit('stopped', reason='per_minute_quota_retry_limit', retry_limit=minute_limit)
                    raise
                delay = gemini_retry_delay_seconds(exc, minute_quota_attempt)
                trace.wait(delay, 'per_minute_quota')
                wait_with_progress(
                    delay,
                    f"{label} cham gioi han token/phut; giu tien do va thu lai dung request",
                    update_progress=update_progress,
                    sleep_fn=sleep_fn,
                )
                minute_quota_attempt += 1
                continue
            retryable = (
                is_transient_gemini_error(exc)
            )
            is_capacity_503 = is_gemini_capacity_503_error(exc)
            if capacity_wait_mode:
                if not (resume_503_forever and is_capacity_503):
                    trace.emit('stopped', reason='capacity_wait_interrupted_by_other_error')
                    raise
                trace.wait(capacity_retry_delay_seconds, 'capacity_wait_mode')
                wait_with_progress(
                    capacity_retry_delay_seconds,
                    f"{label} van bi Gemini qua tai; giu nguyen tien do va thu lai",
                    update_progress=update_progress,
                    sleep_fn=sleep_fn,
                )
                continue
            if not retryable:
                trace.emit('stopped', reason='non_retryable')
                raise
            capacity_retries = (
                GEMINI_503_MAX_RETRIES
                if capacity_max_retries is None else
                max(0, int(capacity_max_retries))
            )
            attempt_limit = (
                capacity_retries + 1
                if is_capacity_503 else fast_attempt_limit
            )
            if ((is_gemini_transport_timeout(exc) and retry_timeouts)
                    or (is_gemini_transport_error(exc) and retry_transport_errors
                        and not is_gemini_transport_timeout(exc))):
                # A timeout replays only this request, once, even for stages
                # configured with one semantic attempt. Completed work stays.
                if timeout_retries >= 1:
                    trace.emit('stopped', reason='transport_retry_limit')
                    raise
                timeout_retries += 1
                attempt_limit = attempt + 2
            if attempt + 1 >= attempt_limit:
                if not (resume_503_forever and is_capacity_503):
                    trace.emit('stopped', reason='attempt_limit', attempt_limit=attempt_limit)
                    raise
                capacity_wait_mode = True
                trace.wait(capacity_retry_delay_seconds, 'capacity_wait_mode')
                wait_with_progress(
                    capacity_retry_delay_seconds,
                    f"{label} van bi Gemini qua tai; giu nguyen tien do va thu lai",
                    update_progress=update_progress,
                    sleep_fn=sleep_fn,
                )
                continue
            delay = (
                float(GEMINI_503_RETRY_DELAYS[min(
                    attempt, len(GEMINI_503_RETRY_DELAYS) - 1
                )])
                if is_capacity_503 else
                gemini_retry_delay_seconds(exc, attempt)
            )
            remaining = delay
            trace.wait(delay, 'provider_503' if is_capacity_503 else 'transient_error')
            while remaining > 0:
                if update_progress is not None:
                    update_progress(
                        f"{label} tam gian doan; tu thu lai sau {int(math.ceil(remaining))} giay..."
                    )
                step = min(5.0, remaining)
                sleep_fn(step)
                remaining -= step
            attempt += 1


def generate_story_plan_with_model_fallback(
    client, contents, config, label, update_progress=None, stream=True,
    sleep_fn=time.sleep, usage_callback=None, quota_pacer=None,
    estimated_input_tokens=45000, model_sequence=STORY_PLAN_MODEL_SEQUENCE,
    capacity_delay_seconds=STORY_PLAN_503_RETRY_DELAY_SECONDS,
    model_callback=None,
):
    """Retry transient transport errors in place; step down models on 503."""
    models = tuple(str(model or "").strip() for model in model_sequence if str(model or "").strip())
    if not models:
        raise ValueError("Khong co model Gemini lap ke hoach.")
    last_error = None
    model_index = 0
    timeout_retries = 0
    for attempt_index in range(len(models)):
        model = models[model_index]
        if model_callback:
            model_callback(model, attempt_index)
        try:
            text = generate_gemini_text_with_recovery(
                client,
                model,
                contents,
                config,
                f"{label} [{model}]",
                update_progress=update_progress,
                stream=stream,
                max_attempts=1,
                sleep_fn=sleep_fn,
                resume_503_forever=False,
                usage_callback=usage_callback,
                quota_pacer=quota_pacer,
                estimated_input_tokens=estimated_input_tokens,
                # Capacity switching is owned here so the generic helper does
                # not retry the same overloaded model first.
                capacity_max_retries=0,
                retry_timeouts=False,
                retry_transport_errors=False,
            )
            return text, model
        except Exception as exc:
            last_error = exc
            if is_gemini_transport_timeout(exc):
                if timeout_retries >= 1:
                    raise
                timeout_retries += 1
            if not is_transient_gemini_error(exc) or attempt_index + 1 >= len(models):
                raise
            if is_gemini_capacity_503_error(exc):
                model_index = min(model_index + 1, len(models) - 1)
            next_model = models[model_index]
            retry_action = (
                f"thu lai {next_model}" if next_model == model
                else f"chuyen sang {next_model}"
            )
            wait_with_progress(
                capacity_delay_seconds,
                f"{model} tam gian doan; giu checkpoint va {retry_action}",
                update_progress=update_progress,
                sleep_fn=sleep_fn,
            )
    raise last_error


def generate_recap_with_stream_checkpoints(
    client, model, base_prompt, config, task_id, target_words,
    update_progress=None, sleep_fn=time.sleep,
):
    """Resume an interrupted recap after the last complete streamed scene."""
    checkpoints = []
    checkpoint_keys = set()
    update_task_fields(
        task_id,
        recap_stream_checkpoint_scenes=[],
        recap_stream_checkpoint_words=0,
    )

    def add_checkpoints(text_value):
        added = 0
        for scene in extract_complete_recap_scenes(text_value):
            key = (
                str(scene.get("story_beat_id", "")).strip().upper(),
                re.sub(r"\s+", " ", str(scene.get("tts", ""))).strip().casefold(),
            )
            if key in checkpoint_keys:
                continue
            checkpoint_keys.add(key)
            checkpoints.append(scene)
            added += 1
        if added:
            checkpoint_words = sum(
                len(re.findall(r"\b\w+\b", str(row.get("tts", "")), re.UNICODE))
                for row in checkpoints
            )
            update_task_fields(
                task_id,
                recap_stream_checkpoint_scenes=copy.deepcopy(checkpoints),
                recap_stream_checkpoint_words=checkpoint_words,
            )
        return added

    for attempt in range(GEMINI_503_MAX_RETRIES + 1):
        completed_words = sum(
            len(re.findall(r"\b\w+\b", str(row.get("tts", "")), re.UNICODE))
            for row in checkpoints
        )
        prompt = base_prompt
        if checkpoints:
            counts_by_anchor = {}
            for row in checkpoints:
                beat_id = str(row.get("story_beat_id", "")).strip().upper()
                counts_by_anchor[beat_id] = counts_by_anchor.get(beat_id, 0) + len(
                    re.findall(r"\b\w+\b", str(row.get("tts", "")), re.UNICODE)
                )
            tail = [{
                "story_beat_id": row.get("story_beat_id"),
                "tts": row.get("tts"),
            } for row in checkpoints[-2:]]
            prompt += f"""

TIEP TUC SAU STREAM BI GIAN DOAN:
- Da luu an toan {len(checkpoints)} scene hoan chinh, tong {completed_words} tu.
- KHONG viet lai cac cau da hoan thanh va KHONG mo dau lai cau chuyen.
- Bat dau bang cau tiep theo cua story anchor dang do; cau dang viet do o lan truoc da bi huy.
- Con khoang {max(0, int(target_words) - completed_words)} tu de dat muc tieu.
- So tu da hoan thanh theo anchor: {json.dumps(counts_by_anchor, ensure_ascii=False)}
- Hai cau cuoi da luu de noi mach: {json.dumps(tail, ensure_ascii=False)}
- Van tra JSON dung 3 parts, nhung `scenes` CHI chua cac scene MOI chua tung viet.
"""
        streamed_parts = []
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=[prompt], config=config
            ):
                chunk_text = str(getattr(chunk, "text", "") or "")
                if not chunk_text:
                    continue
                streamed_parts.append(chunk_text)
                add_checkpoints("".join(streamed_parts))
            result_text = "".join(streamed_parts).strip()
            if not result_text:
                raise RuntimeError("Gemini tra ve noi dung rong.")
            add_checkpoints(result_text)
            return result_text, checkpoints
        except Exception as exc:
            add_checkpoints("".join(streamed_parts))
            message = str(exc).lower()
            is_capacity_503 = is_gemini_capacity_503_error(exc)
            if not is_capacity_503 or attempt >= GEMINI_503_MAX_RETRIES:
                raise
            delay = float(GEMINI_503_RETRY_DELAYS[attempt])
            wait_with_progress(
                delay,
                f"Gemini viet recap bi 503; da giu {len(checkpoints)} cau va se viet tiep",
                update_progress=update_progress,
                sleep_fn=sleep_fn,
            )


def wait_with_progress(seconds, message, update_progress=None, sleep_fn=time.sleep):
    """Wait in short visible steps so a long Gemini cooldown stays observable."""
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        if update_progress is not None:
            update_progress(f"{message} ({int(math.ceil(remaining))}s)...")
        step = min(5.0, remaining)
        sleep_fn(step)
        remaining -= step


def run_with_gemini_capacity_resume(
    operation, label, update_progress=None,
    cycles=GEMINI_CAPACITY_CYCLES,
    resume_delay=GEMINI_CAPACITY_RESUME_DELAY_SECONDS,
    sleep_fn=time.sleep,
):
    """Resume the same operation after an exhausted 503 recovery cycle."""
    last_error = None
    for cycle in range(max(1, int(cycles))):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if not is_transient_gemini_error(exc) or cycle + 1 >= cycles:
                raise
            wait_with_progress(
                resume_delay,
                f"{label} van bi Gemini qua tai; giu ket qua cu va cho de tiep tuc dung batch",
                update_progress=update_progress,
                sleep_fn=sleep_fn,
            )
    raise last_error or RuntimeError(f"{label} that bai.")


def is_gemini_auth_error(error):
    """Return True for credentials/permission failures that no model can repair."""
    message = gemini_error_chain_text(error).lower()
    return any(marker in message for marker in (
        "api_key_invalid", "api key not valid", "invalid api key",
        "unauthenticated", "permission_denied", "permission denied",
    )) or re.search(r"(?:^|\D)(?:401|403)(?:\D|$)", message) is not None


def wait_for_gemini_file(
    client, file_name, max_wait_seconds=900, poll_interval=5,
    sleep_fn=time.sleep, on_wait=None,
):
    """Wait for an uploaded Gemini file; never treat a 404 as ACTIVE."""
    waited = 0
    while waited < max_wait_seconds:
        try:
            info = client.files.get(name=file_name)
            state = getattr(info, "state", "")
            state_name = str(getattr(state, "name", state) or "").upper()
            if state_name == "ACTIVE":
                return info
            if state_name == "FAILED":
                raise RuntimeError("Gemini khong the xu ly file video.")
        except Exception as exc:
            error_text = str(exc)
            if is_gemini_auth_error(exc):
                raise
            if not (
                "404" in error_text
                or "not found" in error_text.lower()
                or is_transient_gemini_error(exc)
            ):
                raise
        sleep_fn(poll_interval)
        waited += poll_interval
        if on_wait is not None:
            on_wait(waited, max_wait_seconds)
    raise RuntimeError(
        f"Qua {max_wait_seconds} giay nhung file video Gemini van chua ACTIVE."
    )


def validate_gemini_api_key(api_key):
    """Reject only values that cannot safely be placed in an HTTP header."""
    key = str(api_key or "").strip()
    if not key:
        return "API Key khong duoc bo trong."
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        return (
            "API Key co ky tu Unicode (vi du chu tieng Viet hoac dau ngoac la). "
            "Hay xoa key hien tai va dan lai key goc tu Google AI Studio."
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in key):
        return "API Key co ky tu dieu khien khong hop le. Hay dan lai key goc."
    return ""


def ascii_upload_filename(filename):
    """Convert an uploaded basename to a safe, readable ASCII filename."""
    basename = os.path.basename(str(filename or "").replace("\\", "/"))
    stem, extension = os.path.splitext(basename)
    stem = stem.replace("đ", "d").replace("Đ", "D")
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "upload"
    extension = re.sub(r"[^A-Za-z0-9.]", "", extension).lower()
    return f"{stem}{extension}"


def available_upload_path(directory, filename):
    """Return a non-existing destination so an upload never overwrites a file."""
    stem, extension = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{counter}{extension}")
        counter += 1
    return candidate


def normalize_video_file(directory, filename):
    """Rename one existing video to ASCII and return its current basename."""
    original = os.path.basename(str(filename or "").replace("\\", "/"))
    source = os.path.join(directory, original)
    if not original or not os.path.isfile(source):
        return original
    normalized = ascii_upload_filename(original)
    if normalized == original:
        return original
    target = available_upload_path(directory, normalized)
    os.replace(source, target)
    return os.path.basename(target)


def normalize_existing_video_filenames(directory):
    """Normalize every supported video already present in the tool directory."""
    normalized = []
    for filename in os.listdir(directory):
        if os.path.splitext(filename)[1].lower() not in ALLOWED_VIDEO_EXT:
            continue
        normalized.append(normalize_video_file(directory, filename))
    return sorted(normalized, key=str.casefold)


def stage_gemini_upload(file_path, staging_dir, safe_stem):
    """Expose an upload through an ASCII-only path for strict HTTP clients."""
    source = os.path.abspath(file_path)
    extension = os.path.splitext(source)[1].lower() or ".bin"
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(safe_stem or "upload")).strip("_")
    target = os.path.abspath(os.path.join(staging_dir, f"{safe_stem or 'upload'}{extension}"))
    try:
        target.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "Thu muc tam upload Gemini van chua ky tu Unicode; hay dat TEMP/TMP vao duong dan ASCII."
        ) from exc
    if os.path.normcase(source) != os.path.normcase(target):
        shutil.copy2(source, target)
    return target


def split_video_into_three_parts(source_path, output_dir, duration_sec, ffmpeg_exe, boundaries=None):
    """Create three exact, clip-local MP4 files whose durations cover the source."""
    duration_sec = max(0.01, float(duration_sec))
    boundaries = list(boundaries or [0.0, duration_sec / 3.0, duration_sec * 2.0 / 3.0, duration_sec])
    if len(boundaries) != 4 or abs(boundaries[0]) > 0.01 or abs(boundaries[-1] - duration_sec) > 1.0:
        raise ValueError("Can dung bon moc thoi gian bao phu toan video.")
    proxy_fps = analysis_proxy_fps(duration_sec)
    outputs = []
    for index in range(3):
        start = boundaries[index]
        clip_duration = boundaries[index + 1] - start
        output_path = os.path.join(output_dir, f"source_part_{index + 1}.mp4")
        result = subprocess.run(
            [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", source_path,
                "-t", f"{clip_duration:.3f}",
                "-vf", f"scale=-2:480,fps={proxy_fps}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "24000",
                "-movflags", "+faststart", output_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3600,
        )
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise RuntimeError(f"Khong the cat video Gemini phan {index + 1}.")
        outputs.append(output_path)
    return boundaries, outputs


def split_video_into_batches(
    source_path, output_dir, duration_sec, batch_count, ffmpeg_exe, prefix,
    boundaries=None,
):
    """Split video using validated story-aware or equal boundaries."""
    duration = max(0.01, float(duration_sec))
    count = max(1, int(batch_count))
    supplied = list(boundaries or [])
    if (
        len(supplied) == count + 1
        and abs(float(supplied[0])) <= 0.01
        and abs(float(supplied[-1]) - duration) <= 0.05
        and all(
            float(right) - float(left) > 0.01
            for left, right in zip(supplied, supplied[1:])
        )
    ):
        boundaries = [float(value) for value in supplied]
    else:
        boundaries = [duration * index / count for index in range(count + 1)]
    if count == 1:
        return boundaries, [source_path]
    fps = analysis_proxy_fps(duration)
    outputs = []
    for index in range(count):
        start = boundaries[index]
        clip_duration = boundaries[index + 1] - start
        output_path = os.path.join(output_dir, f"{prefix}_{index + 1}.mp4")
        result = subprocess.run(
            [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", source_path,
                "-t", f"{clip_duration:.3f}",
                "-vf", f"scale=-2:480,fps={fps}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "24000",
                "-movflags", "+faststart", output_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3600,
        )
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise RuntimeError(f"Khong the cat video Gemini batch {index + 1}.")
        outputs.append(output_path)
    return boundaries, outputs


def split_video_into_overlapping_batches(
    source_path, output_dir, duration_sec, boundaries, ffmpeg_exe, prefix,
    overlap_seconds=15.0, indexed=False,
):
    """Cut core chapters with right-side lookahead so open events can finish.

    The next clip still starts at the core boundary.  Consequently the
    lookahead is observed by both adjacent requests without adding an equally
    large left overlap or losing the original global timestamp contract.
    """
    duration = max(0.01, float(duration_sec))
    points = [float(value) for value in boundaries]
    if (
        len(points) < 2
        or abs(points[0]) > 0.01
        or abs(points[-1] - duration) > 0.05
        or any(right <= left for left, right in zip(points, points[1:]))
    ):
        raise ValueError("Ranh gioi chapter clip khong hop le.")
    overlap = max(0.0, min(3 * 60.0, float(overlap_seconds)))
    fps = analysis_proxy_fps(duration)
    windows = []
    outputs = []
    for index in range(len(points) - 1):
        start = max(0.0, points[index])
        end = min(
            duration,
            points[index + 1] + (overlap if index < len(points) - 2 else 0.0),
        )
        clip_duration = end - start
        output_path = os.path.join(output_dir, f"{prefix}_{index + 1}.mp4")
        if indexed:
            from grounded_pipeline import indexed_clip
            indexed_clip(source_path, output_path, start, end, ffmpeg_exe, fps=DETAIL_VIDEO_ANALYSIS_FPS)
            windows.append((start, end))
            outputs.append(output_path)
            continue
        result = subprocess.run(
            [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", source_path,
                "-t", f"{clip_duration:.3f}",
                "-vf", f"scale=-2:480,fps={fps}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "24000",
                "-movflags", "+faststart", output_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3600,
        )
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise RuntimeError(f"Khong the cat chapter clip Gemini {index + 1}.")
        windows.append((start, end))
        outputs.append(output_path)
    return windows, outputs


def _bg_cleanup_loop():
    """Tu dong don dep: preview TTS cu + TASKS da hoan thanh."""
    import time
    preview_dir = os.path.join(WORKSPACE_DIR, "static", "temp_preview")
    while True:
        try:
            now = time.time()
            # Don dep preview TTS cu hon 5 phut
            for filename in os.listdir(preview_dir):
                filepath = os.path.join(preview_dir, filename)
                if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > 300:
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
            # Don dep TASKS da hoan thanh/that bai hon 30 phut
            with task_lock:
                expired = [k for k, v in TASKS.items()
                          if v["status"] in ("success", "failed")
                          and now - v.get("created_at", 0) > 1800]
                for k in expired:
                    del TASKS[k]
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=_bg_cleanup_loop, daemon=True).start()

# =====================================================================
# BACKGROUND TASK RUNNERS
# =====================================================================
def normalize_simple_gemini_timeline(script_text, video_duration_sec):
    """Apply only the minimum formatting and timestamp safety needed by rendering."""
    timeline = str(script_text or "").strip()
    timeline = format_timeline_rows(timeline)
    timeline = remove_non_story_timeline_rows(timeline)
    timeline = clamp_final_timeline_end(timeline, video_duration_sec)
    timeline = repair_nonpositive_timeline_ranges(timeline, video_duration_sec)
    timeline = trim_timeline_ends_to_locked_starts(timeline)
    validate_timeline_ranges(timeline, video_duration_sec)
    if not timeline.strip():
        raise ValueError("Gemini khong tra ve dong TIME hop le.")
    return timeline.strip()


def resumable_chapter_evidence(
    checkpoint, expected_windows, video_name, target_words,
):
    """Return only a contiguous, compatible chapter checkpoint prefix."""
    try:
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        if int(checkpoint.get("evidence_contract_version") or 0) != EVIDENCE_CONTRACT_VERSION:
            return []
        evidence = checkpoint.get("evidence") or []
        windows = checkpoint.get("windows") or []
        if not isinstance(evidence, list) or not isinstance(windows, list):
            return []
        if len(windows) != len(expected_windows):
            return []
        if not all(isinstance(saved, dict) for saved in windows):
            return []
        if not all(
            abs(float(saved.get("start_seconds", -1)) - float(expected["start_seconds"])) <= 0.5
            and abs(float(saved.get("end_seconds", -1)) - float(expected["end_seconds"])) <= 0.5
            for saved, expected in zip(windows, expected_windows)
        ):
            return []
        if int(checkpoint.get("total") or 0) != len(expected_windows):
            return []
        if checkpoint.get("video_name", video_name) != video_name:
            return []
        if int(checkpoint.get("target_words", target_words)) != int(target_words):
            return []
        completed = min(
            int(checkpoint.get("completed") or 0), len(evidence), len(expected_windows)
        )
        prefix = evidence[:completed]
        if not all(
            isinstance(chunk, dict)
            and chunk.get("events")
            and not chunk.get("fallback_evidence")
            for chunk in prefix
        ):
            return []
        return copy.deepcopy(prefix)
    except (TypeError, ValueError, OverflowError):
        return []


def resumable_part_responses(
    checkpoint, boundaries, target_words, video_name, model,
    thinking_effort, style, event_ids, visual_alignment_version=0,
):
    """Return completed recap parts only when their locked plan is identical."""
    try:
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        if visual_alignment_version and checkpoint.get("visual_alignment_version") != visual_alignment_version:
            return []
        if int(checkpoint.get("evidence_contract_version") or 0) != EVIDENCE_CONTRACT_VERSION:
            return []
        responses = checkpoint.get("responses") or []
        saved_boundaries = checkpoint.get("boundaries") or []
        if not isinstance(responses, list) or len(responses) > 3:
            return []
        if not all(isinstance(value, str) and value.strip() for value in responses):
            return []
        if len(saved_boundaries) != len(boundaries) or not all(
            abs(float(saved) - float(current)) <= 0.5
            for saved, current in zip(saved_boundaries, boundaries)
        ):
            return []
        expected = {
            "target_words": int(target_words), "video_name": video_name,
            "model": model, "thinking_effort": thinking_effort, "style": style,
            "event_ids": event_ids,
        }
        for key, value in expected.items():
            saved = checkpoint.get(key)
            if key == "target_words":
                saved = int(saved or 0)
            if saved != value:
                return []
        return copy.deepcopy(responses)
    except (TypeError, ValueError, OverflowError):
        return []


def resumable_story_plan(
    checkpoint, video_name, target_words, model, thinking_effort,
    style, evidence_event_count, planning_version=0, workflow_version=0,
):
    """Return a locked story plan only when its evidence/config still match."""
    try:
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        if planning_version and checkpoint.get("planning_version") != planning_version:
            return None
        if workflow_version and checkpoint.get("workflow_version") != workflow_version:
            return None
        if int(checkpoint.get("evidence_contract_version") or 0) != EVIDENCE_CONTRACT_VERSION:
            return None
        plan = checkpoint.get("plan")
        if not isinstance(plan, dict):
            return None
        if (
            int(plan.get("editorial_contract_version", 0) or 0) != 1
            or not isinstance(plan.get("editorial_anchors"), dict)
        ):
            return None
        if any(
            str(event.get("source", "")).lower() in {
                "transcript_bridge", "transcript_anchor"
            }
            for part_events in plan.get("events_by_part", [])
            if isinstance(part_events, list)
            for event in part_events
            if isinstance(event, dict)
        ):
            return None
        expected = {
            "video_name": video_name,
            "target_words": int(target_words),
            "model": model,
            "thinking_effort": thinking_effort,
            "style": style,
            "evidence_event_count": int(evidence_event_count),
        }
        for key, value in expected.items():
            saved = checkpoint.get(key)
            if key in {"target_words", "evidence_event_count"}:
                saved = int(saved if saved is not None else -1)
            if saved != value:
                return None
        return copy.deepcopy(plan)
    except (TypeError, ValueError, OverflowError):
        return None


def _analyze_video_bg_once(
    task_id, api_key, model_name, video_filename, style, custom_prompt,
    enable_multi_voice=False, review_minutes="auto", groq_api_key="",
    thinking_effort="low", subtitle_text="", voice_rate=DEFAULT_VOICE_RATE,
):
    def update_prog(prog, msg):
        with task_lock:
            TASKS[task_id]["progress"] = prog
            TASKS[task_id]["message"] = msg

    compressed_path = ""
    silent_analysis_path = ""
    upload_staging_dir = ""
    remote_files = []
    client = None
    visual_client = None
    client_upload = None
    partial_script_lines = []
    transcript_executor = None
    transcript_future = None
    gemini_usage_totals = {
        "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0,
        "tool_tokens": 0, "total_tokens": 0,
    }
    gemini_quota_pacer = GeminiMinuteQuotaPacer()
    gemini_usage_lock = threading.Lock()

    def record_gemini_usage(label, usage):
        with gemini_usage_lock:
            record_gemini_usage_locked(label, usage)

    def record_request_diagnostic(row):
        append_task_log(task_id, 'REQUEST', 'Gemini request diagnostic',
                        json.dumps(row, ensure_ascii=False, default=str))

    record_gemini_usage.diagnostic_callback = record_request_diagnostic

    def record_gemini_usage_locked(label, usage):
        row = {key: max(0, int((usage or {}).get(key, 0) or 0)) for key in gemini_usage_totals}
        for key, value in row.items():
            gemini_usage_totals[key] += value
        update_task_fields(
            task_id,
            gemini_usage=copy.deepcopy(gemini_usage_totals),
            gemini_token_budget=GEMINI_TASK_SOFT_TOKEN_BUDGET,
            gemini_token_budget_exceeded=(
                gemini_usage_totals["total_tokens"] > GEMINI_TASK_SOFT_TOKEN_BUDGET
            ),
        )
        append_task_log(
            task_id,
            "USAGE",
            f"{label}: {row['total_tokens']} token; tong "
            f"{gemini_usage_totals['total_tokens']}/{GEMINI_TASK_SOFT_TOKEN_BUDGET}.",
        )

    def publish_partial_script(error_message=None):
        partial = "\n".join(partial_script_lines).strip()
        result = script_with_error(partial, error_message) if error_message else partial
        with task_lock:
            TASKS[task_id]["result"] = result
            TASKS[task_id]["partial_result"] = bool(error_message)
        return result

    def clock_seconds(value):
        text = str(value or "0").strip()
        try:
            pieces = [float(piece) for piece in text.split(":")]
        except ValueError:
            return 0.0
        total = 0.0
        for piece in pieces:
            total = total * 60.0 + piece
        return max(0.0, total)

    def cleanup_analysis_artifacts(preserve_local=False):
        if client is not None:
            for remote_file in list(remote_files):
                try:
                    client.files.delete(name=remote_file.name)
                except Exception:
                    pass
            remote_files.clear()
        for connection in (visual_client, client_upload, client):
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass  # Cleanup must not hide the original API error/result.
        for local_path in (compressed_path, silent_analysis_path):
            if not preserve_local and local_path and os.path.isfile(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass
        if not preserve_local and upload_staging_dir and os.path.isdir(upload_staging_dir):
            shutil.rmtree(upload_staging_dir, ignore_errors=True)
        if not preserve_local:
            update_task_fields(task_id, analysis_artifact_checkpoint={})
        if transcript_executor is not None:
            transcript_executor.shutdown(wait=False, cancel_futures=True)

    try:
        video_path = os.path.join(WORKSPACE_DIR, video_filename)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Khong tim thay video: {video_filename}")

        update_prog(10, "Dang cau hinh ket noi API Gemini...")
        from google.genai import types as genai_types
        generation_http_opts = genai_types.HttpOptions(
            # A stalled request should split into smaller work instead of
            # occupying the analysis thread for 12 minutes per attempt.
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS * 1000,
            retry_options=genai_types.HttpRetryOptions(
                # Recovery is owned by generate_gemini_text_with_recovery.
                # Keep the SDK from multiplying the same request internally.
                attempts=1,
                initial_delay=1.0,
                max_delay=30.0,
                exp_base=2.0,
                jitter=1.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        )
        client = genai.Client(api_key=api_key, http_options=generation_http_opts)
        with task_lock:
            artifact_checkpoint = copy.deepcopy(
                TASKS.get(task_id, {}).get("analysis_artifact_checkpoint") or {}
            )
        saved_staging_dir = str(artifact_checkpoint.get("staging_dir") or "")
        saved_proxy_path = str(artifact_checkpoint.get("proxy_path") or "")
        reusable_proxy = (
            artifact_checkpoint.get("video_name") == video_filename
            and saved_staging_dir and os.path.isdir(saved_staging_dir)
            and saved_proxy_path and os.path.isfile(saved_proxy_path)
        )
        upload_staging_dir = (
            saved_staging_dir if reusable_proxy
            else tempfile.mkdtemp(prefix=f"gemini_{task_id[:8]}_")
        )

        # One analysis format for every source duration. This copy preserves
        # the complete timeline; only spatial/temporal bitrate is reduced.
        update_prog(15, "Dang tao ban nen Gemini 480p voi FPS thich ung...")
        from video_renderer import get_ffmpeg_exe, get_video_info
        ffmpeg_exe = get_ffmpeg_exe()
        video_duration_sec, _, _ = get_video_info(video_path)
        subtitle_payload = parse_subtitle_text(
            subtitle_text, duration_seconds=video_duration_sec
        ) if str(subtitle_text or "").strip() else {
            "provider": "", "model": "", "segments": [],
            "removed_promotional": 0,
        }
        subtitle_segments = subtitle_payload.get("segments") or []
        if str(subtitle_text or "").strip():
            append_task_log(
                task_id,
                "TRANSCRIPT",
                f"Da doc {len(subtitle_segments)} moc SRT hop le; loai "
                f"{int(subtitle_payload.get('removed_promotional', 0))} dong quang cao.",
            )

        # Start Whisper while the reusable proxy is prepared.  Its timestamped
        # text will replace repeated Gemini audio processing in both the global
        # director pass and the detailed chapter passes.
        transcript_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="whisper"
        )
        transcript_future = transcript_executor.submit(
            transcribe_video,
            video_path,
            video_duration_sec,
            ffmpeg_exe,
            TRANSCRIPT_CACHE_DIR,
            groq_api_key,
            progress=lambda message: (
                update_task_fields(task_id, transcript_message=message),
                append_task_log(task_id, "TRANSCRIPT", message),
            ),
        )
        update_task_fields(
            task_id,
            transcript_status="running",
            visual_analysis_status="running",
        )
        
        compressed_path = (
            saved_proxy_path if reusable_proxy
            else os.path.join(upload_staging_dir, "source_proxy.mp4")
        )
        try:
            if reusable_proxy:
                upload_path = compressed_path
                update_prog(18, "Chuyen API: dung lai ban nen Gemini da tao; khong nen video lai.")
            else:
                proxy_fps = analysis_proxy_fps(video_duration_sec)
                orig_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                update_prog(12, f"Dang nen video {orig_size_mb:.0f}MB o {proxy_fps} FPS, giu nguyen timeline...")
                cmd_compress = [
                    ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path,
                    "-vf", f"scale=-2:480,fps={proxy_fps}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "24000",
                    "-movflags", "+faststart",
                    compressed_path,
                ]
                res = subprocess.run(
                    cmd_compress,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3600,
                )
                if res.returncode == 0 and os.path.exists(compressed_path):
                    compressed_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                    update_prog(18, f"Da nen Gemini: {orig_size_mb:.0f}MB -> {compressed_mb:.0f}MB")
                    upload_path = compressed_path
                    update_task_fields(
                        task_id,
                        analysis_artifact_checkpoint={
                            "video_name": video_filename,
                            "staging_dir": upload_staging_dir,
                            "proxy_path": compressed_path,
                        },
                    )
                else:
                    upload_path = video_path
        except subprocess.TimeoutExpired:
            update_prog(18, "Nen trung gian qua lau; se tai file goc len Gemini...")
            upload_path = video_path
        except Exception:
            upload_path = video_path

        transcript_payload = {}
        transcript_segments = []
        gemini_visual_path = upload_path
        try:
            update_prog(19, "Dang cho Whisper hoan tat de tao video Gemini khong audio...")
            transcript_payload = transcript_future.result()
            whisper_segments = transcript_payload.get("segments") or []
            transcript_segments = merge_preferred_subtitles(
                subtitle_segments, whisper_segments
            )
            from transcript_pipeline import filter_transcript_evidence
            transcript_segments, transcript_filter = filter_transcript_evidence(transcript_segments)
            update_task_fields(task_id, transcript_filter=transcript_filter)
            if not transcript_segments:
                raise RuntimeError("Whisper khong tra ve loi thoai co timestamp.")
            if subtitle_segments:
                transcript_payload = {
                    **transcript_payload,
                    "provider": "subtitle+" + str(
                        transcript_payload.get("provider") or "whisper"
                    ),
                    "model": "user-srt+" + str(
                        transcript_payload.get("model") or "whisper"
                    ),
                }
            silent_analysis_path = os.path.join(
                upload_staging_dir, "source_proxy_silent.mp4"
            )
            silent_result = subprocess.run(
                [
                    ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", upload_path, "-map", "0:v:0", "-c:v", "copy",
                    "-an", "-movflags", "+faststart", silent_analysis_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1800,
            )
            if silent_result.returncode != 0 or not os.path.isfile(silent_analysis_path):
                raise RuntimeError("Khong tao duoc proxy Gemini khong audio.")
            gemini_visual_path = silent_analysis_path
            update_task_fields(
                task_id,
                transcript_status="ready",
                transcript_provider=transcript_payload.get("provider"),
                transcript_model=transcript_payload.get("model"),
                transcript_segment_count=len(transcript_segments),
                gemini_audio_fallback=False,
            )
            append_task_log(
                task_id,
                "INFO",
                f"Transcript da tao {len(transcript_segments)} moc; Gemini se xem video "
                "khong audio va doc transcript theo timestamp.",
            )
        except Exception as transcript_error:
            # Preserve the old audio-backed path only when Whisper/silent-proxy
            # preparation fails. This costs more tokens but keeps automation
            # usable instead of silently analyzing a movie without dialogue.
            if subtitle_segments:
                transcript_segments = subtitle_segments
                transcript_payload = subtitle_payload
                gemini_visual_path = upload_path
                update_task_fields(
                    task_id,
                    transcript_status="subtitle_only",
                    transcript_provider="subtitle",
                    transcript_model="user-srt",
                    transcript_segment_count=len(transcript_segments),
                    gemini_audio_fallback=True,
                    transcript_error=str(transcript_error),
                )
                append_task_log(
                    task_id, "WARNING",
                    "Whisper/proxy khong audio loi; van giu SRT va audio video lam fallback.",
                    str(transcript_error),
                )
            else:
                transcript_segments = []
                gemini_visual_path = upload_path
                update_task_fields(
                    task_id,
                    transcript_status="fallback_audio",
                    gemini_audio_fallback=True,
                    transcript_error=str(transcript_error),
                )
                append_task_log(
                    task_id,
                    "WARNING",
                    "Whisper/proxy khong audio khong san sang; chi lan chay nay Gemini "
                    "se dung audio goc lam fallback.",
                    str(transcript_error),
                )

        import time

        # Auto mode uses exactly one fifth of the source duration.
        (
            target_recap_minutes,
            target_recap_words,
            min_recap_words,
            max_recap_words,
        ) = calculate_script_budget(
            video_duration_sec / 60.0,
            review_minutes=review_minutes,
            style=style,
            voice_rate=voice_rate,
        )
        budget = narration_budget(target_recap_minutes, voice_rate)
        update_task_fields(task_id, narration_budget=budget,
                           target_sentence_count=budget['target_sentence_count'],
                           planned_voice_rate=budget['voice_rate'],
                           video_analysis_fps=DETAIL_VIDEO_ANALYSIS_FPS)
        append_task_log(task_id, 'INFO',
            f"[NARRATION_BUDGET] {target_recap_minutes:.1f} phút; tốc độ {budget['voice_rate']}; "
            f"Bắt buộc đúng {budget['target_sentence_count']} câu; các câu được gom theo tình huống. "
            "Thời lượng là ước tính, sẽ đối chiếu TTS thực tế.", json.dumps(budget, ensure_ascii=False))
        depth_instruction = recap_depth_instruction(
            video_duration_sec / 60.0, target_recap_minutes
        )
        custom_prompt = "\n".join(
            value for value in (str(custom_prompt or "").strip(), depth_instruction)
            if value
        )
        base_target_recap_words = target_recap_words
        target_recap_words = expanded_word_target(base_target_recap_words)
        _, min_recap_words, max_recap_words = requested_word_range(
            target_recap_words
        )
        part_word_targets = three_part_targets(target_recap_words)
        duration_text = _format_review_time(video_duration_sec)
        with task_lock:
            TASKS[task_id]["source_duration_seconds"] = video_duration_sec
            TASKS[task_id]["source_duration_text"] = duration_text
            TASKS[task_id]["target_minutes"] = target_recap_minutes
            TASKS[task_id]["target_words"] = target_recap_words
        update_prog(
            20,
            f"Phim {duration_text}; recap {target_recap_minutes:.1f} phút, "
            f"viết theo tình huống, giọng đọc {budget['voice_rate']}.",
        )

        selected_model = resolve_selected_gemini_model(model_name)
        analysis_model = TWO_PASS_ANALYSIS_MODEL
        update_task_fields(task_id, recap_writing_model=selected_model,
                           video_analysis_model=analysis_model)
        append_task_log(task_id, 'INFO',
                        f'Model xem video: {analysis_model}; viet recap va kiem chung: {selected_model}.')
        analysis_thinking = thinking_config_for_effort(
            genai_types, analysis_model, thinking_effort
        )
        writing_thinking = thinking_config_for_effort(
            genai_types, selected_model, thinking_effort
        )
        story_plan_thinking = thinking_config_for_effort(
            genai_types, STORY_PLAN_PRIMARY_MODEL, thinking_effort
        )
        pass1_config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=16384,
            response_mime_type="application/json",
            thinking_config=analysis_thinking,
        )
        from grounded_pipeline import observation_response_schema
        observation_config = pass1_config.model_copy(update={
            'response_json_schema': observation_response_schema()})
        story_plan_config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=16384,
            response_mime_type="application/json",
            thinking_config=story_plan_thinking,
        )
        writing_config = genai_types.GenerateContentConfig(
            temperature=0.4,
            # Text-only parts can contain many scenes. Keep enough output room
            # to avoid truncated JSON; video requests still use the 16k cap.
            max_output_tokens=32768,
            response_mime_type="application/json",
            thinking_config=writing_thinking,
        )
        upload_size_mb = os.path.getsize(gemini_visual_path) / (1024 * 1024)
        upload_timeout = max(120, int(upload_size_mb * 3))
        client_upload = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=upload_timeout * 1000),
        )
        from grounded_pipeline import observation_cache_path, load_observation_cache, save_observation_cache
        grounding_cache_path = observation_cache_path(
            os.path.join(WORKSPACE_DIR, '.cache'), video_path, 0,
            EVIDENCE_CONTRACT_VERSION, subtitle_text, fps=DETAIL_VIDEO_ANALYSIS_FPS)
        cached_grounding = load_observation_cache(grounding_cache_path)
        with task_lock:
            current_grounding = copy.deepcopy(TASKS.get(task_id, {}))
        restored_raw = dict(cached_grounding.get('primary_grounding_responses') or {})
        previous_contract = (current_grounding.get('observation_contract_version') or
                             (current_grounding.get('chapter_evidence_checkpoint') or {}).get('evidence_contract_version'))
        if previous_contract == EVIDENCE_CONTRACT_VERSION:
            restored_raw.update(current_grounding.get('primary_grounding_responses') or {})
        update_task_fields(task_id, primary_grounding_responses=restored_raw,
                           observation_contract_version=EVIDENCE_CONTRACT_VERSION)
        full_video_outline = (current_grounding.get('full_video_outline')
                              if current_grounding.get('full_video_outline_complete') else None)
        if not full_video_outline and cached_grounding.get('full_video_outline_complete'):
            full_video_outline = cached_grounding.get('full_video_outline')
        full_outline_calls = 0
        if full_video_outline:
            update_task_fields(task_id, full_video_outline=copy.deepcopy(full_video_outline),
                               full_video_outline_complete=True)
            append_task_log(task_id, 'INFO', 'Dung lai outline da luu cho dung video va muc tieu; khong gui lai toan phim.')
        else:
            try:
                # Reuse the already uploaded low-resolution proxy. This pass only
                # creates a compact global story map; short chapter clips remain
                # the source of timestamped evidence and detailed actions.
                update_prog(21, "Gemini đang xem nhẹ toàn bộ phim để lập bản đồ câu chuyện...")
                overview_file = upload_gemini_file_with_recovery(
                    client_upload,
                    gemini_visual_path,
                    label="proxy toàn phim",
                    update_progress=lambda message: update_prog(21, message),
                    max_attempts=2,
                )
                remote_files.append(overview_file)
                wait_for_gemini_file(
                    client,
                    overview_file.name,
                    max_wait_seconds=900,
                    poll_interval=5,
                    sleep_fn=time.sleep,
                    on_wait=lambda waited, limit: update_prog(
                        21, f"Gemini đang đọc toàn phim... ({waited}s/{limit}s)"
                    ),
                )
                overview_content = build_full_video_gemini_part(
                    genai_types, overview_file, fps=FULL_VIDEO_ANALYSIS_FPS
                )
                overview_prompt = build_full_video_outline_prompt(
                    video_duration_sec,
                    target_recap_words,
                    style=style,
                    custom_prompt=custom_prompt,
                    transcript_text=format_transcript_window(
                        transcript_segments, 0.0, video_duration_sec,
                        max_characters=160000,
                    ),
                )
                full_outline_calls += 1
                overview_text = generate_gemini_text_with_recovery(
                    client,
                    analysis_model,
                    [overview_content, overview_prompt],
                    genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                        thinking_config=analysis_thinking,
                    ),
                    "Gemini xem toàn phim lập outline",
                    update_progress=lambda message: update_prog(21, message),
                    stream=True,
                    max_attempts=1,
                    resume_503_forever=False,
                    usage_callback=record_gemini_usage,
                    quota_pacer=gemini_quota_pacer,
                    estimated_input_tokens=estimate_gemini_video_input_tokens(
                        video_duration_sec, FULL_VIDEO_ANALYSIS_FPS
                    ),
                )
                full_video_outline = normalize_full_video_outline(
                    overview_text, video_duration_sec
                )
                outline_beats = full_video_outline.get("story_beats") or []
                if not outline_beats:
                    raise RuntimeError("Gemini khong tao duoc story beat toan phim.")
                # This is a chapter map, not the final narration scene count.
                # Long films need enough consecutive chapters to guide the second
                # video pass; sparse maps fall back to equal chapter clips.
                minimum_outline_beats = max(
                    4, math.ceil(video_duration_sec / (20 * 60))
                )
                if len(outline_beats) < minimum_outline_beats:
                    append_task_log(
                        task_id,
                        "WARNING",
                        "Ban do toan phim chi co "
                        f"{len(outline_beats)}/{minimum_outline_beats} chuong; van giu "
                        "story contract va so nhan vat, con backend se chia clip deu "
                        "de thu thap bang chung chi tiet.",
                    )
                outline_has_ending = any(
                    str(beat.get("story_role", "")).lower() == "ending"
                    or str(beat.get("narrative_function", "")).lower() == "ending"
                    for beat in outline_beats
                )
                if not outline_has_ending:
                    append_task_log(
                        task_id,
                        "WARNING",
                        "Event map chua xac dinh duoc ending; se chi xac minh ID cuoi bang reel ngan.",
                    )
                update_task_fields(
                    task_id,
                    full_video_outline=copy.deepcopy(full_video_outline),
                    full_video_outline_complete=True,
                )
                append_task_log(
                    task_id,
                    "INFO",
                    "Lượt xem nhẹ toàn phim tạo được "
                    f"{len(full_video_outline.get('story_beats') or [])} story beat "
                    f"ở {FULL_VIDEO_ANALYSIS_FPS:g} FPS.",
                )
            except Exception as overview_error:
                if (
                    is_daily_project_gemini_quota_error(overview_error)
                    or is_gemini_auth_error(overview_error)
                ):
                    raise
                full_video_outline = {}
                update_task_fields(
                    task_id,
                    full_video_outline={},
                    full_video_outline_complete=False,
                )
                append_task_log(
                    task_id,
                    "INFO",
                    "Lượt xem nhẹ toàn phim không hoàn tất; backend tự quay về "
                    "phân tích các clip chồng lấn nên tác vụ vẫn tiếp tục.",
                )

        def persist_grounding_cache():
            with task_lock:
                state = TASKS.get(task_id, {})
                saved = {key: copy.deepcopy(state.get(key)) for key in (
                    'primary_grounding_responses', 'grounding_recovery_responses',
                    'full_video_outline', 'full_video_outline_complete')}
            save_observation_cache(grounding_cache_path, saved)

        persist_grounding_cache()

        # Bounded observation clips share a global outline, rolling state and
        # source IDs. Their overlap preserves actions crossing a boundary.
        from grounded_pipeline import shot_index, observation_prompt, bind_observations, parse_observations, recover_observations
        source_shot_index = shot_index(video_duration_sec)
        evidence_chunk_count = max(1, math.ceil(video_duration_sec / (8 * 60)))
        adaptive_evidence_boundaries = [
            video_duration_sec * index / evidence_chunk_count
            for index in range(evidence_chunk_count + 1)
        ]
        evidence_windows, evidence_paths = split_video_into_overlapping_batches(
            video_path,
            upload_staging_dir,
            video_duration_sec,
            adaptive_evidence_boundaries,
            ffmpeg_exe,
            prefix="story_evidence",
            overlap_seconds=30.0, indexed=True,
        )
        append_task_log(
            task_id,
            "INFO",
            "Ranh gioi clip bang chung theo cum truyen: "
            + ", ".join(
                f"{start:.1f}-{end:.1f}s" for start, end in evidence_windows
            ),
        )
        expected_checkpoint_windows = [
            {"start_seconds": start, "end_seconds": end}
            for start, end in evidence_windows
        ]
        with task_lock:
            evidence_checkpoint = copy.deepcopy(
                TASKS.get(task_id, {}).get("chapter_evidence_checkpoint") or {}
            )
        chunk_evidence = resumable_chapter_evidence(
            evidence_checkpoint,
            expected_checkpoint_windows,
            video_filename,
            target_recap_words,
        )
        from character_identity import VERSION as CHARACTER_REGISTRY_VERSION, compact_registry, needs_identity_rebuild
        if chunk_evidence and needs_identity_rebuild(evidence_checkpoint):
            append_task_log(task_id, 'CACHE', 'Checkpoint identity rebuild', json.dumps({
                'expected_registry_version': CHARACTER_REGISTRY_VERSION,
                'saved_registry_version': evidence_checkpoint.get('character_registry_version'),
                'chunk_versions': sorted({str(c.get('character_registry_version'))
                    if isinstance(c, dict) else 'invalid_chunk' for c in chunk_evidence}),
                'discarded_normalized_chunks': len(chunk_evidence),
                'kept_raw_responses': len(restored_raw)}, ensure_ascii=False))
            chunk_evidence = []
            update_task_fields(task_id, chapter_evidence_checkpoint={}, story_plan_checkpoint={},
                grounded_plan_checkpoint={}, part_writing_checkpoint={}, evidence_writing_batches={},
                evidence_writing_candidates={}, evidence_text_audit={}, grounded_narration_audit={},
                character_registry_checkpoint={})
            append_task_log(task_id, 'INFO',
                'Khoi phuc ho so nhan vat va fact tu phan hoi goc; khong dung checkpoint da thay ten kieu cu. '
                'Uu tien raw da luu, khong xem lai clip khi raw con hop le.')
        with task_lock:
            saved_registry = copy.deepcopy(TASKS.get(task_id, {}).get('character_registry_checkpoint') or {})
        registry_outline = ({'characters': saved_registry['characters']}
            if saved_registry.get('version') == CHARACTER_REGISTRY_VERSION and saved_registry.get('characters')
            else full_video_outline)
        registry_outline = copy.deepcopy(registry_outline)
        for person in registry_outline.get('characters') or []:
            person['name_source'] = 'gemini'
        completed_chunks = len(chunk_evidence)
        evidence_contents = [None] * len(evidence_paths)
        if completed_chunks:
            append_task_log(
                task_id,
                "INFO",
                f"Chuyen API: giu {completed_chunks}/{len(evidence_paths)} doan Gemini da xong; "
                f"tiep tuc tu doan {completed_chunks + 1}.",
            )

        def ensure_evidence_content(chunk_index, progress):
            if evidence_contents[chunk_index] is not None:
                return evidence_contents[chunk_index]
            chunk_number = chunk_index + 1
            evidence_file = upload_gemini_file_with_recovery(
                client_upload,
                evidence_paths[chunk_index],
                label=f"bang chung {chunk_number}",
                update_progress=lambda message, p=progress: update_prog(p, message),
                max_attempts=3,
            )
            remote_files.append(evidence_file)
            wait_for_gemini_file(
                client,
                evidence_file.name,
                max_wait_seconds=900,
                poll_interval=5,
                sleep_fn=time.sleep,
                on_wait=lambda waited, limit, p=progress: update_prog(
                    p, f"Gemini dang xu ly doan ngan... ({waited}s/{limit}s)"
                ),
            )
            evidence_contents[chunk_index] = build_full_video_gemini_part(
                genai_types, evidence_file, fps=DETAIL_VIDEO_ANALYSIS_FPS
            )
            return evidence_contents[chunk_index]

        desired_scene_count = required_tts_scene_count(target_recap_words)
        primary_observation_calls = 0
        for chunk_index, evidence_path in enumerate(evidence_paths):
            chunk_number = chunk_index + 1
            chunk_start, chunk_end = evidence_windows[chunk_index]
            chunk_duration = chunk_end - chunk_start
            progress = 22 + int(13 * chunk_number / len(evidence_paths))
            if chunk_index < completed_chunks:
                continue
            with task_lock:
                raw_observations = copy.deepcopy(TASKS.get(task_id, {}).get('primary_grounding_responses') or {})
            evidence_text = raw_observations.get(str(chunk_number))
            if evidence_text:
                try:
                    cached_bound = bind_observations(parse_observations(evidence_text), source_shot_index,
                                                     chunk_start, chunk_end, chunk_number)
                    if not cached_bound['events']:
                        raise ValueError('cached_observations_empty')
                except (ValueError, TypeError, AttributeError):
                    append_task_log(task_id, 'WARNING',
                        f'Doan {chunk_number}: phan hoi cu khong dung duoc; chi yeu cau lai doan nay.')
                    evidence_text = None
            if chunk_index and not evidence_text:
                wait_with_progress(
                    BATCH_REQUEST_COOLDOWN_SECONDS,
                    "Dang nghi giua cac doan de tranh Gemini qua tai",
                    update_progress=lambda message, p=progress: update_prog(p, message),
                )
            update_prog(
                progress,
                (f"Dang doc lai bang chung da luu {chunk_number}/{len(evidence_paths)}..." if evidence_text else
                 f"Dang tai doan bang chung {chunk_number}/{len(evidence_paths)} len Gemini..."),
            )
            # Detection is deliberately independent from the requested recap
            # duration. A short recap must not make pass 1 overlook a causal
            # action that the global selector may later need.
            core_duration = (
                adaptive_evidence_boundaries[chunk_number]
                - adaptive_evidence_boundaries[chunk_number - 1]
            )
            outline_window = story_outline_for_window(
                full_video_outline, chunk_start, chunk_end
            )
            # Carry both confirmed names and locked neutral aliases. Unknown
            # people keep one stable descriptive label across all three parts.
            from situation_recap import character_registry, bind_character_registry
            rolling_character_registry = character_registry(registry_outline, chunk_evidence, transcript_segments=transcript_segments)
            evidence_prompt = observation_prompt(
                chunk_number, chunk_start, chunk_end,
                max(6, math.ceil(core_duration / 60.0 * 8)),
                context={"outline": outline_window, "characters": compact_registry(rolling_character_registry),
                         "non_story_ranges": (full_video_outline or {}).get('non_story_ranges') or [],
                         "previous_end_state": chunk_evidence[-1].get("end_state") if chunk_evidence else None,
                         "transcript": format_transcript_window(transcript_segments, chunk_start, chunk_end,
                                                               max_characters=20000)})
            normalized_evidence = None
            evidence_responses = []
            if not evidence_text:
                evidence_content = ensure_evidence_content(chunk_index, progress)
                primary_observation_calls += 1
                evidence_text = generate_gemini_text_with_recovery(
                    client,
                    analysis_model,
                    [evidence_content, evidence_prompt],
                    observation_config,
                    f"Gemini trich xuat bang chung doan {chunk_number}",
                    update_progress=lambda message, p=progress: update_prog(p, message),
                    stream=True,
                    # Never resend the expensive video merely to repair JSON.
                    max_attempts=1,
                    resume_503_forever=False,
                    usage_callback=record_gemini_usage,
                    quota_pacer=gemini_quota_pacer,
                    estimated_input_tokens=estimate_gemini_video_input_tokens(
                        chunk_duration, DETAIL_VIDEO_ANALYSIS_FPS
                    ),
                )
            else:
                append_task_log(task_id, 'INFO', f'Doan {chunk_number}: doc lai phan hoi goc da luu; khong upload/goi Gemini lai.')
            evidence_responses.append(evidence_text)
            with task_lock:
                raw_observations = copy.deepcopy(TASKS.get(task_id, {}).get('primary_grounding_responses') or {})
            raw_observations[str(chunk_number)] = evidence_text
            update_task_fields(task_id, primary_grounding_responses=raw_observations)
            persist_grounding_cache()  # Save the complete document before parsing.
            def save_observation_attempt(text, attempt):
                with task_lock:
                    history = copy.deepcopy(TASKS.get(task_id, {}).get('grounding_recovery_responses') or {})
                    latest = copy.deepcopy(TASKS.get(task_id, {}).get('primary_grounding_responses') or {})
                history.setdefault(str(chunk_number), []).append({'attempt': attempt, 'text': text})
                latest[str(chunk_number)] = text
                update_task_fields(task_id, grounding_recovery_responses=history,
                                   primary_grounding_responses=latest)
                persist_grounding_cache()

            def regenerate_observations(correction):
                append_task_log(task_id, 'WARNING',
                    f'Doan {chunk_number}: du lieu hoac SHOT ID khong hop le; Gemini se xem lai dung clip de doi chieu.',
                    correction)
                return generate_gemini_text_with_recovery(
                    client, analysis_model,
                    [ensure_evidence_content(chunk_index, progress), evidence_prompt + '\n' + correction],
                    observation_config, f'Gemini doi chieu lai SHOT doan {chunk_number}',
                    update_progress=lambda message, p=progress: update_prog(p, message),
                    stream=True, max_attempts=1, resume_503_forever=False,
                    usage_callback=record_gemini_usage, quota_pacer=gemini_quota_pacer,
                    estimated_input_tokens=estimate_gemini_video_input_tokens(
                        chunk_duration, DETAIL_VIDEO_ANALYSIS_FPS))

            indexed_payload = recover_observations(
                evidence_text, source_shot_index, chunk_start, chunk_end, chunk_number,
                regenerate_observations, save=save_observation_attempt)
            if indexed_payload.get('review_candidates'):
                append_task_log(task_id, 'WARNING',
                    f'Doan {chunk_number}: con hanh dong chua xac minh; '
                    'giu trong nhat ky, khong dung lam bang chung chac chan.',
                    json.dumps(indexed_payload['review_candidates'], ensure_ascii=False))
            if indexed_payload.get('_shot_id_normalizations'):
                append_task_log(task_id, 'INFO',
                    f'Doan {chunk_number}: da chuan hoa {len(indexed_payload["_shot_id_normalizations"])} SHOT ID theo cung gia tri so; van kiem tra vung clip.',
                    json.dumps(indexed_payload['_shot_id_normalizations'], ensure_ascii=False))
            if indexed_payload.get('_grounding_rejections'):
                append_task_log(task_id, 'WARNING',
                    f'Doan {chunk_number}: da loai {len(indexed_payload["_grounding_rejections"])} muc khong co SHOT ID hop le hoac ngoai cot truyen.',
                    json.dumps(indexed_payload['_grounding_rejections'][:20], ensure_ascii=False))
            indexed_payload = bind_character_registry(indexed_payload, rolling_character_registry,
                identity_shot_ids={sid for sid, shot in source_shot_index.items()
                                   if shot['start_seconds'] < chunk_end and shot['end_seconds'] > chunk_start},
                transcript_segments=[r for r in transcript_segments
                    if r['start_seconds'] < chunk_end and r['end_seconds'] > chunk_start])
            evidence_text = json.dumps(indexed_payload, ensure_ascii=False)
            evidence_responses = [evidence_text]
            try:
                normalized_evidence = normalize_chunk_evidence(
                    evidence_text, chunk_start, chunk_duration,
                    chunk_number=chunk_number,
                    primary_end_sec=adaptive_evidence_boundaries[chunk_number],
                )
                if not (normalized_evidence.get("events") or []):
                    raise ValueError(
                        f"Doan {chunk_number} khong co event; khong duoc coi "
                        "phan tich rong la thanh cong."
                    )
            except Exception as evidence_error:
                append_task_log(
                    task_id,
                    "WARNING",
                    f"Doan {chunk_number}: khong chuan hoa duoc du kien da doi chieu SHOT; "
                    "da luu phan hoi goc de chan doan; khong tu tao bang chung thay the.",
                    str(evidence_error),
                )
            if normalized_evidence is None or not normalized_evidence.get('events'):
                raise ValueError(f'GROUNDED_OBSERVATIONS_MISSING: doan {chunk_number}; '
                                 'phan hoi goc da luu, khong tu tao timestamp hay du kien thay the.')
            normalized_evidence['character_registry_version'] = CHARACTER_REGISTRY_VERSION
            chunk_evidence.append(normalized_evidence)
            persistent_registry = character_registry(registry_outline, chunk_evidence,
                                                      transcript_segments=transcript_segments)
            update_task_fields(
                task_id,
                chapter_evidence_checkpoint={
                    "character_registry_version": CHARACTER_REGISTRY_VERSION,
                    "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                    "completed": chunk_number,
                    "total": len(evidence_paths),
                    "windows": expected_checkpoint_windows,
                    "evidence": copy.deepcopy(chunk_evidence),
                    "video_name": video_filename,
                    "target_words": int(target_recap_words),
                },
            )
            from character_identity import cache_reference_frames
            try:
                persistent_registry = cache_reference_frames(video_path, persistent_registry,
                    os.path.join(WORKSPACE_DIR, '.cache', 'character_reference_frames'),
                    ffmpeg_exe, video_duration_sec,
                    sampled_clips=[(path, start, end) for path, (start, end) in zip(evidence_paths, evidence_windows)])
            except OSError as reference_error:
                append_task_log(task_id, 'WARNING', 'Chua luu duoc anh tham chieu nhan vat; giu ho so chu.',
                                str(reference_error))
            update_task_fields(task_id, character_registry_checkpoint={
                'version': CHARACTER_REGISTRY_VERSION, 'characters': persistent_registry})

        from grounded_pipeline import deduplicate_observations, supplement_observations, indexed_clip
        chunk_evidence, overlap_duplicates = deduplicate_observations(chunk_evidence)

        def extract_grounded_gap(start, end, deficit, attempt, focus=None):
            nonlocal grounded_gap_calls
            import hashlib
            gap_key = hashlib.sha256(json.dumps([start, end, focus], sort_keys=True).encode()).hexdigest()
            gap_chunk_number = 1000 + int(gap_key[:6], 16)
            with task_lock:
                gap_cache = copy.deepcopy(TASKS.get(task_id, {}).get('grounded_gap_responses') or {})
            cached_gap = gap_cache.get(gap_key)
            if cached_gap:
                bound = bind_observations(parse_observations(cached_gap), source_shot_index, start, end, gap_chunk_number)
                bound = bind_character_registry(bound, character_registry(registry_outline, chunk_evidence,
                    transcript_segments=transcript_segments),
                    identity_shot_ids={sid for sid, s in source_shot_index.items()
                        if s['start_seconds'] < end and s['end_seconds'] > start}, transcript_segments=transcript_segments)
                return normalize_chunk_evidence(json.dumps(bound, ensure_ascii=False), start, end-start,
                    chunk_number=gap_chunk_number, primary_end_sec=end)['events']
            path = os.path.join(upload_staging_dir, f'grounded_gap_{attempt}.mp4')
            indexed_clip(video_path, path, start, end, ffmpeg_exe, fps=DETAIL_VIDEO_ANALYSIS_FPS)
            remote = upload_gemini_file_with_recovery(client_upload, path, label='bo sung du kien co SHOT ID',
                update_progress=lambda message: update_prog(35, message), max_attempts=3)
            remote_files.append(remote)
            wait_for_gemini_file(client, remote.name, max_wait_seconds=900, poll_interval=5,
                sleep_fn=time.sleep, on_wait=lambda waited, limit: update_prog(
                    35, f'Gemini dang xu ly clip bo sung... ({waited}s/{limit}s)'))
            prompt = observation_prompt(1000 + attempt, start, end, deficit,
                context={'known_facts': [f['text'] for c in chunk_evidence for e in c['events']
                                        if e.get('start_seconds', 0) < end and e.get('end_seconds', 0) > start
                                        for f in e.get('grounded_facts') or []],
                         'outline': story_outline_for_window(full_video_outline, start, end),
                         'missing_plot_focus': focus,
                         'characters': compact_registry(character_registry(registry_outline, chunk_evidence,
                             transcript_segments=transcript_segments)),
                         'transcript': format_transcript_window(transcript_segments, start, end, max_characters=4000),
                         'instruction': 'Chi bo sung du kien MOI quan sat, khong lap known_facts.'})
            grounded_gap_calls += 1
            response = generate_gemini_text_with_recovery(client, analysis_model,
                [build_full_video_gemini_part(genai_types, remote, fps=DETAIL_VIDEO_ANALYSIS_FPS), prompt],
                observation_config, f'Gemini bo sung bang chung truoc ke hoach {attempt}/2',
                update_progress=lambda message: update_prog(35, message), stream=True, max_attempts=1,
                usage_callback=record_gemini_usage, quota_pacer=gemini_quota_pacer,
                estimated_input_tokens=estimate_gemini_video_input_tokens(end-start, DETAIL_VIDEO_ANALYSIS_FPS))
            with task_lock:
                gap_responses = copy.deepcopy(TASKS.get(task_id, {}).get('grounded_gap_responses') or {})
            gap_responses[gap_key] = response
            update_task_fields(task_id, grounded_gap_responses=gap_responses)
            bound = bind_observations(parse_observations(response), source_shot_index, start, end, gap_chunk_number)
            bound = bind_character_registry(bound, character_registry(registry_outline, chunk_evidence,
                transcript_segments=transcript_segments),
                identity_shot_ids={sid for sid, s in source_shot_index.items()
                    if s['start_seconds'] < end and s['end_seconds'] > start}, transcript_segments=transcript_segments)
            return normalize_chunk_evidence(json.dumps(bound, ensure_ascii=False), start, end-start,
                chunk_number=gap_chunk_number, primary_end_sec=end)['events']

        # Duration is a prose budget, never a minimum number of visual facts.
        grounded_gap_calls = 0
        from plot_coverage import plot_coverage, missing_plot_windows, merge_plot_events
        with task_lock:
            plot_checkpoint = copy.deepcopy(TASKS.get(task_id, {}).get('plot_coverage_checkpoint') or {})
        reviewed_plot_ids = list(plot_checkpoint.get('reviewed_ids') or [])
        supplemental_events = list(plot_checkpoint.get('supplemental_events') or [])
        merge_plot_events(chunk_evidence, supplemental_events)
        plot_report = plot_coverage(full_video_outline, chunk_evidence, video_duration_sec, reviewed_plot_ids)
        plot_windows = missing_plot_windows(plot_report, budget=MAX_TARGETED_VISUAL_REVIEWS, max_seconds=60)
        for plot_index, window in enumerate(plot_windows, 1):
            append_task_log(task_id, 'COVERAGE', 'Thieu nguon cho mat xich chinh; xem dung cua so ngan.',
                            json.dumps(window, ensure_ascii=False))
            extra_events = extract_grounded_gap(window['start_seconds'], window['end_seconds'], 1, plot_index,
                                                focus={'beat_id': window['beat_id'], 'summary': window['summary']})
            merge_plot_events(chunk_evidence, extra_events)
            supplemental_events.extend(extra_events)
            reviewed_plot_ids.append(window['beat_id'])
            plot_report = plot_coverage(full_video_outline, chunk_evidence, video_duration_sec, reviewed_plot_ids)
            update_task_fields(task_id, plot_coverage_checkpoint={
                'reviewed_ids': reviewed_plot_ids, 'supplemental_events': supplemental_events, 'report': plot_report})
        if plot_windows:
            chunk_evidence, _ = deduplicate_observations(chunk_evidence)
        plot_report = plot_coverage(full_video_outline, chunk_evidence, video_duration_sec, reviewed_plot_ids)
        update_task_fields(task_id, plot_coverage_checkpoint={'reviewed_ids': reviewed_plot_ids,
            'supplemental_events': supplemental_events, 'report': plot_report})
        append_task_log(task_id, 'COVERAGE', 'Doi chieu cot truyen truoc khi lap ke hoach.',
                        json.dumps(plot_report, ensure_ascii=False))
        grounded_fact_count = sum(len(e.get('grounded_facts') or [])
                                  for c in chunk_evidence for e in c['events'])
        update_task_fields(task_id, grounded_fact_count=grounded_fact_count,
                           narration_unit='situation')
        append_task_log(task_id, 'INFO',
            f'Da thu thap {grounded_fact_count} fact; viet theo tinh huong, khong bo sung de du quota cau.')
        if overlap_duplicates:
            append_task_log(
                task_id, "INFO",
                f"Da gom {overlap_duplicates} fact trung hoac chung vung hinh; khong cat deu shot de tao them cau.",
            )
        chunk_evidence, stitched_boundaries, unresolved_boundaries = (
            stitch_cross_chunk_event_continuations(chunk_evidence)
        )
        append_task_log(
            task_id,
            "INFO" if not unresolved_boundaries else "WARNING",
            f"Da noi {stitched_boundaries} chuoi su kien xuyen bien; "
            f"con {len(unresolved_boundaries)} su kien dang do can kiem tra lai.",
            json.dumps(unresolved_boundaries[:30], ensure_ascii=False),
        )
        # Exact observation/shot identity was already deduplicated. Broad prose
        # similarity must not erase distinct observed facts or their proof IDs.
        semantic_duplicates = 0
        from evidence_granularity import separate_event_granularity
        chunk_evidence, granularity_report = separate_event_granularity(chunk_evidence, preserve_unlocated_events=True)
        append_task_log(
            task_id, "INFO", "Da tach tong quan khoi su kien dung de phan bo cau.",
            json.dumps(granularity_report, ensure_ascii=False),
        )
        if semantic_duplicates:
            append_task_log(
                task_id, "INFO",
                f"Backend da gom {semantic_duplicates} event dien dat lai cung mot fact; "
                "cac shot van duoc giu lam visual_sources.",
            )
        chunk_evidence, ending_promoted = ensure_grounded_terminal_ending(
            chunk_evidence, video_duration_sec
        )
        if ending_promoted:
            append_task_log(
                task_id, "INFO",
                "Backend xac nhan event resolution cuoi la ending dua tren timestamp va trang thai ket thuc.",
            )

        # Capacity is based on observed fact+shot units, before any prose.
        evidence_replenishment_calls = grounded_gap_calls
        for chunk in chunk_evidence:
            for event in chunk.get('events') or []:
                event['sentence_first'] = True


        append_task_log(task_id, "INFO", "Da giu fact kem SHOT ID nguon; se khoa hinh cho tung o truoc khi viet.")
        update_task_fields(
            task_id,
            chapter_evidence_checkpoint={
                "character_registry_version": CHARACTER_REGISTRY_VERSION,
                "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                "completed": len(evidence_paths),
                "total": len(evidence_paths),
                "windows": expected_checkpoint_windows,
                "evidence": copy.deepcopy(chunk_evidence),
                "video_name": video_filename,
                "target_words": int(target_recap_words),
            },
        )

        coverage = assess_story_evidence_coverage(
            chunk_evidence, target_recap_words, video_duration_sec
        )
        append_task_log(
            task_id, "INFO",
            "Kiem tra bang chung: " + json.dumps(coverage, ensure_ascii=False),
        )
        if coverage["capacity_deficit"] > 0:
            append_task_log(
                task_id, "WARNING",
                f"Uoc luong {coverage['narration_capacity']} y ban dau cho muc tieu "
                f"{coverage['required_capacity']} cau; chuyen sang lap ke hoach phan bo DUNG muc tieu. "
                "Khong dung truoc ke hoach vi uoc luong so fact; moi o van phai co y rieng va ID nguon.",
            )
        if not coverage["has_terminal_ending"]:
            append_task_log(
                task_id,
                "WARNING",
                "Chua xac minh duoc ending tu cac luot xem chinh; giu trang thai chua ro, "
                "khong mo lai chapter hay ep chon su kien cuoi lam ending.",
            )

        # Identity is deterministic: explicit high-confidence names from the
        # video chunks are immutable, and unknown people stay unknown. A
        # text-only Gemini audit previously had enough authority to rename
        # Doraemon as the villain, so that call is deliberately removed. The
        # backend still creates objective video-review windows for real time
        # gaps; those reviews may add missed events but cannot rename a locked
        # character.
        update_prog(35, "Backend dang khoa danh tinh va kiem tra khoang trong...")
        from situation_recap import character_registry, registry_audit_payload
        audit = normalize_global_evidence_audit(
            json.dumps(registry_audit_payload(character_registry(registry_outline, chunk_evidence, transcript_segments=transcript_segments),
                                             chunk_evidence), ensure_ascii=False),
            chunk_evidence,
            video_duration_sec,
        )
        audit = add_objective_evidence_gap_windows(
            audit, chunk_evidence, video_duration_sec
        )
        evidence_audit_calls = 0
        append_task_log(
            task_id,
            "INFO",
            "Backend khoa "
            f"{len(audit.get('canonical_characters') or [])} nhan vat tu ten "
            "da xac nhan; khong goi Gemini de suy doan lai danh tinh.",
        )

        verification_texts = []
        # Keep the visual budget at the three primary uploads. Boundary overlap
        # plus Whisper handle uncertainty; reopening extra reels here previously
        # multiplied quota use and made long jobs unpredictable.
        review_candidates = []
        review_windows = []
        if not coverage["has_terminal_ending"]:
            ending_candidates = [
                row for row in review_candidates
                if "ending" in str(row.get("reason", "")).lower()
                or float(row.get("end_seconds", 0.0))
                >= video_duration_sec * 0.9
            ]
            if ending_candidates:
                review_windows.append(ending_candidates[-1])
        if unresolved_boundaries and len(review_windows) < MAX_TARGETED_VISUAL_REVIEWS:
            unresolved_ids = {str(value).upper() for value in unresolved_boundaries}
            boundary_candidate = next((
                row for row in review_candidates
                if unresolved_ids & {
                    str(value).upper() for value in (row.get("evidence_ids") or [])
                }
            ), None)
            if boundary_candidate is not None and boundary_candidate not in review_windows:
                review_windows.append(boundary_candidate)
        review_windows = review_windows[:max(0, MAX_TARGETED_VISUAL_REVIEWS - evidence_replenishment_calls)]
        for review_index, window in enumerate(review_windows, start=1):
            start = float(window["start_seconds"])
            end = float(window["end_seconds"])
            review_path = os.path.join(
                upload_staging_dir, f"evidence_review_{review_index:02d}.mp4"
            )
            review_result = subprocess.run([
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", upload_path,
                "-t", f"{max(0.1, end - start):.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart",
                review_path,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if review_result.returncode != 0 or not os.path.isfile(review_path):
                append_task_log(
                    task_id, "WARNING",
                    f"Khong cat duoc cua so kiem chung {start:.1f}-{end:.1f}s; "
                    "giu danh tinh o trang thai chua xac dinh.",
                )
                continue
            progress = 35 + int(3 * review_index / max(1, len(review_windows)))
            update_prog(
                progress,
                f"Dang xem lai diem mau thuan {review_index}/{len(review_windows)}...",
            )
            review_file = upload_gemini_file_with_recovery(
                client_upload,
                review_path,
                label=f"kiem chung {review_index}",
                update_progress=lambda message, p=progress: update_prog(p, message),
                max_attempts=3,
            )
            remote_files.append(review_file)
            wait_for_gemini_file(
                client,
                review_file.name,
                max_wait_seconds=900,
                poll_interval=5,
                sleep_fn=time.sleep,
                on_wait=lambda waited, limit, p=progress: update_prog(
                    p, f"Gemini dang xem lai diem mau thuan... ({waited}s/{limit}s)"
                ),
            )
            review_content = build_full_video_gemini_part(
                genai_types, review_file, fps=DETAIL_VIDEO_ANALYSIS_FPS
            )
            review_prompt = build_evidence_review_prompt(
                window, audit, chunk_evidence
            )
            review_text = generate_gemini_text_with_recovery(
                client,
                analysis_model,
                [review_content, review_prompt],
                pass1_config,
                f"Gemini xac minh mau thuan {review_index}",
                update_progress=lambda message, p=progress: update_prog(p, message),
                stream=True,
                max_attempts=1,
                resume_503_forever=False,
                usage_callback=record_gemini_usage,
                quota_pacer=gemini_quota_pacer,
                estimated_input_tokens=estimate_gemini_video_input_tokens(
                    end - start, DETAIL_VIDEO_ANALYSIS_FPS
                ),
            )
            review_payload = parse_json_object(review_text)
            review_payload["_review_window"] = {
                "start_seconds": start,
                "end_seconds": end,
                "allow_missing_events": bool(window.get("objective_gap")),
            }
            verification_texts.append(json.dumps(
                review_payload, ensure_ascii=False
            ))
            evidence_audit_calls += 1

        remaining_review_budget = max(
            0, MAX_TARGETED_VISUAL_REVIEWS - evidence_replenishment_calls - len(review_windows)
        )
        identity_comparisons = (audit.get("identity_comparisons") or [])[
            :remaining_review_budget
        ]
        for comparison_index, comparison in enumerate(identity_comparisons, start=1):
            comparison_clips = []
            for interval_index, (start_key, end_key) in enumerate((
                ("first_start_seconds", "first_end_seconds"),
                ("second_start_seconds", "second_end_seconds"),
            ), start=1):
                start = float(comparison[start_key])
                end = float(comparison[end_key])
                clip_path = os.path.join(
                    upload_staging_dir,
                    f"identity_{comparison_index:02d}_{interval_index}.mp4",
                )
                clip_result = subprocess.run([
                    ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start:.3f}", "-i", upload_path,
                    "-t", f"{max(0.1, end - start):.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart",
                    clip_path,
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if clip_result.returncode != 0 or not os.path.isfile(clip_path):
                    comparison_clips = []
                    break
                comparison_clips.append(clip_path)
            if len(comparison_clips) != 2:
                append_task_log(
                    task_id, "WARNING",
                    f"Khong tao duoc reel doi chieu nhan vat {comparison_index}; "
                    "giu danh tinh chua xac dinh.",
                )
                continue
            concat_manifest = os.path.join(
                upload_staging_dir, f"identity_{comparison_index:02d}_concat.txt"
            )
            with open(concat_manifest, "w", encoding="utf-8", newline="\n") as manifest:
                for clip_path in comparison_clips:
                    safe_path = os.path.abspath(clip_path).replace(chr(92), "/")
                    manifest.write(f"file '{safe_path}'\n")
            reel_path = os.path.join(
                upload_staging_dir, f"identity_compare_{comparison_index:02d}.mp4"
            )
            concat_result = subprocess.run([
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concat_manifest,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart", reel_path,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if concat_result.returncode != 0 or not os.path.isfile(reel_path):
                append_task_log(
                    task_id, "WARNING",
                    f"Khong ghep duoc reel doi chieu nhan vat {comparison_index}; "
                    "giu danh tinh chua xac dinh.",
                )
                continue
            progress = 38
            update_prog(
                progress,
                f"Dang doi chieu nhan vat {comparison_index}/{len(identity_comparisons)}...",
            )
            comparison_file = upload_gemini_file_with_recovery(
                client_upload,
                reel_path,
                label=f"doi chieu nhan vat {comparison_index}",
                update_progress=lambda message, p=progress: update_prog(p, message),
                max_attempts=3,
            )
            remote_files.append(comparison_file)
            wait_for_gemini_file(
                client,
                comparison_file.name,
                max_wait_seconds=900,
                poll_interval=5,
                sleep_fn=time.sleep,
                on_wait=lambda waited, limit, p=progress: update_prog(
                    p, f"Gemini dang doi chieu nhan vat... ({waited}s/{limit}s)"
                ),
            )
            comparison_content = build_full_video_gemini_part(
                genai_types, comparison_file, fps=DETAIL_VIDEO_ANALYSIS_FPS
            )
            comparison_prompt = build_identity_comparison_prompt(
                comparison, audit, chunk_evidence
            )
            verification_texts.append(generate_gemini_text_with_recovery(
                client,
                analysis_model,
                [comparison_content, comparison_prompt],
                pass1_config,
                f"Gemini doi chieu nhan vat {comparison_index}",
                update_progress=lambda message, p=progress: update_prog(p, message),
                stream=True,
                max_attempts=1,
                resume_503_forever=False,
                usage_callback=record_gemini_usage,
                quota_pacer=gemini_quota_pacer,
                estimated_input_tokens=estimate_gemini_video_input_tokens(
                    sum(
                        max(0.1, float(comparison[end_key]) - float(comparison[start_key]))
                        for start_key, end_key in (
                            ("first_start_seconds", "first_end_seconds"),
                            ("second_start_seconds", "second_end_seconds"),
                        )
                    ),
                    DETAIL_VIDEO_ANALYSIS_FPS,
                ),
            ))
            evidence_audit_calls += 1

        chunk_evidence = apply_verified_evidence_results(
            chunk_evidence, audit, verification_texts
        )
        chunk_evidence, _ = ensure_grounded_terminal_ending(
            chunk_evidence, video_duration_sec
        )
        coverage_after_review = assess_story_evidence_coverage(
            chunk_evidence, target_recap_words, video_duration_sec
        )
        if not coverage_after_review["has_terminal_ending"]:
            append_task_log(
                task_id,
                "WARNING",
                "Ending van chua chac chan; day la muc uu tien thap, khong xem lai rieng de ep co ket cuc. "
                "Chi ke ngan gon ket qua da co bang chung.",
            )
        unresolved_count = sum(
            1 for chunk in chunk_evidence
            for character in (chunk.get("characters") or [])
            if character.get("identity_status") != "confirmed"
        )
        append_task_log(
            task_id,
            "INFO",
            f"Kiem dinh toan phim: {len(audit.get('canonical_characters') or [])} "
            f"nhan vat chuan, {len(audit.get('contradictions') or [])} mau thuan, "
            f"{len(review_windows)} cua so xem lai, {unresolved_count} danh tinh chua chac chan.",
        )

        append_task_log(task_id, "INFO", "Dung lai shot tu luot xem chinh; khong kiem chung lai hang loat canh.")
        update_task_fields(task_id, visual_analysis_status="ready")
        update_prog(49, "Da xem xong hinh anh; dang hop nhat transcript vao evidence...")
        chunk_evidence, transcript_links = attach_transcript_to_evidence(
            chunk_evidence, transcript_segments
        )
        update_task_fields(
            task_id,
            transcript_status="ready",
            transcript_provider=transcript_payload.get("provider"),
            transcript_model=transcript_payload.get("model"),
            transcript_segment_count=len(transcript_segments),
            transcript_evidence_links=transcript_links,
            transcript_bridge_events=0,
            transcript_target_events=0,
        )
        append_task_log(
            task_id,
            "INFO",
            f"Da gan {len(transcript_segments)} doan phu de/Whisper vao "
            f"{transcript_links} lien ket voi su kien hinh; "
            "khong tao them su kien tu loi thoai.",
        )

        chunk_evidence, granularity_report = separate_event_granularity(chunk_evidence, preserve_unlocated_events=True)
        update_task_fields(task_id, event_granularity_report=granularity_report)
        append_task_log(
            task_id, "INFO", "Su kien tong quat chi lam context; muc chua ro duoc giu de kiem tra, khong phan bo cau.",
            json.dumps(granularity_report, ensure_ascii=False),
        )
        with task_lock:
            saved_story_checkpoint = copy.deepcopy(
                TASKS.get(task_id, {}).get("story_plan_checkpoint") or {}
            )
        evidence_event_count = sum(
            len(chunk.get("events") or []) for chunk in chunk_evidence
            if isinstance(chunk, dict)
        )
        story_plan = resumable_story_plan(
            saved_story_checkpoint,
            video_filename,
            target_recap_words,
            STORY_PLAN_PRIMARY_MODEL,
            thinking_effort,
            style,
            evidence_event_count,
            planning_version=2,
            workflow_version=9,
        )
        story_plan_model_used = str(
            saved_story_checkpoint.get("generated_model")
            or STORY_PLAN_PRIMARY_MODEL
        )
        if story_plan is not None:
            append_task_log(
                task_id,
                "INFO",
                "Da nap lai ke hoach toan phim; tiep tuc viet cac nhom nho.",
            )
        else:
            append_task_log(
                task_id,
                "INFO",
                f"Backend sap {evidence_event_count} moc va khoa fact+shot. "
                "Gom theo tinh huong va giu nguyen nhom hinh truoc khi viet doan ke.",
            )
        pass1_calls = (
            primary_observation_calls + evidence_replenishment_calls
            + evidence_audit_calls + full_outline_calls
        )
        if story_plan is None:
            from situation_recap import build_situation_plan
            story_plan = build_situation_plan(chunk_evidence, full_video_outline,
                                               video_duration_sec, target_recap_words, single_timeline=True)
            story_plan_model_used = 'grounded_situation_plan_v1'
            append_task_log(task_id, 'INFO',
                'Lap ke hoach theo tinh huong thuc; khong chia thanh quota cau theo so phut.')

        from global_writing import global_writing_plan
        story_plan = global_writing_plan(story_plan)
        story_plan['plot_coverage'] = plot_report
        update_task_fields(
            task_id,
            story_plan_checkpoint={
                "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                "planning_version": 2,
                "workflow_version": 9,
                "plan": copy.deepcopy(story_plan),
                "video_name": video_filename,
                "target_words": int(target_recap_words),
                "model": STORY_PLAN_PRIMARY_MODEL,
                "generated_model": story_plan_model_used,
                "thinking_effort": thinking_effort,
                "style": style,
                "evidence_event_count": evidence_event_count,
            },
            story_plan_model_used=story_plan_model_used,
        )
        story_plan['characters'] = character_registry(registry_outline, chunk_evidence,
                                                     transcript_segments=transcript_segments)
        from grounded_pipeline import lock_plan_visuals
        from narrative_contract import allocate_sentences
        story_plan = allocate_sentences(story_plan, budget['target_sentence_count'], target_recap_words)
        update_task_fields(task_id, sentence_contract=copy.deepcopy(story_plan['sentence_contract']))
        story_plan = lock_plan_visuals(story_plan, chunk_evidence)
        from evidence_writer import build_sentence_plan
        story_plan = build_sentence_plan(story_plan)
        if story_plan.get('visual_order_repaired'):
            append_task_log(task_id, 'WARNING',
                'Da sap nhom hinh theo thu tu nguon truoc khi viet doan ke.')
        with task_lock:
            grounded_story_checkpoint = copy.deepcopy(TASKS.get(task_id, {}).get('story_plan_checkpoint') or {})
        grounded_story_checkpoint['plan'] = copy.deepcopy(story_plan)
        update_task_fields(task_id, story_plan_checkpoint=grounded_story_checkpoint,
                           grounded_plan_checkpoint=copy.deepcopy(story_plan))
        append_task_log(task_id, 'INFO', 'Da khoa cac tinh huong va nhom hinh nguon; moi doan ke duoc dung nhieu canh.')
        boundaries = story_plan["boundaries"]
        converted_timestamp_parts = [
            index + 1
            for index, mode in enumerate(story_plan.get("timestamp_modes", []))
            if mode == "local_converted"
        ]
        if converted_timestamp_parts:
            print(
                "[i] Da tu chuyen timestamp cuc bo sang toan cuc cho phan: "
                + ", ".join(str(number) for number in converted_timestamp_parts),
                flush=True,
            )
        reassigned_timestamp_parts = [
            index + 1
            for index, mode in enumerate(story_plan.get("timestamp_modes", []))
            if mode == "global_reassigned"
        ]
        if reassigned_timestamp_parts:
            print(
                "[i] Da tu chuyen event ve dung phan theo timestamp toan cuc; "
                "phan tra sai: "
                + ", ".join(str(number) for number in reassigned_timestamp_parts),
                flush=True,
            )
        scene_contract = story_plan.get("scene_contract") or {}
        if scene_contract.get("ending_inferred"):
            append_task_log(
                task_id,
                "WARNING",
                "Gemini khong gan nhan ending; backend da dung moc resolution/"
                "su kien hoan chinh cuoi timeline lam diem ket thuc, khong bia them canh.",
            )
        part_scene_counts = list(scene_contract.get("part_scene_counts") or [
            len(events) for events in story_plan.get("events_by_part", [])
        ])
        total_scene_count = int(
            scene_contract.get("scene_count") or sum(part_scene_counts)
        )
        if total_scene_count != desired_scene_count and not story_plan.get('situation_plan_version'):
            raise ValueError(f'EXACT_SENTENCE_BUDGET_MISMATCH: yêu cầu {desired_scene_count}, '
                             f'kế hoạch có {total_scene_count}; không tự giảm số câu.')
        locked_target_words = int(
            scene_contract.get("target_words") or target_recap_words
        )
        part_word_targets = list(scene_contract.get("part_word_targets") or [])
        writing_group_count = len(story_plan['events_by_part'])
        if len(part_word_targets) != writing_group_count:
            part_word_targets = proportional_part_targets(
                part_scene_counts, target_recap_words
            )
        append_task_log(
            task_id,
            "INFO",
            f"Backend da khoa {total_scene_count} scene / {locked_target_words} tu; "
            f"phan bo scene {part_scene_counts}, tu {part_word_targets}.",
        )
        source_event_count = int(scene_contract.get("source_event_count") or 0)
        covered_source_count = int(
            scene_contract.get("covered_source_event_count") or 0
        )
        append_task_log(
            task_id,
            "INFO",
            f"Do phu event sau rut gon: {covered_source_count}/{source_event_count}; "
            f"da nen {int(scene_contract.get('budget_events_collapsed') or 0)} "
            "event lien ke vao cum TTS ma khong xoa timestamp nguon.",
        )
        if locked_target_words < int(target_recap_words):
            append_task_log(
                task_id,
                "WARNING",
                f"Video chi co du fact/hinh doc lap cho {locked_target_words} tu; "
                f"giam tu muc tieu {target_recap_words} de tranh lap y va dong khung.",
            )
        update_task_fields(
            task_id,
            locked_scene_contract=copy.deepcopy(scene_contract),
        )
        update_prog(42, "Da khoa fact+shot va ke hoach cau; chuan bi viet bang van ban...")
        plan_event_ids = [
            str(event.get("event_id", "")).strip().upper()
            for events in (story_plan.get("events_by_part") or [])
            for event in (events or []) if isinstance(event, dict)
            and str(event.get("event_id", "")).strip()
        ]
        with task_lock:
            saved_part_checkpoint = copy.deepcopy(
                TASKS.get(task_id, {}).get("part_writing_checkpoint") or {}
            )
        part_responses = resumable_part_responses(
            saved_part_checkpoint,
            boundaries,
            target_recap_words,
            video_filename,
            selected_model,
            thinking_effort,
            style,
            plan_event_ids,
            visual_alignment_version=6,
        )
        from evidence_writer import has_direct_dialogue
        for part_index, saved_part in enumerate(part_responses):
            try:
                saved_payload = json.loads(saved_part) if isinstance(saved_part, str) else saved_part
                saved_scenes = [scene for part in saved_payload.get('parts') or []
                                for scene in part.get('scenes') or []]
            except (ValueError, TypeError, AttributeError):
                part_responses = part_responses[:part_index]
                append_task_log(task_id, 'WARNING',
                    f'Checkpoint phan {part_index + 1} khong doc duoc; dung lai bang chung de viet tiep.')
                break
            if any(has_direct_dialogue(scene.get('tts')) for scene in saved_scenes):
                part_responses = part_responses[:part_index]
                append_task_log(task_id, 'INFO',
                    f'Phan {part_index + 1} cu con trich hoi thoai; viet lai bang loi nguoi ke tu bang chung da luu.')
                break
        if part_responses:
            append_task_log(
                task_id,
                "INFO",
                "Da khoi phuc recap toan phim da viet; "
                "tiep tuc cac nhom con lai.",
            )
        from grounded_pipeline import audit_narrations, apply_local_visual_fallbacks
        with task_lock:
            grounded_audit_cache = copy.deepcopy(TASKS.get(task_id, {}).get('grounded_narration_audit') or {})

        from recap_visual_media import build_visual_pack, visual_batches, evidence_writing_batches
        visual_client = genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(
            timeout=120000, retry_options=genai_types.HttpRetryOptions(attempts=1)))
        visual_pack_contents = {}
        visual_pack_lock = threading.Lock()
        visual_call_lock = threading.Lock()

        def ensure_visual_pack(cards, source_offset, label):
            # Protect upload/cache mutation while independent Gemini checks run
            # concurrently. The API request itself runs outside this lock.
            with visual_pack_lock:
                return prepare_visual_pack(cards, source_offset, label)

        def prepare_visual_pack(cards, source_offset, label):
            update_prog(82 if 'kiem' in label.lower() else 48, 'Dang chuan bi hinh: ' + label)
            path, manifest = build_visual_pack(
                video_path, os.path.join(WORKSPACE_DIR, '.cache', 'recap_visual_media'),
                cards, ffmpeg=ffmpeg_exe, source_offset=source_offset)
            if any(row['source_end'] > video_duration_sec + .05 for row in manifest):
                raise ValueError('VISUAL_PACK_OUTSIDE_SOURCE')
            if path not in visual_pack_contents:
                remote = upload_gemini_file_with_recovery(
                    client_upload, path, label=label,
                    update_progress=lambda message: update_prog(80, message), max_attempts=1)
                remote_files.append(remote)
                wait_for_gemini_file(client, remote.name, max_wait_seconds=120, poll_interval=5,
                                     sleep_fn=time.sleep,
                                     on_wait=lambda waited, limit: update_prog(80, f'{label}: {waited}/{limit}s'))
                visual_pack_contents[path] = build_full_video_gemini_part(genai_types, remote, fps=2.0)
            return visual_pack_contents[path], manifest, path

        def audit_grounded_payload(payload, source_offset=0.0, cache_only=False, stage='part_verification'):
            from evidence_writer import has_direct_dialogue
            from source_grounded_writer import is_source_grounded
            source_scenes = [s for p in payload.get('parts') or [] for s in p.get('scenes') or []]
            if any(has_direct_dialogue(s.get('tts')) for s in source_scenes):
                raise ValueError('RECAP_DIRECT_DIALOGUE: chi dung loi nguoi ke, khong trich Whisper')
            if source_scenes and all(s.get('source_grounded_writer_version') for s in source_scenes):
                if not all(is_source_grounded(s) for s in source_scenes):
                    raise ValueError('SOURCE_WRITER_LOCK_CHANGED: loi/hinh da doi sau khi chot')
                if not any(s.get('evidence_sentence_plan_version') for s in source_scenes):
                    checked = copy.deepcopy(payload)
                    for part in checked['parts']:
                        for scene in part.get('scenes') or []:
                            scene.update(grounding_audit_pending=False, grounding_audit_passed=True,
                                         grounding_audit_method='source_grounded_generation')
                    append_task_log(task_id, 'INFO', 'Loi da viet truc tiep voi hinh va bang chung; backend da kiem ID, quota va lien ket. Khong goi kiem chung video lai toan bai.')
                    return checked
            from visual_audit_diagnostics import parse_audit_response, inspect_audit_response, summarize_audit_scenes
            audit_id = uuid.uuid4().hex[:12]
            audit_context = {
                'audit_id': audit_id, 'model': selected_model, 'fps': 2.0,
                'source_offset_seconds': source_offset, 'cache_only': cache_only,
                'stage': stage,
                'parts': [p.get('part') for p in payload.get('parts') or []],
            }
            def diagnostic(event, data):
                record = dict(audit_context, event=event, **data)
                labels = {
                    'AUDIT_CACHE_STATUS': 'Kiểm tra cache xác minh lời–hình',
                    'AUDIT_BATCH_RESULT': 'Kết quả ID và định dạng phản hồi kiểm chứng',
                    'AUDIT_PENDING_AFTER_ATTEMPT': 'Các câu chưa có phản hồi hợp lệ sau lượt kiểm chứng',
                    'AUDIT_SCENE_UNRESOLVED': 'Chi tiết câu chưa xác minh hoặc bị bác bỏ',
                    'AUDIT_CALLBACK_ERROR': 'Lỗi xử lý phản hồi kiểm chứng',
                    'AUDIT_REQUEST_START': 'Bắt đầu request kiểm chứng hình',
                    'AUDIT_RESPONSE_RAW': 'Phản hồi gốc Gemini kiểm chứng hình',
                    'AUDIT_REQUEST_RESULT': 'Đối chiếu ID mong đợi và ID Gemini trả về',
                    'AUDIT_REQUEST_ERROR': 'Lỗi request kiểm chứng hình',
                    'AUDIT_STATUS_TRANSITION': 'Thay đổi trạng thái xác minh sau xử lý lời–hình',
                }
                issue = (event in ('AUDIT_CALLBACK_ERROR', 'AUDIT_REQUEST_ERROR', 'AUDIT_SCENE_UNRESOLVED')
                         or bool(data.get('unresolved') or data.get('rejected') or data.get('schema_error')))
                append_task_log(task_id, 'WARNING' if issue else 'INFO',
                    f"[{event}] {labels.get(event, event)}; audit={audit_id}"
                    + (f"; request={data['request_id']}" if data.get('request_id') else '')
                    + (f"; câu={data['scene_id']}" if data.get('scene_id') else ''),
                    json.dumps(record, ensure_ascii=False, indent=2))

            def check(cards):
                nonlocal pass1_calls
                checks = []
                for batch in visual_batches(cards):
                    request_id = uuid.uuid4().hex[:12]
                    started_at = time.monotonic()
                    expected_ids = [card['scene_id'] for card in batch]
                    manifest, media_path, response, phase = [], None, None, 'prepare_media'
                    diagnostic('AUDIT_REQUEST_START', {'request_id': request_id,
                        'expected_ids': expected_ids, 'cards': batch})
                    try:
                        media, manifest, media_path = ensure_visual_pack(batch, source_offset, 'Kiem chung cau voi hinh')
                        prompt = (
                            'XEM VIDEO DINH KEM, kiem chung tung cau voi HINH THUC TE va observed_facts. '
                            'Nhan tren hinh la scene_id, visual_id va SHOT nguon. Video la cac shot ghep, '
                            'khong dung thoi gian trong video ghep de suy ra phim nguon. '
                            'Moi menh de ve nhan vat, hanh dong, do vat va thu tu phai khop hinh cua DUNG scene_id. '
                            'Khong chap nhan chi vi observed_facts noi vay neu hinh khong chung minh. '
                            'Khong bat cau ke lai tat ca facts; chi kiem tra nhung y no thuc su ke. '
                            'Loi thoai/ten da xac nhan trong facts co the ho tro ngu canh; khong suy dien noi tam. '
                            'Neu hinh khong ro hoac khong du bang chung: faithful=false va neu cu the. '
                            'Tra JSON {"checks":[{"scene_id":"S0001","faithful":true,"reason":""}]} '
                            'du moi scene_id, khong lap ID.\nMANIFEST: ' + json.dumps(manifest, ensure_ascii=False)
                            + '\nCAC CAU: ' + json.dumps(batch, ensure_ascii=False))
                        with visual_call_lock:
                            pass1_calls += 1
                        phase = 'generate'
                        response = generate_gemini_text_with_recovery(
                            visual_client, selected_model, [media, prompt], writing_config,
                            f'Kiem chung {len(batch)} cau bang video',
                            update_progress=lambda message: update_prog(83, message),
                            stream=True, max_attempts=1, capacity_max_retries=2, retry_timeouts=True,
                            minute_quota_max_retries=0,
                            usage_callback=record_gemini_usage, quota_pacer=gemini_quota_pacer,
                            estimated_input_tokens=max(1000, len(prompt)) + estimate_gemini_video_input_tokens(
                                sum(r['source_end'] - r['source_start'] for r in manifest), 2.0))
                        diagnostic('AUDIT_RESPONSE_RAW', {'request_id': request_id,
                            'expected_ids': expected_ids, 'media_path': media_path, 'manifest': manifest,
                            'elapsed_seconds': round(time.monotonic() - started_at, 3),
                            'raw_response': response})
                        phase = 'parse_json'
                        parsed = parse_audit_response(response)
                        report, _ = inspect_audit_response(parsed, expected_ids)
                        diagnostic('AUDIT_REQUEST_RESULT', dict(report, request_id=request_id))
                        if isinstance(parsed.get('checks'), list):
                            checks.extend(parsed['checks'])
                    except Exception as exc:
                        category = (
                            'AUTH_ERROR' if is_gemini_auth_error(exc) else
                            'DAILY_QUOTA' if is_daily_project_gemini_quota_error(exc) else
                            'TIMEOUT' if is_gemini_transport_timeout(exc) else
                            'CAPACITY_503' if is_gemini_capacity_503_error(exc) else
                            'MINUTE_QUOTA' if is_per_minute_gemini_quota_error(exc) else
                            'JSON_PARSE_ERROR' if phase == 'parse_json' else
                            'MEDIA_PREPARATION_ERROR' if phase == 'prepare_media' else 'REQUEST_ERROR')
                        # Never include the key in diagnostics, even if an SDK
                        # exception happens to echo a request URL.
                        error_text = gemini_error_chain_text(exc)
                        error_trace = traceback.format_exc()
                        if api_key:
                            error_text = error_text.replace(api_key, '[REDACTED_API_KEY]')
                            error_trace = error_trace.replace(api_key, '[REDACTED_API_KEY]')
                        diagnostic('AUDIT_REQUEST_ERROR', {'request_id': request_id,
                            'category': category, 'phase': phase, 'expected_ids': expected_ids,
                            'elapsed_seconds': round(time.monotonic() - started_at, 3),
                            'error_type': type(exc).__name__, 'error': error_text,
                            'traceback': error_trace, 'raw_response': response,
                            'media_path': media_path, 'manifest': manifest,
                            'status': 'pending_not_rejected'})
                        if is_daily_project_gemini_quota_error(exc) or is_gemini_auth_error(exc):
                            raise
                        append_task_log(task_id, 'WARNING',
                            'Kiem chung hinh gian doan; giu trang thai chua xac minh, thu lai co gioi han.', str(exc))
                return {'checks': checks}

            if source_scenes and all(s.get('evidence_sentence_plan_version') == 1 for s in source_scenes):
                from evidence_writer import audit_planned_payload, text_audit_prompt, AUDIT_SCHEMA
                with task_lock:
                    text_cache = copy.deepcopy(TASKS.get(task_id, {}).get('evidence_text_audit') or {})

                def check_text(cards):
                    nonlocal pass1_calls
                    prompt = text_audit_prompt(cards)
                    config = writing_config.model_copy(update={
                        'temperature': 0.1, 'response_json_schema': AUDIT_SCHEMA})
                    pass1_calls += 1
                    response = generate_gemini_text_with_recovery(
                        client, selected_model, [prompt], config, 'Kiem chung loi voi bang chung da luu',
                        update_progress=lambda message: update_prog(progress_base + 8, message),
                        stream=True, max_attempts=1, capacity_max_retries=2,
                        usage_callback=record_gemini_usage, quota_pacer=gemini_quota_pacer,
                        estimated_input_tokens=max(1000, len(prompt)))
                    return parse_audit_response(response)

                def save_text(state):
                    text_cache.clear()
                    text_cache.update(state)
                    update_task_fields(task_id, evidence_text_audit=copy.deepcopy(state))

                def save_video(state):
                    grounded_audit_cache.clear()
                    grounded_audit_cache.update(state)
                    update_task_fields(task_id, grounded_narration_audit=copy.deepcopy(state))

                source_stat = os.stat(video_path)
                checked = audit_planned_payload(payload, check_text, check, text_cache=text_cache,
                    video_cache=grounded_audit_cache, save_text=save_text, save_video=save_video,
                    cache_only=cache_only, source_identity=[os.path.realpath(video_path),
                        source_stat.st_size, source_stat.st_mtime_ns], source_offset=source_offset)
                summary = summarize_audit_scenes(checked)
                append_task_log(task_id, 'INFO',
                    f'Kiem chung bang fact: dat {len(summary["passed_ids"])}/{summary["total"]}; '
                    f'cho {len(summary["pending_ids"])}; can sua {len(summary["rejected_ids"])}. '
                    'Chi mo lai video cho menh de chua ro, khong quet lai toan bai.',
                    json.dumps(summary, ensure_ascii=False))
                if summary['pending_ids']:
                    update_task_fields(task_id, pending_narration_audit_payload=checked)
                    raise ValueError('NARRATION_AUDIT_INCOMPLETE:' + ','.join(summary['pending_ids']))
                return checked

            def remember(state):
                grounded_audit_cache.clear()
                grounded_audit_cache.update(state)
                update_task_fields(task_id, grounded_narration_audit=copy.deepcopy(state))

            source_stat = os.stat(video_path)
            checked = audit_narrations(
                payload, check, checkpoint=grounded_audit_cache, save=remember, visual_evidence=True,
                source_identity=[os.path.realpath(video_path), source_stat.st_size,
                                 source_stat.st_mtime_ns], visual_source_offset=source_offset,
                max_attempts=0 if cache_only else 2, max_workers=2, diagnostic=diagnostic)
            before = {s['scene_id']: s for p in payload.get('parts') or [] for s in p.get('scenes') or []}
            for part in checked.get('parts') or []:
                for scene in part.get('scenes') or []:
                    old = before.get(scene['scene_id'], {})
                    if scene.get('grounding_audit_pending') and (old.get('grounding_audit_passed') or
                            (cache_only and not old.get('grounding_audit_pending'))):
                        diagnostic('AUDIT_STATUS_TRANSITION', {
                            'scene_id': scene['scene_id'], 'before_passed': old.get('grounding_audit_passed'),
                            'before_pending': old.get('grounding_audit_pending'),
                            'before_fallback': bool(old.get('visual_fallback')),
                            'after_status': 'pending', 'reason': 'no_cached_verdict_for_current_signature',
                            'previous_errors': old.get('visual_alignment_errors') or []})
            summary = summarize_audit_scenes(checked)
            append_task_log(task_id, 'WARNING' if summary['pending_ids'] or summary['rejected_ids'] else 'INFO',
                f"[AUDIT_TOTAL] audit={audit_id}; tổng {summary['total']} câu; "
                f"đạt {len(summary['passed_ids'])}; chưa có kết quả {len(summary['pending_ids'])}; "
                f"bị bác bỏ {len(summary['rejected_ids'])}; dùng hình dự phòng {len(summary['fallback_ids'])}.",
                json.dumps(dict(audit_context, **summary), ensure_ascii=False, indent=2))
            pending_ids = [s['scene_id'] for p in checked.get('parts', [])
                           for s in p.get('scenes', []) if s.get('grounding_audit_pending')]
            if pending_ids:
                update_task_fields(task_id, pending_narration_audit_payload=checked)
                append_task_log(task_id, 'WARNING',
                    'Chua du ket qua kiem chung; se cho phep hinh du phong co nhat ky neu het luot sua.',
                    ', '.join(pending_ids))
            return checked

        def use_visual_fallbacks(payload, source_offset=0.0):
            for part in payload.get('parts') or []:
                for scene in part.get('scenes') or []:
                    if scene.get('evidence_sentence_plan_version') and scene.get('visual_alignment_errors'):
                        raise ValueError('EVIDENCE_NARRATION_NOT_VERIFIED:' + str(scene.get('scene_id'))
                            + ':' + ','.join(map(str, scene['visual_alignment_errors'])))
            def record(report):
                from recap_pipeline import _time_seconds, _format_time
                report = copy.deepcopy(report)
                report['start'] = _format_time(_time_seconds(report['start']) + source_offset)
                report['end'] = _format_time(_time_seconds(report['end']) + source_offset)
                report['scope_seconds'] = [v + source_offset for v in report['scope_seconds']]
                report['time_reference'] = 'source_video'
                with task_lock:
                    reports = copy.deepcopy(TASKS.get(task_id, {}).get('visual_fallback_reports') or [])
                if report in reports:
                    return
                reports.append(report)
                update_task_fields(task_id, visual_fallback_reports=reports,
                                   visual_fallback_count=len(reports))
                append_task_log(task_id, 'WARNING',
                    f"HINH GAN DUNG - can xem lai: phan {report.get('part')}, cau {report.get('scene_id')}; "
                    f"shot {', '.join(report['shot_ids'])}; {report['start']}-{report['end']}. "
                    'Chua xac minh khop loi ke; tiep tuc xu ly.',
                    json.dumps(dict(report, diagnostic_category=(
                        'MISSING_VALID_VERDICT' if any(str(r).startswith('narration_audit_pending:')
                            for r in report.get('reasons') or []) else
                        'CONTENT_REJECTED' if any(str(r).startswith('unsupported_narration:')
                            for r in report.get('reasons') or []) else 'VISUAL_LINK_INVALID'),
                        verified=False), ensure_ascii=False))
            return apply_local_visual_fallbacks(payload, log=record)

        append_task_log(task_id, 'INFO',
            'Viet tu ke hoach tung cau, khong gui video cho writer; '
            'kiem chung bang fact truoc, chi xem clip con nghi van. Giu dung tong so cau.')
        update_prog(42, "Dang viet recap tu event map da khoa...")
        for part_index in range(len(part_responses), writing_group_count):
            part_number = part_index + 1
            progress_base = 40 + part_index * 14
            story_context = build_part_story_context(
                story_plan,
                part_number,
                previous_response=(part_responses[-1] if part_responses else None),
            )
            required_events = story_context["required_events"]
            required_event_ids = [event["event_id"] for event in required_events]
            story_beat_by_event_id = {
                str(event.get("event_id", "")).strip().upper():
                str(event.get("story_beat_id", "")).strip().upper()
                for event in required_events
                if str(event.get("event_id", "")).strip()
                and str(event.get("story_beat_id", "")).strip()
            }
            locked_event_by_id = {
                str(event.get("event_id", "")).strip().upper(): event
                for event in required_events
                if str(event.get("event_id", "")).strip()
            }
            required_scene_count = len(required_events)
            required_ending_sec = story_context["required_ending_sec"]
            def write_visual_batches(targeted, requested, base_response, base_requirements):
                import hashlib
                from recap_visual_media import inspect_writing_batch, parse_writing_response
                from recap_pipeline import build_continuous_recap_draft_prompt, parse_continuous_recap_draft
                from narrative_contract import chapter_prompt, validate_chapter, best_chapter_candidate, sentences
                nonlocal pass1_calls
                events = [e for e in required_events if not targeted or e['event_id'] in requested]
                combined = []
                if events and all(e.get('evidence_sentence_plan_version') == 1 for e in events):
                    from evidence_writer import writing_prompt, writing_batches, writing_schema, write_batch, rejected_cached_events, has_direct_dialogue
                    for batch_index, batch in enumerate(writing_batches(events), 1):
                        update_task_fields(task_id, active_writing_context={
                            'part': part_number, 'batch': batch_index,
                            'event_ids': [e['event_id'] for e in batch], 'model': selected_model,
                            'mode': 'planned_text_only', 'video_seconds': 0})
                        context = copy.deepcopy(story_context)
                        context.update(style=style, custom_prompt=custom_prompt)
                        context['previous_part_tail'] = (list(context.get('previous_part_tail') or [])
                            + [{'tts': row['tts']} for row in combined])[-3:]
                        last = required_events.index(batch[-1])
                        context['next_batch_context'] = [{'event': e.get('event')}
                            for e in required_events[last + 1:last + 3]]
                        if targeted:
                            previous = parse_json_object(base_response)
                            scene_ids = {e['scene_id'] for e in batch}
                            context['custom_prompt'] += ('\nSUA RIENG CAU BI BAO SAI, GIU KE HOACH CAU: '
                                + json.dumps({'previous': [s for p in previous.get('parts') or []
                                    for s in p.get('scenes') or [] if s.get('scene_id') in scene_ids],
                                    'issues': [r for name in ('invalid_timestamp_scenes', 'invalid_tts_scenes', 'invalid_character_mentions')
                                        for r in (base_requirements or {}).get(name) or [] if r.get('scene_id') in scene_ids]},
                                    ensure_ascii=False))
                        prompt = writing_prompt(batch, context)
                        key = hashlib.sha256(json.dumps(['evidence_paragraph_writer_v2', prompt,
                            selected_model, thinking_effort], ensure_ascii=False).encode()).hexdigest()
                        with task_lock:
                            saved = copy.deepcopy(TASKS.get(task_id, {}).get('evidence_writing_batches') or {})
                            candidates = copy.deepcopy(TASKS.get(task_id, {}).get('evidence_writing_candidates') or {})
                            text_verdicts = copy.deepcopy(TASKS.get(task_id, {}).get('evidence_text_audit') or {})
                        rows = saved.get(key)
                        if targeted and isinstance(rows, list):
                            rejected_ids = rejected_cached_events(rows, text_verdicts)
                            if rejected_ids:
                                candidates[key] = [{'scenes': [r for r in draft.get('scenes') or []
                                    if isinstance(r, dict) and r.get('event_id') not in rejected_ids]}
                                    for draft in candidates.get(key) or [] if isinstance(draft, dict)]
                                rows = None
                        if rows is not None:
                            from source_grounded_writer import is_source_grounded
                            valid_cache = (isinstance(rows, list)
                                and len(rows) == len(batch)
                                and all(isinstance(r, dict) and is_source_grounded(r)
                                        and not has_direct_dialogue(r.get('tts')) for r in rows)
                                and [r.get('event_id') for r in rows] == [e['event_id'] for e in batch])
                            if not valid_cache:
                                rows = None
                                append_task_log(task_id, 'WARNING',
                                    f'Khoi phuc nhom {batch_index}: cache loi/hinh khong con hop le; dung ban cau goc da luu.')
                        if rows is None:
                            def generate_evidence_batch(request_events, request_prompt):
                                nonlocal pass1_calls
                                from structured_generation import generate_with_schema_recovery
                                config = writing_config.model_copy(update={
                                    'temperature': 0.2, 'response_schema': None,
                                    'response_json_schema': writing_schema(request_events)})
                                pass1_calls += 1
                                def generate_config(request_config):
                                    return generate_gemini_text_with_recovery(
                                        client, selected_model, [request_prompt], request_config,
                                        f'Viet recap toan phim, nhom {batch_index}',
                                        update_progress=lambda message: update_prog(progress_base + 7, message),
                                        stream=True, max_attempts=1, capacity_max_retries=2, retry_timeouts=True,
                                        resume_503_forever=False, usage_callback=record_gemini_usage,
                                        quota_pacer=gemini_quota_pacer,
                                        estimated_input_tokens=max(1000, len(request_prompt)))
                                return generate_with_schema_recovery(generate_config, config,
                                    report=lambda message: append_task_log(task_id, 'WARNING', message))

                            def save_evidence_candidates(drafts):
                                candidates[key] = copy.deepcopy(drafts)
                                update_task_fields(task_id, evidence_writing_candidates=candidates)

                            rows = write_batch(batch, context, generate_evidence_batch,
                                candidates=candidates.get(key), save=save_evidence_candidates,
                                characters=story_plan.get('characters') or [])
                            saved[key] = rows
                            update_task_fields(task_id, evidence_writing_batches=saved)
                        combined.extend(copy.deepcopy(rows))
                    return json.dumps({'scenes': combined}, ensure_ascii=False)
                batches = list(visual_batches(events) if targeted else evidence_writing_batches(events))
                for batch_index, batch in enumerate(batches, 1):
                    batch_ids = [e['event_id'] for e in batch]
                    update_task_fields(task_id, active_writing_context={
                        'part': part_number, 'batch': batch_index, 'event_ids': batch_ids, 'model': selected_model})
                    batch_locked = {e['event_id']: e for e in batch}
                    if targeted:
                        scene_ids = {str(e.get('scene_id', '')) for e in batch}
                        requirements = copy.deepcopy(base_requirements)
                        for key in ('invalid_timestamp_scenes', 'invalid_tts_scenes', 'invalid_character_mentions'):
                            requirements[key] = [r for r in requirements.get(key) or []
                                                 if str(r.get('scene_id', '')) in scene_ids]
                        requirements['missing_event_ids'] = [eid for eid in requirements.get('missing_event_ids') or []
                                                             if eid in batch_ids]
                        batch_prompt = build_targeted_narration_repair_prompt(
                            parse_json_object(base_response), requirements, batch_locked)
                        batch_prompt += ('\nCAU CHI LOI LIEN KET HINH (khong co unsupported_narration, '
                            'loi TTS hay ten nhan vat): GIU NGUYEN tts, chi sua visual_links. '
                            'Trigger giua cau phai la cum tu co that trong tts goc; '
                            'trigger dau tien la chuoi rong.')
                    else:
                        context = copy.deepcopy(story_context)
                        context['required_events'] = batch
                        if combined:
                            context['previous_part_tail'] = (
                                list(story_context.get('previous_part_tail') or [])
                                + [{'tts': r.get('tts', '')} for r in combined])[-8:]
                        last_index = required_events.index(batch[-1])
                        context['next_batch_context'] = [
                            {'event': e.get('narration_focus') or e.get('event')}
                            for e in required_events[last_index + 1:last_index + 3]]
                        context.update(style=style, custom_prompt=custom_prompt)
                        batch_prompt = build_single_part_video_prompt(
                            part_number, boundaries[part_number] - boundaries[part_index],
                            boundaries[part_index], boundaries[part_number],
                            sum(int(e.get('tts_target_words', 20)) for e in batch),
                            style=style, custom_prompt=custom_prompt, enable_multi_voice=enable_multi_voice,
                            story_context=context)
                    # Evidence extraction already viewed the source. Ordinary
                    # writing/length repairs need text, not another video pass.
                    batch_prompt += (
                        '\nCHI DIEN DAT BANG CHUNG DA KHOA: observed_facts la du kien, '
                        'ngu canh chi de noi mach, khong chung minh hanh dong hay noi tam. '
                        'Khong doi ID, timestamp hoac them du kien de du so tu.')
                    contents = [batch_prompt]
                    visual_repair = targeted and any(
                        any(str(reason).startswith('unsupported_narration:')
                            for reason in row.get('reasons') or [])
                        for row in requirements.get('invalid_timestamp_scenes') or []
                        if str(row.get('scene_id', '')).upper() in
                        {str(e.get('scene_id', '')).upper() for e in batch})
                    if visual_repair:
                        media, manifest, _ = ensure_visual_pack(
                            batch, boundaries[part_index], f'Sua cau sai hinh phan {part_number}, nhom {batch_index}')
                        batch_prompt += ('\nXem clip de bo chi tiet bi bac bo; chi ke dieu ca hinh va '
                            'observed_facts cung ho tro. MANIFEST: ' + json.dumps(manifest, ensure_ascii=False))
                        contents = [media, batch_prompt]
                    situation_batch = all(e.get('situation_narration') for e in batch)
                    draft_prompt = (build_continuous_recap_draft_prompt(batch, context)
                                    if not targeted and not situation_batch else '')
                    key = hashlib.sha256(json.dumps(['continuous_evidence_writer_v3', batch, selected_model, thinking_effort, batch_prompt, draft_prompt],
                                                    ensure_ascii=False).encode()).hexdigest()
                    with task_lock:
                        saved_batches = copy.deepcopy(TASKS.get(task_id, {}).get('visual_writing_batches') or {})
                    rows = saved_batches.get(key)
                    if not rows:
                        if not targeted and all(e.get('sentence_quota') for e in batch):
                            prose_prompt = chapter_prompt(batch, context)
                            prose_key = hashlib.sha256((key + prose_prompt).encode()).hexdigest()
                            with task_lock:
                                chapter_cache = copy.deepcopy(TASKS.get(task_id, {}).get('chapter_prose_batches') or {})
                            prose = chapter_cache.get(prose_key)
                            if prose is None:
                                pass1_calls += 1
                                prose_raw = generate_gemini_text_with_recovery(
                                    visual_client, selected_model, [prose_prompt], writing_config,
                                    f'Viet chuong dung so cau phan {part_number}, nhom {batch_index}',
                                    update_progress=lambda message, p=progress_base + 7: update_prog(p, message),
                                    stream=True, max_attempts=1, capacity_max_retries=0, retry_timeouts=False,
                                    resume_503_forever=False, usage_callback=record_gemini_usage,
                                    quota_pacer=gemini_quota_pacer, estimated_input_tokens=max(1000, len(prose_prompt)))
                                candidate = parse_writing_response(prose_raw)
                                with task_lock:
                                    candidate_cache = copy.deepcopy(TASKS.get(task_id, {}).get('chapter_sentence_candidates') or {})
                                candidates = candidate_cache.setdefault(prose_key, [])
                                candidates.append(candidate)
                                candidate_cache[prose_key] = candidates[-2:]
                                update_task_fields(task_id, chapter_sentence_candidates=candidate_cache)
                                try:
                                    prose = validate_chapter(candidate, batch)
                                except ValueError as chapter_error:
                                    if not str(chapter_error).startswith('CHAPTER_SENTENCE_QUOTA_MISMATCH:') or len(candidates) < 2:
                                        raise
                                    prose = best_chapter_candidate(candidates, batch)
                                    missing = [e for e, row in zip(batch, prose)
                                               if len(sentences(row['tts'])) != e['sentence_quota']]
                                    if missing:
                                        repair_prompt = chapter_prompt(missing, context) + '\nSUA BAN TOT NHAT BEN DUOI: giu dien bien va thu tu, tach cau ghep thanh cau hoan chinh de du quota; khong them tinh tiet hay cau dem.\n' + json.dumps([r for r in prose if r['event_id'] in {e['event_id'] for e in missing}], ensure_ascii=False)
                                        pass1_calls += 1
                                        repair_raw = generate_gemini_text_with_recovery(
                                            visual_client, selected_model, [repair_prompt], writing_config,
                                            f'Sua so cau ban tot nhat phan {part_number}, nhom {batch_index}',
                                            stream=True, max_attempts=1, capacity_max_retries=0, retry_timeouts=False,
                                            resume_503_forever=False, usage_callback=record_gemini_usage,
                                            quota_pacer=gemini_quota_pacer, estimated_input_tokens=max(1000, len(repair_prompt)))
                                        repaired = validate_chapter(parse_writing_response(repair_raw), missing)
                                        replacements = {r['event_id']: r for r in repaired}
                                        prose = [replacements.get(r['event_id'], r) for r in prose]
                                    append_task_log(task_id, 'WARNING', f'Phan {part_number}: da chon ban gan so cau nhat va can lai dung quota; giu cac tinh huong da dat.')
                                chapter_cache[prose_key] = prose
                                update_task_fields(task_id, chapter_prose_batches=chapter_cache)
                            contents = [batch_prompt + '\nBUOC GHEP HINH: loi ke da hoan tat ben duoi. '
                                'GIU NGUYEN TUYET DOI tts, khong viet lai, khong them/bot cau. '
                                'Chi chon visual_links tu kho hinh cua tinh huong cho loi ke nay.\n'
                                + json.dumps(prose, ensure_ascii=False)]
                        if not targeted and not situation_batch:
                            with task_lock:
                                drafts = copy.deepcopy(TASKS.get(task_id, {}).get('continuous_recap_drafts') or {})
                            draft = drafts.get(key)
                            if not draft:
                                pass1_calls += 1
                                draft_raw = generate_gemini_text_with_recovery(
                                    visual_client, selected_model, [draft_prompt], writing_config,
                                    f'Viet mach truyen phan {part_number}, nhom {batch_index}/{len(batches)}',
                                    update_progress=lambda message, p=progress_base + 7: update_prog(p, message),
                                    stream=True, max_attempts=1, capacity_max_retries=0, retry_timeouts=False,
                                    resume_503_forever=False, usage_callback=record_gemini_usage,
                                    quota_pacer=gemini_quota_pacer,
                                    estimated_input_tokens=max(1000, len(draft_prompt)))
                                draft = parse_continuous_recap_draft(draft_raw)
                                drafts[key] = draft
                                update_task_fields(task_id, continuous_recap_drafts=drafts)
                            contents = [batch_prompt + '\nBUOC 2: PHAN DOAN KE VE CAC O HINH DA KHOA. '
                                'Dung doan duoi lam mach ke, khong viet lai thanh danh sach mo ta shot. '
                                'Chi chinh tu ngu can thiet de khop fact cua tung o, giu thu tu, visual_links '
                                'va gioi han tu. Doc lai cac tts noi lien de sua lap y va cho dut mach. '
                                'Khong dua fact cua o sau ve o truoc.\nDOAN KE LIEN MACH:\n' + draft]
                        pass1_calls += 1
                        raw = generate_gemini_text_with_recovery(
                            visual_client, selected_model, contents, writing_config,
                            f'Gemini viet tu bang chung phan {part_number}, nhom {batch_index}/{len(batches)}',
                            update_progress=lambda message, p=progress_base + 7: update_prog(p, message),
                            stream=True, max_attempts=1, capacity_max_retries=0, retry_timeouts=False,
                            resume_503_forever=False, usage_callback=record_gemini_usage,
                            quota_pacer=gemini_quota_pacer,
                            estimated_input_tokens=max(1000, len(batch_prompt)) + (estimate_gemini_video_input_tokens(
                                sum(r['source_end'] - r['source_start'] for r in manifest), 2.0) if visual_repair else 0))
                        diagnostic = {'expected_ids': batch_ids}
                        try:
                            parsed = parse_writing_response(raw)
                            rows, diagnostic = inspect_writing_batch(parsed, batch_ids)
                            if not diagnostic['valid']:
                                raise ValueError('VISUAL_WRITING_BATCH_IDS_INVALID: '
                                                 + json.dumps(diagnostic, ensure_ascii=False))
                            if not targeted and all(e.get('sentence_quota') for e in batch):
                                if [r.get('tts') for r in rows] != [r['tts'] for r in prose]:
                                    raise ValueError('VISUAL_MAPPING_CHANGED_LOCKED_PROSE')
                        except Exception as exc:
                            failure = {'part': part_number, 'batch': batch_index,
                                       'model': selected_model, 'error_type': type(exc).__name__,
                                       'error': str(exc), 'diagnostic': diagnostic, 'raw_response': raw}
                            with task_lock:
                                failures = copy.deepcopy(TASKS.get(task_id, {}).get('visual_writing_failures') or [])
                            failures.append(failure)
                            update_task_fields(task_id, visual_writing_failures=failures)
                            append_task_log(task_id, 'ERROR',
                                f'CHI TIET LOI NHOM CAU: phan {part_number}, nhom {batch_index}, model {selected_model}.',
                                json.dumps(failure, ensure_ascii=False, indent=2))
                            raise
                        saved_batches[key] = rows
                        update_task_fields(task_id, visual_writing_batches=saved_batches)
                    else:
                        append_task_log(task_id, 'INFO',
                            f'Dung lai nhom cau da viet tu bang chung: phan {part_number}, nhom {batch_index}.')
                    combined.extend(copy.deepcopy(rows))
                return json.dumps({'scenes': combined}, ensure_ascii=False)

            part_error = None
            accepted_response = None
            accepted_repeat_issues = []
            part_candidates = []
            repair_base_response = None
            repair_base_requirements = None

            def assess_candidate_repetition(candidate_text):
                candidate_payload = parse_json_object(candidate_text)
                current_parts = [
                    copy.deepcopy(part)
                    for part in (candidate_payload.get("parts") or [])
                    if isinstance(part, dict)
                ]
                current_scene_ids = {
                    str(scene.get("scene_id", "")).strip().upper()
                    for part in current_parts
                    for scene in (part.get("scenes") or [])
                    if isinstance(scene, dict)
                }
                combined_parts = []
                combined_characters = copy.deepcopy(
                    story_plan.get("characters") or []
                )
                for previous_text in part_responses:
                    previous_payload = parse_json_object(previous_text)
                    combined_parts.extend(
                        copy.deepcopy(part)
                        for part in (previous_payload.get("parts") or [])
                        if isinstance(part, dict)
                    )
                combined_parts.extend(current_parts)
                combined_payload = {
                    "characters": combined_characters,
                    "parts": combined_parts,
                }
                quality_payload = clamp_payload_narrations(combined_payload)
                repeated = [
                    issue for issue in inspect_narration_quality(
                        quality_payload, maximum=200
                    )
                    if issue.get("code") == "repeated_narration_idea"
                    and str(issue.get("scene_id", "")).strip().upper()
                    in current_scene_ids
                ]
                return quality_payload, repeated

            def repair_event_ids(candidate_text, requirements):
                payload = parse_json_object(candidate_text)
                scene_to_event = {}
                for part in payload.get("parts") or []:
                    for scene in part.get("scenes") or []:
                        if not isinstance(scene, dict):
                            continue
                        scene_id = str(
                            scene.get("scene_id", "")
                        ).strip().upper()
                        mapped = scene.get("covered_event_ids") or []
                        if not isinstance(mapped, list):
                            mapped = [mapped]
                        event_id = next((
                            str(value).strip().upper() for value in mapped
                            if str(value).strip().upper() in locked_event_by_id
                        ), "")
                        if scene_id and event_id:
                            scene_to_event[scene_id] = event_id
                result = []
                defect_rows = (
                    list(requirements.get("invalid_tts_scenes") or [])
                    + list(requirements.get("invalid_character_mentions") or [])
                    + list(requirements.get("invalid_timestamp_scenes") or [])
                )
                for item in defect_rows:
                    reasons = item.get('reasons') or []
                    if reasons and all(str(reason).startswith('narration_audit_pending:') for reason in reasons):
                        continue  # Missing verifier replies must never rewrite valid prose.
                    event_id = scene_to_event.get(
                        str(item.get("scene_id", "")).strip().upper(), ""
                    )
                    if event_id and event_id not in result:
                        result.append(event_id)
                for value in requirements.get("missing_event_ids") or []:
                    event_id = str(value).strip().upper()
                    if event_id in locked_event_by_id and event_id not in result:
                        result.append(event_id)
                return result

            # One initial draft and one local repair. Missing verification is
            # handled by the verifier's bounded retry, never by another draft.
            for part_attempt in range(2):
                requested_repairs = (
                    repair_event_ids(
                        repair_base_response, repair_base_requirements
                    )
                    if repair_base_response is not None
                    and repair_base_requirements is not None else []
                )
                targeted_repair = bool(part_attempt and requested_repairs)
                if part_attempt and repair_base_response is not None and not requested_repairs:
                    break
                update_prog(
                    progress_base + 7,
                    (
                        f"Gemini dang sua rieng {len(requested_repairs)} cau "
                        f"cua recap toan phim "
                        f"(luot {part_attempt}/1)..."
                        if targeted_repair else
                        f"Gemini dang viet recap toan phim"
                        + (f" (thu lai {part_attempt}/1)..." if part_attempt else "...")
                    ),
                )
                response_text = ""
                try:
                    response_text = write_visual_batches(
                        targeted_repair, requested_repairs, repair_base_response, repair_base_requirements)
                    if targeted_repair:
                        repaired_payload = apply_targeted_narration_repairs(
                            parse_json_object(repair_base_response),
                            response_text,
                            requested_repairs,
                            locked_event_by_id,
                        )
                        from grounded_pipeline import preserve_structure_only_narrations
                        repaired_payload = preserve_structure_only_narrations(
                            parse_json_object(repair_base_response), repaired_payload,
                            repair_base_requirements)
                        response_text = json.dumps(
                            repaired_payload, ensure_ascii=False
                        )
                    response_text = coerce_single_part_response(
                        response_text,
                        part_number,
                        boundaries[part_number] - boundaries[part_index],
                        part_word_targets[part_index],
                        required_event_ids=required_event_ids,
                        story_beat_by_event_id=story_beat_by_event_id,
                        locked_event_by_id=locked_event_by_id,
                        locked_characters=story_plan.get("characters") or [],
                    )
                    part_requirements = single_part_missing_requirements(
                        response_text,
                        boundaries[part_number] - boundaries[part_index],
                        part_word_targets[part_index],
                        required_event_ids=required_event_ids,
                        require_word_floor=False,
                        required_ending_sec=required_ending_sec,
                        require_tts_word_range=True,
                        required_scene_count=required_scene_count,
                    )
                    if targeted_repair:
                        old_payload = parse_json_object(repair_base_response)
                        new_payload = parse_json_object(response_text)
                        old_scene_by_id = {
                            str(scene.get("scene_id", "")).strip().upper(): scene
                            for part in old_payload.get("parts") or []
                            for scene in (part.get("scenes") or [])
                            if isinstance(scene, dict)
                        }
                        new_scene_by_id = {
                            str(scene.get("scene_id", "")).strip().upper(): scene
                            for part in new_payload.get("parts") or []
                            for scene in (part.get("scenes") or [])
                            if isinstance(scene, dict)
                        }
                        for event_id in requested_repairs:
                            scene_id = str(
                                locked_event_by_id[event_id].get("scene_id", "")
                            ).strip().upper()
                            old_scene = old_scene_by_id.get(scene_id)
                            new_scene = new_scene_by_id.get(scene_id)
                            if old_scene is None or new_scene is None:
                                continue
                            if narration_scene_issue_rank(
                                part_requirements, scene_id
                            ) > narration_scene_issue_rank(
                                repair_base_requirements, scene_id
                            ):
                                new_scene.clear()
                                new_scene.update(copy.deepcopy(old_scene))
                        response_text = json.dumps(
                            new_payload, ensure_ascii=False
                        )
                        part_requirements = single_part_missing_requirements(
                            response_text,
                            boundaries[part_number] - boundaries[part_index],
                            part_word_targets[part_index],
                            required_event_ids=required_event_ids,
                            require_word_floor=False,
                            required_ending_sec=required_ending_sec,
                            require_tts_word_range=True,
                            required_scene_count=required_scene_count,
                        )
                    # Normalize BEFORE verification so the final join does not
                    # silently edit a sentence that was already checked.
                    response_text = json.dumps(audit_grounded_payload(
                        clamp_payload_narrations(parse_json_object(response_text)),
                        source_offset=boundaries[part_index],
                        stage='part_repair' if targeted_repair else 'part_draft'), ensure_ascii=False)
                    if targeted_repair:
                        from grounded_pipeline import preserve_verified_scenes
                        response_text = json.dumps(preserve_verified_scenes(
                            parse_json_object(repair_base_response),
                            parse_json_object(response_text)), ensure_ascii=False)
                    part_requirements = single_part_missing_requirements(
                        response_text, boundaries[part_number] - boundaries[part_index],
                        part_word_targets[part_index], required_event_ids=required_event_ids,
                        require_word_floor=False, required_ending_sec=required_ending_sec,
                        require_tts_word_range=True, required_scene_count=required_scene_count)
                    quality_payload, repeat_issues = assess_candidate_repetition(
                        response_text
                    )
                    candidate_rank = narration_candidate_rank(
                        part_requirements, len(repeat_issues)
                    )
                    part_candidates.append({
                        "rank": candidate_rank,
                        "response": response_text,
                        "repeat_issues": repeat_issues,
                        "requirements": part_requirements,
                    })
                    hard_retry_needed = (
                        single_part_has_hard_failures(part_requirements)
                        or bool(part_requirements.get("invalid_character_mentions"))
                        or bool(part_requirements.get("invalid_tts_scenes"))
                    )
                    if not hard_retry_needed:
                        accepted_response = response_text
                        accepted_repeat_issues = repeat_issues
                        break
                    selected_so_far = min(
                        part_candidates, key=lambda row: row["rank"]
                    )
                    repair_base_response = selected_so_far["response"]
                    repair_base_requirements = selected_so_far["requirements"]
                    repetition_feedback = build_narration_repetition_feedback(
                        quality_payload, repeat_issues
                    )
                    part_error = ValueError(
                        f"{part_requirements['action']}: con "
                        f"{len(part_requirements['invalid_tts_scenes'])} TTS sai, "
                        f"lech {part_requirements['scene_count_gap']} scene, "
                        f"thieu {len(part_requirements['missing_event_ids'])} ID, "
                        f"nham {len(part_requirements.get('invalid_character_mentions') or [])} nhan vat, "
                        f"lap {len(repeat_issues)} y."
                        + " Loi kiem chung/hinh: " + json.dumps(
                            part_requirements.get('invalid_timestamp_scenes') or [], ensure_ascii=False)
                        + " ID can sua: " + ", ".join(
                            str(item.get("scene_id", ""))
                            for item in (
                                list(part_requirements.get("invalid_tts_scenes") or [])
                                + list(part_requirements.get("invalid_character_mentions") or [])
                            )[:12]
                        )
                        + repetition_feedback
                    )
                    append_task_log(
                        task_id,
                        "WARNING",
                        f"Recap toan phim con loi cung "
                        f"(lan {part_attempt + 1}/2); chi sua ID co loi, khong viet lai vi thieu ket qua kiem chung.",
                        str(part_error),
                    )
                    continue
                except Exception as exc:
                    part_error = exc
                    if str(exc).startswith('NARRATION_AUDIT_INCOMPLETE:'):
                        raise  # Rewriting valid text cannot complete a missing verifier reply.
                    if (
                        is_daily_project_gemini_quota_error(exc)
                        or is_gemini_auth_error(exc)
                    ):
                        raise
                    if is_transient_gemini_error(exc) or is_per_minute_gemini_quota_error(exc):
                        append_task_log(task_id, 'WARNING',
                            f'Recap toan phim tam gian doan; giu nhom da viet, khong tieu hao luot sua noi dung.',
                            str(exc))
                        raise
                    append_task_log(
                        task_id,
                        "WARNING",
                        f"Loi xu ly phan {part_number}, "
                            f"lan {part_attempt + 1}/2: {type(exc).__name__}: {exc}",
                        traceback.format_exc() + '\n' +
                        (gemini_response_diagnostic(response_text) if response_text else
                         'Chua nhan duoc ket qua tong hop cua phan; neu loi nhom cau, xem ban ghi CHI TIET LOI NHOM CAU phia tren.'),
                    )

            clip_duration = boundaries[part_number] - boundaries[part_index]
            if accepted_response is None and part_candidates:
                selected_candidate = min(
                    part_candidates, key=lambda row: row["rank"]
                )
                accepted_response = selected_candidate["response"]
                accepted_repeat_issues = selected_candidate["repeat_issues"]
                append_task_log(
                    task_id,
                    "WARNING",
                    f"Recap toan phim het retry; backend giu ban tot nhat "
                    f"voi {len(accepted_repeat_issues)} y con trung.",
                )
            if accepted_response is None:
                raise RuntimeError(
                    f"Recap toan phim chua co loi recap hop le sau mot luot sua rieng. "
                    "Ke hoach da luu, co the thu lai ma khong xem phim tu dau. "
                    f"Loi cuoi: {part_error}"
                )

            best_response = json.dumps(use_visual_fallbacks(
                parse_json_object(accepted_response), source_offset=boundaries[part_index]), ensure_ascii=False)
            best_requirements = single_part_missing_requirements(
                best_response,
                clip_duration,
                part_word_targets[part_index],
                required_event_ids=required_event_ids,
                require_word_floor=False,
                required_ending_sec=required_ending_sec,
                require_tts_word_range=True,
                required_scene_count=required_scene_count,
            )
            accepted_response = best_response
            if single_part_has_hard_failures(best_requirements):
                append_task_log(
                    task_id, "ERROR",
                    f"Recap toan phim con lien ket hinh/ID chua hop le sau sua rieng.",
                    json.dumps(best_requirements.get("invalid_timestamp_scenes") or [], ensure_ascii=False),
                )
                raise RuntimeError(
                    f"Recap toan phim chua dat kiem tra loi ke/hinh. "
                    + '; '.join(
                        str(row.get('scene_id', '')) + ': ' + ', '.join(map(str, row.get('reasons') or []))
                        for row in (best_requirements.get('invalid_timestamp_scenes') or [])[:5])
                    + (f"; thieu ID: {', '.join(map(str, best_requirements.get('missing_event_ids') or []))}"
                       if best_requirements.get('missing_event_ids') else '')
                )
            if any(
                item.get("language_issues")
                for item in best_requirements.get("invalid_tts_scenes") or []
            ):
                raise RuntimeError(
                    f"Recap toan phim con loi recap khong phai tieng Viet "
                    "sau mot luot sua rieng."
                )
            remaining_character_mentions = list(
                best_requirements.get("invalid_character_mentions") or []
            )
            if remaining_character_mentions:
                append_task_log(
                    task_id,
                    "WARNING",
                    f"Recap toan phim con "
                    f"{len(remaining_character_mentions)} cau chua chac ten nhan vat "
                    "sau mot luot sua rieng; giu ung vien tot nhat va tiep tuc.",
                    json.dumps(
                        remaining_character_mentions[:20], ensure_ascii=False
                    ),
                )
            if best_requirements["needs_revision"]:
                append_task_log(
                    task_id,
                    "WARNING",
                    f"Recap toan phim het retry; giu ung vien tot nhat "
                    f"voi {len(best_requirements['invalid_tts_scenes'])} "
                    "canh con loi TTS mem.",
                )
            elif accepted_repeat_issues:
                append_task_log(
                    task_id,
                    "WARNING",
                    f"Recap toan phim giu ung vien it lap nhat sau retry; "
                    f"con {len(accepted_repeat_issues)} cap scene canh bao.",
                    json.dumps(accepted_repeat_issues[:20], ensure_ascii=False),
                )

            # Once the retry budget is exhausted, only structural/evidence
            # violations may fail the part. Soft wording defects are preserved
            # in the best candidate instead of replacing the whole script.
            validate_single_part_response(
                accepted_response,
                part_number,
                part_word_targets[part_index],
                require_coverage=True,
                required_event_ids=required_event_ids,
                require_word_floor=False,
                required_ending_sec=required_ending_sec,
                require_tts_word_range=False,
                required_scene_count=required_scene_count,
            )
            part_responses.append(accepted_response)
            written_rows = parse_json_object(accepted_response)["parts"][0].get("scenes") or []
            append_task_log(
                task_id, "INFO",
                f"Recap toan phim: da khoa lien ket y ke-hinh cho {len(written_rows)} cau; "
                f"{sum(len(row.get('visual_beats') or []) for row in written_rows)} diem doi hanh dong. "
                "Renderer se dat diem cat theo word timing TTS, khong ep shot 2-3 giay.",
            )
            update_task_fields(
                task_id,
                part_writing_checkpoint={
                    "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                    "visual_alignment_version": 6,
                    "completed": part_number,
                    "total": 3,
                    "responses": copy.deepcopy(part_responses),
                    "boundaries": list(boundaries),
                    "target_words": int(target_recap_words),
                    "video_name": video_filename,
                    "model": selected_model,
                    "thinking_effort": thinking_effort,
                    "style": style,
                    "event_ids": plan_event_ids,
                },
            )

        pass1_payload = merge_single_part_responses(
            part_responses, boundaries, target_recap_words, single_timeline=True
        )
        # Character identity is owned by the video-analysis pass. The writer
        # may use names in prose but cannot rename or add registry entries.
        pass1_payload["characters"] = copy.deepcopy(
            story_plan.get("characters") or []
        )

        issues_before = inspect_three_part_payload(pass1_payload)
        final_payload = pass1_payload
        repair_calls = 0
        repair_fallback_used = False
        # No extra whole-script smoothing request. It often returned another
        # large invalid schema while consuming a full text generation call.
        append_task_log(
            task_id,
            "INFO",
            "Da ghep cac nhom thanh mot recap toan phim; khong viet lai theo ba phan.",
        )

        # No final network sweep. Exact evidence signatures authorize reuse;
        # anything changed during merging stays explicitly unverified and goes
        # through the existing logged fallback instead of being marked passed.
        final_payload = use_visual_fallbacks(audit_grounded_payload(
            clamp_payload_narrations(final_payload), cache_only=True, stage='final_cache_check'))
        from visual_audit_diagnostics import summarize_audit_scenes
        export_summary = summarize_audit_scenes(final_payload)
        append_task_log(task_id,
            'WARNING' if export_summary['fallback_ids'] or export_summary['pending_ids']
                         or export_summary['rejected_ids'] else 'INFO',
            f"[AUDIT_EXPORT_SUMMARY] Kết quả cuối: {export_summary['total']} câu; "
            f"đã xác minh {len(export_summary['passed_ids'])}; "
            f"chưa có kết quả hợp lệ {len(export_summary['pending_ids'])}; "
            f"bị bác bỏ {len(export_summary['rejected_ids'])}; "
            f"hình dự phòng chưa xác minh {len(export_summary['fallback_ids'])}.",
            json.dumps(export_summary, ensure_ascii=False, indent=2))
        update_task_fields(task_id, narration_draft_payload=copy.deepcopy(final_payload))
        for part in final_payload['parts']:
            for scene in part.get('scenes') or []:
                if (not scene.get('grounded_plan_locked')
                        or not (scene.get('grounding_audit_passed') or scene.get('visual_fallback'))
                        or scene.get('visual_selection_pending') or scene.get('visual_alignment_errors')
                        or not scene.get('locked_visual_shots')):
                    raise ValueError('GROUNDED_EXPORT_BLOCKED:' + str(scene.get('scene_id')))
        update_task_fields(task_id, matched_narration_payload=copy.deepcopy(final_payload))
        final_quality_issues = inspect_narration_quality(
            final_payload, maximum=200
        )
        repeated_quality_issues = [
            issue for issue in final_quality_issues
            if issue.get("code") == "repeated_narration_idea"
        ]
        final_scene_count = sum(
            len(part.get("scenes") or [])
            for part in final_payload.get("parts") or []
        )
        if final_scene_count != total_scene_count:
            raise ValueError(f'GROUNDED_SITUATION_COUNT_MISMATCH:{final_scene_count}/{total_scene_count}')
        allowed_repeat_issues = max(3, math.floor(final_scene_count * 0.05))
        if len(repeated_quality_issues) > allowed_repeat_issues:
            append_task_log(
                task_id,
                "WARNING",
                "Ban tot nhat sau mot vong sua van con cau lap y; van tra ket qua.",
                json.dumps(repeated_quality_issues[:20], ensure_ascii=False),
            )
        script_text = payload_to_timeline(final_payload)
        script_text = normalize_simple_gemini_timeline(
            script_text, video_duration_sec
        )
        word_count, event_count = measure_script_narration(script_text)
        from narrative_contract import verify_script, sentences as spoken_sentences
        actual_sentence_count, word_count = verify_script(
            script_text, budget['target_sentence_count'], target_recap_words)
        for part in final_payload.get('parts') or []:
            for scene in part.get('scenes') or []:
                if len(spoken_sentences(scene['tts'])) != scene.get('sentence_quota'):
                    raise ValueError('FINAL_PARAGRAPH_QUOTA_MISMATCH:' + scene['scene_id'])
        minimum_accepted_scenes = total_scene_count
        duration_contract_met = event_count == minimum_accepted_scenes

        flat_final_scenes = [
            scene
            for part in final_payload.get("parts") or []
            for scene in part.get("scenes") or []
            if isinstance(scene, dict)
        ]
        final_ids = [
            str(scene.get("scene_id", "")).strip().upper()
            for scene in flat_final_scenes
        ]
        expected_ids = [f"S{index:04d}" for index in range(1, total_scene_count + 1)]
        if final_ids != expected_ids:
            raise RuntimeError(
                "FINAL_SCENE_CONTRACT_FAILED: ID hoac so scene da thay doi sau khi backend khoa."
            )
        soft_length_warnings = [
            scene_id
            for scene_id, scene in zip(final_ids, flat_final_scenes)
            if not 15 <= len(str(scene.get("tts", "")).split()) <= int(
                scene.get("tts_max_words", 25)
            )
        ]
        if soft_length_warnings:
            append_task_log(
                task_id,
                "WARNING",
                "Mot so cau TTS dai/ngan khac thuong; chi canh bao, khong retry.",
                ", ".join(soft_length_warnings[:50]),
            )
        invalid_final_language = [
            str(scene.get("scene_id", ""))
            for scene in flat_final_scenes
            if vietnamese_narration_issues(scene.get("tts", ""))
        ]
        if invalid_final_language:
            raise RuntimeError(
                "LOI_TTS_NGON_NGU: Con "
                f"{len(invalid_final_language)} cau khong dat tieng Viet: "
                + ", ".join(invalid_final_language[:12])
            )

        length_within_range = (
            min_recap_words is None or word_count >= min_recap_words
        ) and (
            max_recap_words is None or word_count <= max_recap_words
        )
        if not length_within_range:
            append_task_log(
                task_id, "WARNING",
                f"Recap co {word_count} tu; muc tieu la "
                f"{min_recap_words}-{max_recap_words} tu. Sau mot vong sua van tra "
                "ban tot nhat, ke ca khi mot so TTS nam ngoai khoang.",
            )
        if not duration_contract_met:
            append_task_log(
                task_id, "WARNING",
                f"Ban tot nhat dat {event_count} scene; backend da khoa "
                f"{minimum_accepted_scenes} scene. So tu {word_count} chi de thong ke.",
            )
        update_prog(95, "Dang don dep...")
        cleanup_analysis_artifacts()
        with task_lock:
            TASKS[task_id]["status"] = "success"
            TASKS[task_id]["progress"] = 100
            TASKS[task_id]["message"] = (
                "Tao kich ban thanh cong!"
                if scene_contract.get("sentence_tolerance_met", True)
                else f"Tao {event_count} cau tu su kien that; ngan hon muc tieu "
                     "vi thieu bang chung hinh anh."
            )
            TASKS[task_id]["result"] = script_text
            TASKS[task_id]["analysis_method"] = "image-first-video-verified-v2"
            fallback_scenes = [s for s in flat_final_scenes if s.get('visual_fallback')]
            TASKS[task_id]['visual_fallback_count'] = len(fallback_scenes)
            TASKS[task_id]['final_visual_fallback_reports'] = [s['visual_fallback'] for s in fallback_scenes]
            TASKS[task_id]['video_verified_scene_count'] = sum(
                bool(s.get('grounding_audit_passed') and s.get('grounding_audit_method') == 'video')
                for s in flat_final_scenes)
            if fallback_scenes:
                TASKS[task_id]['message'] += f' Co {len(fallback_scenes)} cau dung hinh gan dung; xem nhat ky HINH GAN DUNG.'
            TASKS[task_id]["generation_calls"] = pass1_calls + repair_calls
            TASKS[task_id]["pass1_calls"] = pass1_calls
            TASKS[task_id]["repair_calls"] = repair_calls
            TASKS[task_id]["repair_fallback_used"] = repair_fallback_used
            TASKS[task_id]["issues_before_repair"] = len(issues_before)
            TASKS[task_id]["issues_after_repair"] = len(
                inspect_three_part_payload(final_payload)
            )
            TASKS[task_id]["narration_quality_issues"] = len(
                final_quality_issues
            )
            TASKS[task_id]["story_parts"] = final_payload["parts"]
            TASKS[task_id]["timestamp_modes"] = story_plan.get("timestamp_modes", [])
            TASKS[task_id]["characters"] = final_payload["characters"]
            TASKS[task_id]["target_minutes"] = target_recap_minutes
            TASKS[task_id]["base_target_words"] = base_target_recap_words
            TASKS[task_id]["target_words"] = target_recap_words
            TASKS[task_id]["target_scenes"] = total_scene_count
            TASKS[task_id]["requested_target_scenes"] = int(
                scene_contract.get("requested_scene_count") or total_scene_count
            )
            TASKS[task_id]["sentence_tolerance_met"] = bool(
                scene_contract.get("sentence_tolerance_met", True)
            )
            TASKS[task_id]["part_scene_targets"] = part_scene_counts
            TASKS[task_id]["part_word_targets"] = part_word_targets
            TASKS[task_id]["min_words"] = min_recap_words
            TASKS[task_id]["max_words"] = max_recap_words
            TASKS[task_id]["actual_words"] = word_count
            TASKS[task_id]['actual_sentences'] = actual_sentence_count
            TASKS[task_id]['target_sentence_count'] = budget['target_sentence_count']
            TASKS[task_id]['sentence_contract'] = story_plan['sentence_contract']
            TASKS[task_id]["actual_scenes"] = event_count
            TASKS[task_id]["length_within_range"] = length_within_range
            TASKS[task_id]["duration_contract_met"] = duration_contract_met
            TASKS[task_id]["model_used"] = selected_model
            TASKS[task_id]["thinking_effort"] = thinking_effort
        _persist_task_if_needed(task_id)
        return

    except Exception as e:
        daily_quota = is_daily_project_gemini_quota_error(e)
        resumable_error = daily_quota or is_transient_gemini_error(e) or str(e).startswith('NARRATION_AUDIT_INCOMPLETE:')
        cleanup_analysis_artifacts(preserve_local=resumable_error)
        if daily_quota:
            raise
        traceback.print_exc()
        err_msg = str(e)
        append_task_log(
            task_id,
            "ERROR",
            "Phan tich Gemini dung; chi tiet loi goc duoc giu de phan biet "
            "gioi han moi phut, moi request, quota ngay hoac loi may chu.",
            gemini_error_chain_text(e) + '\nTRACEBACK:\n' + traceback.format_exc(),
        )
        if is_per_minute_gemini_quota_error(e):
            user_msg = (
                "Gemini đang vượt giới hạn token/yêu cầu theo phút của API hiện tại. "
                "Đây không phải quota ngày; hãy chờ đúng retryDelay rồi chạy tiếp."
            )
        elif "token" in err_msg.lower() and any(
            marker in err_msg.lower() for marker in (
                "context", "input token", "too many", "maximum", "limit",
            )
        ):
            user_msg = (
                "Một lượt gửi Gemini vượt giới hạn ngữ cảnh/token đầu vào của model."
            )
        elif re.search(r"\b(?:429|RESOURCE_EXHAUSTED|quota)\b", err_msg, re.IGNORECASE):
            user_msg = "Gemini từ chối vì giới hạn quota: " + err_msg[:220]
        elif is_gemini_auth_error(e):
            user_msg = "API Key không hợp lệ! Vui lòng kiểm tra lại key."
        elif "WinError 10013" in err_msg or "forbidden by its access permissions" in err_msg:
            user_msg = (
                "Windows đang chặn kết nối HTTPS của tiến trình tool tới Gemini. "
                "Hãy chạy tool ngoài sandbox và cho phép Python qua Firewall/VPN."
            )
        else:
            user_msg = f"Lỗi: {err_msg[:300]}"
        partial_result = publish_partial_script(user_msg)
        with task_lock:
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["message"] = user_msg
            TASKS[task_id]["result"] = partial_result
            TASKS[task_id]["partial_result"] = True
        _persist_task_if_needed(task_id)


def analyze_video_bg(
    task_id, api_keys, model_name, video_filename, style, custom_prompt,
    enable_multi_voice=False, review_minutes="auto", groq_api_key="",
    thinking_effort="low", subtitle_text="", voice_rate=DEFAULT_VOICE_RATE,
):
    """Run analysis by API priority and rotate only on daily project quota."""
    keys = [str(key or "").strip() for key in (api_keys or []) if str(key or "").strip()]
    if not keys:
        update_task_fields(task_id, status='failed', message='Khong co API Key Gemini hop le.')
        return

    exhausted = []
    api_key_states = [
        {"priority": index, "status": "pending"}
        for index in range(1, len(keys) + 1)
    ]
    update_task_fields(task_id, api_key_states=copy.deepcopy(api_key_states))
    for index, api_key in enumerate(keys, start=1):
        api_key_states[index - 1]["status"] = "active"
        with task_lock:
            TASKS[task_id]["active_api_priority"] = index
            TASKS[task_id]["api_key_count"] = len(keys)
            TASKS[task_id]["api_key_states"] = copy.deepcopy(api_key_states)
            TASKS[task_id]["message"] = (
                f"Dang dung Gemini API {index}/{len(keys)}..."
            )
        try:
            _analyze_video_bg_once(
                task_id,
                api_key,
                model_name,
                video_filename,
                style,
                custom_prompt,
                enable_multi_voice,
                review_minutes,
                groq_api_key,
                thinking_effort,
                subtitle_text,
                voice_rate,
            )
            api_key_states[index - 1]["status"] = "usable"
            update_task_fields(
                task_id, api_key_states=copy.deepcopy(api_key_states)
            )
            return
        except Exception as error:
            raw_error = gemini_error_chain_text(error)
            append_task_log(
                task_id,
                "ERROR",
                f"Gemini API {index} gap loi; da luu nguyen loai quota de chan doan.",
                raw_error[:2000],
            )
            if not is_daily_project_gemini_quota_error(error):
                api_key_states[index - 1]["status"] = "error"
                update_task_fields(
                    task_id, api_key_states=copy.deepcopy(api_key_states)
                )
                message = f"Lỗi: {str(error)[:300]}"
                with task_lock:
                    partial = TASKS[task_id].get("result", "")
                    TASKS[task_id]["status"] = "failed"
                    TASKS[task_id]["message"] = message
                    TASKS[task_id]["result"] = script_with_error(partial, message)
                    TASKS[task_id]["partial_result"] = True
                return
            exhausted.append(index)
            api_key_states[index - 1]["status"] = "exhausted"
            update_task_fields(
                task_id, api_key_states=copy.deepcopy(api_key_states)
            )
            if index < len(keys):
                with task_lock:
                    TASKS[task_id]["progress"] = 5
                    TASKS[task_id]["message"] = (
                        f"API {index} da het quota ngay; dang khoi dong lai bang "
                        f"API {index + 1}..."
                    )

    # No API remains to resume with; release the local proxy kept only for an
    # in-task API handoff. Check the path is the task-owned temp directory.
    with task_lock:
        artifact_checkpoint = copy.deepcopy(
            TASKS.get(task_id, {}).get("analysis_artifact_checkpoint") or {}
        )
    staging_dir = str(artifact_checkpoint.get("staging_dir") or "")
    try:
        resolved_staging = os.path.realpath(staging_dir)
        resolved_temp = os.path.realpath(tempfile.gettempdir())
        expected_prefix = f"gemini_{task_id[:8]}_"
        if (
            staging_dir and os.path.isdir(resolved_staging)
            and os.path.commonpath([resolved_staging, resolved_temp]) == resolved_temp
            and os.path.basename(resolved_staging).startswith(expected_prefix)
        ):
            shutil.rmtree(resolved_staging, ignore_errors=True)
    except (OSError, ValueError):
        pass
    update_task_fields(task_id, analysis_artifact_checkpoint={})

    final_message = (
        "Tất cả Gemini API đã hết quota ngày: "
        + ", ".join(f"API {index}" for index in exhausted)
        + ". Hãy chờ quota reset hoặc thay API Key khác."
    )
    with task_lock:
        partial = TASKS[task_id].get("result", "")
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["message"] = final_message
        TASKS[task_id]["result"] = script_with_error(partial, final_message)
        TASKS[task_id]["partial_result"] = True

def render_video_bg(task_id, input_video_name, bgm_name, bgm_volume, voice_name, voice_rate, sentences, options, export_mode, export_directory=""):
    def update_prog(prog, msg):
        update_task_fields(task_id, progress=prog, message=msg, heartbeat_at=time.time())
        append_task_log(task_id, "STEP", f"{prog}% - {msg}")

    def diagnostic_log(level, message, details=""):
        append_task_log(task_id, level, message, details)

    try:
        diagnostic_log("INFO", f"Bat dau render: mode={export_mode}, video={input_video_name}, so_dong={len(sentences)}, voice={voice_name}, rate={voice_rate}, pid={os.getpid()}")
        diagnostic_log("INFO", f"Nhat ky day du: {_task_log_path(task_id)}")
        input_video_path = os.path.join(WORKSPACE_DIR, input_video_name)
        
        bgm_path = ""
        if bgm_name:
            bgm_path = os.path.join(WORKSPACE_DIR, "static", "bgm", bgm_name)

        if export_mode == "capcut_zip":
            res_url = export_capcut_package(
                task_id,
                input_video_path,
                bgm_path,
                bgm_volume,
                voice_name,
                voice_rate,
                sentences,
                options,
                update_prog,
                diagnostic_log
            )
        else:
            renderer = render_review_video_ffmpeg
            render_input = sentences
            if options.get('render_engine') == 'structured_v2':
                from structured_render import render_structured_video
                renderer = render_structured_video
                render_input = options['structured_plan']
                diagnostic_log('INFO', 'Render mới (JSON trực tiếp). Có thể chọn Render cũ ở bước Xuất video.')
            res_url = renderer(
                task_id,
                input_video_path,
                bgm_path,
                bgm_volume,
                voice_name,
                voice_rate,
                render_input,
                options,
                update_prog,
                diagnostic_log
            )

        update_task_fields(task_id, narration_timing=options.get('_narration_timing'),
                           narration_budget=options.get('narration_budget'))
        export_path, export_error = copy_result_to_export_directory(res_url, export_directory)
        if export_error:
            update_task_fields(
                task_id,
                status="failed",
                progress=100,
                message=f"Render da xong nhung khong luu duoc file: {export_error}",
                result=res_url,
                export_path="",
                export_error=export_error,
                heartbeat_at=time.time(),
            )
            diagnostic_log("ERROR", f"Khong the xuat file ra may: {export_error}")
            return
        update_task_fields(
            task_id,
            status="success",
            progress=100,
            message="Xuất video thành công!",
            result=res_url,
            export_path=export_path,
            export_error=export_error,
            heartbeat_at=time.time(),
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        diagnostic_log("ERROR", f"Render that bai: {type(e).__name__}: {e}", error_trace)
        # Tao thong bao loi than thien cho user
        err_msg = str(e)
        if options.get('render_engine') == 'structured_v2':
            err_msg += ' Đã giữ cache. Có thể chọn Render cũ ở bước Xuất video để quay lại.'
        elif "ffmpeg" in err_msg.lower() or "returncode" in err_msg.lower():
            err_msg = "Loi xu ly video (FFmpeg). Vui long thu lai hoac kiem tra video goc."
        elif "edge_tts" in err_msg.lower() or "tts" in err_msg.lower():
            err_msg = "Loi tao giong doc AI. Kiem tra ket noi internet va thu lai."
        elif "genai" in err_msg.lower() or "api" in err_msg.lower():
            err_msg = "Loi ket noi API Gemini. Kiem tra API Key va thu lai."
        update_task_fields(
            task_id,
            status="failed",
            message=f"Loi: {err_msg}",
            heartbeat_at=time.time(),
        )
    finally:
        diagnostic_log("INFO", "Tac vu render da ket thuc khoi worker (thanh cong hoac that bai).")

# =====================================================================
# FLASK ROUTING & API ENDPOINTS
# =====================================================================
@app.route("/")
def index():
    return render_template(
        "index.html",
        default_key=DEFAULT_API_KEY,
        gemini_models=_load_gemini_model_cache(),
    )

@app.route("/source-video/<path:filename>")
def source_video(filename):
    """Serve an uploaded source video stored at the workspace root."""
    if os.path.splitext(filename)[1].lower() not in ALLOWED_VIDEO_EXT:
        return jsonify({"success": False, "message": "Dinh dang video khong hop le."}), 404
    return send_from_directory(WORKSPACE_DIR, filename)

@app.route("/api/export-directory", methods=["GET", "POST"])
def export_directory():
    if request.method == "GET":
        configured_dir = get_configured_export_directory()
        valid, validation_error = check_export_directory_writable(configured_dir)
        return jsonify({
            "directory": configured_dir,
            "valid": valid,
            "message": "" if valid or not configured_dir else validation_error
        })

    # The tool runs locally, so this opens the Windows folder picker.
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(
            parent=root,
            initialdir=get_export_directory() or WORKSPACE_DIR,
            title="Chon thu muc xuat video"
        )
        root.destroy()
    except Exception as exc:
        return jsonify({"success": False, "message": f"Khong mo duoc hop chon thu muc: {exc}"}), 500

    if not selected_dir:
        return jsonify({"success": False, "cancelled": True, "message": "Ban chua chon thu muc."})

    selected_dir = os.path.abspath(os.path.normpath(str(selected_dir).strip()))
    if not os.path.isdir(selected_dir):
        return jsonify({"success": False, "message": "Thu muc vua chon khong ton tai."}), 400

    # Check write permission before a long render starts.
    writable, validation_error = check_export_directory_writable(selected_dir)
    if not writable:
        return jsonify({"success": False, "message": validation_error}), 400

    settings = _load_settings()
    settings["export_directory"] = selected_dir
    try:
        _save_settings(settings)
    except OSError as exc:
        return jsonify({"success": False, "message": f"Khong luu duoc cai dat: {exc}"}), 500
    return jsonify({"success": True, "directory": selected_dir})

@app.route("/api/check-key", methods=["POST"])
def check_key():
    data = request.json or {}
    api_key = data.get("api_key", "").strip()
    key_error = validate_gemini_api_key(api_key)
    if key_error:
        return jsonify({"valid": False, "message": key_error}), 400
        
    try:
        client = genai.Client(api_key=api_key)
        
        # Chi dung models.list() de xac thuc key (MIEN PHI, khong ton quota)
        models_list = list(client.models.list())
        available = describe_gemini_models(models_list)
        _save_gemini_model_cache(available)
        return jsonify({
            "valid": True,
            "usable": True,
            "status": "usable",
            "message": "Kết nối API Gemini thành công!",
            "models": available,
        })
    except Exception as e:
        err_msg = gemini_error_chain_text(e)
        if is_gemini_auth_error(e):
            return jsonify({
                "valid": False,
                "usable": False,
                "status": "invalid",
                "message": "API Key không hợp lệ! Kiểm tra lại tại https://aistudio.google.com/apikey",
            })
        if "WinError 10013" in err_msg or "forbidden by its access permissions" in err_msg:
            return jsonify({
                "valid": False,
                "usable": False,
                "status": "network_error",
                "message": (
                    "Windows đang chặn kết nối HTTPS tới Gemini. Hãy chạy tool ngoài sandbox "
                    "và cho phép python.exe qua Firewall/VPN."
                ),
            })
        return jsonify({
            "valid": False,
            "usable": False,
            "status": "connection_error",
            "message": f"Không xác thực được kết nối Gemini: {err_msg[:200]}",
        })

@app.route("/api/check-groq-key", methods=["POST"])
def check_groq_key():
    """Validate Groq credentials without spending audio quota."""
    import requests

    key = str((request.json or {}).get("api_key", "") or "").strip()
    if not key:
        return jsonify({"valid": False, "message": "Chua nhap Groq API Key."}), 400
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        if response.status_code in (401, 403):
            return jsonify({"valid": False, "message": "Groq API Key khong hop le."}), 401
        response.raise_for_status()
        model_ids = {
            str(row.get("id", ""))
            for row in (response.json().get("data") or [])
            if isinstance(row, dict)
        }
        if not any("whisper-large-v3" in name for name in model_ids):
            return jsonify({
                "valid": False,
                "message": "Tai khoan Groq chua co Whisper Large V3.",
            }), 400
        return jsonify({"valid": True, "message": "Ket noi Groq Whisper thanh cong."})
    except Exception as exc:
        return jsonify({
            "valid": False,
            "message": f"Khong ket noi duoc Groq: {str(exc)[:160]}",
        }), 502


@app.route("/api/list-videos", methods=["GET"])
def list_videos():
    try:
        return jsonify(normalize_existing_video_filenames(WORKSPACE_DIR))
    except OSError as exc:
        return jsonify({
            "success": False,
            "message": f"Khong the doi ten video sang ASCII: {exc}",
        }), 500


@app.route("/api/video-info", methods=["GET"])
def video_info():
    """Return source duration and its editable one-fifth recap suggestion."""
    requested_name = os.path.basename(
        str(request.args.get("video_name", "") or "").replace("\\", "/")
    )
    if (not requested_name
            or os.path.splitext(requested_name)[1].lower() not in ALLOWED_VIDEO_EXT):
        return jsonify({"success": False, "message": "Ten video khong hop le."}), 400
    video_path = os.path.abspath(os.path.join(WORKSPACE_DIR, requested_name))
    workspace_path = os.path.abspath(WORKSPACE_DIR)
    if (os.path.commonpath([workspace_path, video_path]) != workspace_path
            or not os.path.isfile(video_path)):
        return jsonify({"success": False, "message": "Khong tim thay video."}), 404
    try:
        from video_renderer import get_video_info
        duration_seconds, _, _ = get_video_info(video_path)
        duration_minutes = max(0.0, float(duration_seconds) / 60.0)
        suggested_minutes = max(0.1, duration_minutes * 0.20)
        return jsonify({
            "success": True,
            "source_duration_seconds": round(float(duration_seconds), 3),
            "source_duration_minutes": round(duration_minutes, 2),
            "suggested_review_minutes": round(suggested_minutes, 1),
        })
    except Exception as exc:
        return jsonify({
            "success": False, "message": f"Khong doc duoc video: {exc}",
        }), 400

@app.route('/api/list-bgm', methods=['GET'])
def list_bgm():
    bgm_dir = os.path.join(app.static_folder, 'bgm')
    os.makedirs(bgm_dir, exist_ok=True)
    files = [f for f in os.listdir(bgm_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.aac'))]
    return jsonify(files)

@app.route('/api/list-sfx', methods=['GET'])
def list_sfx():
    sfx_dir = os.path.join(app.static_folder, 'sfx')
    os.makedirs(sfx_dir, exist_ok=True)
    files = [f for f in os.listdir(sfx_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.aac', '.ogg'))]
    return jsonify(files)

@app.route("/api/upload-file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Khong tim thay tep."})
        
    file = request.files['file']
    file_type = request.form.get("type", "video")
    
    if file.filename == '':
        return jsonify({"success": False, "message": "Ten tep khong hop le."})
        
    if file:
        original_filename = os.path.basename(file.filename.replace("\\", "/"))
        filename = ascii_upload_filename(original_filename)
        ext = os.path.splitext(filename)[1].lower()
        
        # Kiem tra dinh dang file
        if file_type == "video":
            if ext not in ALLOWED_VIDEO_EXT:
                return jsonify({"success": False, "message": f"Dinh dang {ext} khong ho tro. Chi chap nhan: {', '.join(ALLOWED_VIDEO_EXT)}"})
            target_path = available_upload_path(WORKSPACE_DIR, filename)
        else:
            if ext not in ALLOWED_AUDIO_EXT:
                return jsonify({"success": False, "message": f"Dinh dang {ext} khong ho tro. Chi chap nhan: {', '.join(ALLOWED_AUDIO_EXT)}"})
            target_path = available_upload_path(
                os.path.join(WORKSPACE_DIR, "static", "bgm"), filename
            )

        filename = os.path.basename(target_path)
            
        file.save(target_path)
        return jsonify({
            "success": True,
            "filename": filename,
            "original_filename": original_filename,
            "renamed": filename != original_filename,
        })
        
    return jsonify({"success": False, "message": "Loi khong xac dinh khi luu file."})

@app.route("/api/analyze-video", methods=["POST"])
def analyze_video():
    data = request.json or {}
    raw_api_keys = data.get("api_keys", [])
    if not isinstance(raw_api_keys, list):
        raw_api_keys = []
    legacy_key = str(data.get("api_key", "") or "").strip()
    candidates = raw_api_keys if raw_api_keys else [legacy_key or DEFAULT_API_KEY]
    api_keys = []
    for slot_index, raw_key in enumerate(candidates[:3], start=1):
        key = str(raw_key or "").strip()
        if not key:
            continue
        if key in api_keys:
            return jsonify({
                "success": False,
                "message": f"API {slot_index} bi trung voi API truoc do.",
            }), 400
        api_keys.append(key)
    model_name = data.get("model_name", "gemini-3.1-flash-lite")
    groq_api_key = str(data.get("groq_api_key", "") or "").strip()
    subtitle_text = str(data.get("subtitle_text", "") or "")
    if len(subtitle_text) > 5_000_000:
        return jsonify({
            "success": False,
            "message": "File phu de qua lon; gioi han 5 MB van ban.",
        }), 400
    thinking_effort = str(data.get("thinking_effort", "low") or "low").strip().lower()
    if thinking_effort not in {"low", "medium", "high"}:
        return jsonify({
            "success": False, "message": "Muc no luc Gemini khong hop le.",
        }), 400
    video_filename = os.path.basename(
        str(data.get("video_name", "")).replace("\\", "/")
    )
    style = data.get("style", "dramatic")
    custom_prompt = data.get("custom_prompt", "").strip()
    enable_multi_voice = data.get("enable_multi_voice", False)
    project_id = str(data.get("project_id", "")).strip() or str(uuid.uuid4())
    review_minutes_raw = data.get("review_minutes", "auto")
    try:
        planned_voice_rate = normalize_voice_rate(data.get('voice_rate', DEFAULT_VOICE_RATE))
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    if str(review_minutes_raw).lower() == "auto":
        review_minutes = "auto"
    else:
        try:
            review_minutes = float(review_minutes_raw)
        except (TypeError, ValueError):
            return jsonify({"success": False, "message": "So phut recap khong hop le."}), 400
        if not math.isfinite(review_minutes) or review_minutes <= 0:
            return jsonify({
                "success": False, "message": "So phut recap phai lon hon 0.",
            }), 400

    if not api_keys:
        return jsonify({"success": False, "message": "Chua nhap Gemini API Key."}), 400
    if not groq_api_key:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return jsonify({
                "success": False,
                "message": (
                    "Hay nhap Groq API Key. Neu khong dung Groq, can cai "
                    "faster-whisper de chay fallback local."
                ),
            }), 400
    for index, api_key in enumerate(api_keys, start=1):
        key_error = validate_gemini_api_key(api_key)
        if key_error:
            return jsonify({
                "success": False,
                "message": f"API {index}: {key_error}",
            }), 400
    
    if not video_filename:
        return jsonify({"success": False, "message": "Chưa chọn video gốc."})

    try:
        video_filename = normalize_video_file(WORKSPACE_DIR, video_filename)
    except OSError as exc:
        return jsonify({
            "success": False,
            "message": f"Khong the doi ten video sang ASCII: {exc}",
        }), 500
        
    from analysis_resume import request_fingerprint, load_recovery, remember_analysis
    recovery_directory = os.path.join(WORKSPACE_DIR, '.cache', 'analysis_resume')
    try:
        fingerprint = request_fingerprint(os.path.join(WORKSPACE_DIR, video_filename), {
            'model': model_name, 'thinking_effort': thinking_effort, 'style': style,
            'custom_prompt': custom_prompt, 'review_minutes': review_minutes,
            'voice_rate': planned_voice_rate, 'subtitle_text': subtitle_text,
            'evidence_contract_version': EVIDENCE_CONTRACT_VERSION, 'enable_multi_voice': enable_multi_voice})
    except OSError:
        return jsonify(success=False, message='Khong tim thay video goc de phan tich.'), 400
    previous_task_id, recovered = load_recovery(recovery_directory, fingerprint, load_task_state)
    task_id = create_task(project_id, video_filename, persistent=True, task_kind="analysis")
    update_task_fields(task_id, **recovered, analysis_request_fingerprint=fingerprint,
                      resumed_from_task_id=previous_task_id, resume_available=True)
    remember_analysis(recovery_directory, fingerprint, task_id)
    if recovered:
        append_task_log(task_id, 'INFO',
            'Tiep tuc tu checkpoint cung video va cau hinh; giu bang chung, ke hoach va nhom cau da dat.',
            f'Task truoc: {previous_task_id}')
    with task_lock:
        PROJECTS[project_id] = {
            "video_name": video_filename,
            "analysis_task_id": task_id,
            "created_at": time.time(),
        }
    
    # Khởi chạy tiểu trình phân tích nền
    thread = threading.Thread(
        target=analyze_video_bg,
        args=(
            task_id, api_keys, model_name, video_filename, style, custom_prompt,
            enable_multi_voice, review_minutes, groq_api_key, thinking_effort,
            subtitle_text, planned_voice_rate,
        )
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "task_id": task_id, "resumed": bool(recovered)})

@app.route("/api/split-sentences", methods=["POST"])
def split_sentences():
    data = request.json or {}
    raw_text = data.get("text", "")
    text = (
        "\n".join(str(item) for item in raw_text)
        if isinstance(raw_text, list)
        else str(raw_text or "")
    ).strip()
    if not text:
        return jsonify([])

    try:
        sentences = split_text_into_sentences(text)
    except Exception as exc:
        # A completed Gemini script must remain editable even if an unusual
        # token trips the richer TTS parser. Preserve each TIME row verbatim.
        print(f"[!] Loi parser phan doan; dung fallback theo dong: {exc}", flush=True)
        sentences = [
            row.strip() for row in format_timeline_rows(text).splitlines()
            if row.strip()
        ]
    sentences = [str(item).strip() for item in sentences if str(item).strip()]
    if not sentences:
        sentences = [text]
    return jsonify(sentences)

@app.route("/api/preview-tts", methods=["POST"])
def preview_tts():
    data = request.json or {}
    text = data.get("text", "")
    voice = data.get("voice", "vi-VN-HoaiMyNeural")
    rate = data.get("rate", DEFAULT_VOICE_RATE)
    
    if not text:
        return jsonify({"success": False, "message": "Nội dung câu trống."})
        
    filename = f"preview_{uuid.uuid4()}.mp3"
    filepath = os.path.join(WORKSPACE_DIR, "static", "temp_preview", filename)
    
    async def run_tts():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await asyncio.wait_for(communicate.save(filepath), timeout=15.0)
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_tts())
    except asyncio.TimeoutError:
        return jsonify({"success": False, "message": "TTS qua thoi gian cho (15s). Thu lai."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    finally:
        loop.close()
        
    return jsonify({"success": True, "url": f"/static/temp_preview/{filename}"})

@app.route("/api/render-video", methods=["POST"])
def render_video():
    from render_guard import RenderFileLock, RenderBusy, render_fingerprint, active_render
    from task_store import _atomic_write_json
    registry_path = os.path.join(WORKSPACE_DIR, '.jobs', 'active_render.json')
    fingerprint = render_fingerprint(request.json or {})
    try:
        with RenderFileLock(os.path.join(WORKSPACE_DIR, '.jobs', 'render_admission.lock')):
            active = active_render(registry_path, load_task_state)
            if active:
                if active.get('fingerprint') == fingerprint:
                    return jsonify({'success': True, 'task_id': active['task_id'], 'reused': True})
                return jsonify({'success': False, 'active_task_id': active['task_id'],
                    'message': 'Máy đang render một video. Hãy chờ tác vụ hiện tại hoàn tất trước khi xuất tiếp.'}), 409
            # A manually launched worker can own the execution lock even when
            # it was not registered by this web process.
            with RenderFileLock(os.path.join(WORKSPACE_DIR, '.jobs', 'render_worker.lock'), timeout=0):
                pass
            result = _create_render_video_request()
            response = app.make_response(result)
            body = response.get_json(silent=True) or {}
            if body.get('success') and body.get('task_id'):
                _atomic_write_json(registry_path, {'task_id': body['task_id'], 'fingerprint': fingerprint})
            return result
    except RenderBusy as exc:
        return jsonify({'success': False, 'message': str(exc)}), 409


def _create_render_video_request():
    data = request.json or {}
    video_name = data.get("video_name", "")
    bgm_name = data.get("bgm_name", "")
    bgm_volume = float(data.get("bgm_volume", 12)) / 100.0
    voice_name = data.get("voice_name", "vi-VN-HoaiMyNeural")
    try:
        voice_rate = normalize_voice_rate(data.get('voice_rate', DEFAULT_VOICE_RATE))
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    sentences = data.get("sentences", [])
    raw_sentence_rows = (
        [str(item) for item in sentences]
        if isinstance(sentences, list) else [str(sentences or "")]
    )

    def is_timeline_control_row(row):
        """VOICE/SFX rows control audio and must not count as visual ranges."""
        _, _, body = extract_time_range(str(row or ""))
        return bool(re.fullmatch(
            r"\s*(?:<|\[)(?:VOICE|SFX)=[^>\]]+(?:>|\])\s*",
            body,
            flags=re.IGNORECASE,
        ))

    raw_timeline_text = "\n".join(
        row for row in raw_sentence_rows if not is_timeline_control_row(row)
    )
    raw_timeline_text = drop_beats_overlapping_next_time(raw_timeline_text)
    cleaned_visual_rows = iter(raw_timeline_text.splitlines())
    rebuilt_rows = []
    for row in raw_sentence_rows:
        if is_timeline_control_row(row):
            rebuilt_rows.append(row)
        else:
            rebuilt_rows.append(next(cleaned_visual_rows, row))
    # Repair stale/browser timelines before persisting a render payload. Older
    # parsers collapsed timestamps beyond 99:59 into one oversized TTS item.
    if rebuilt_rows:
        sentences = split_text_into_sentences("\n".join(rebuilt_rows))
    project_id = str(data.get("project_id", "")).strip()
    review_minutes_raw = data.get("review_minutes", "auto")
    
    # Cấu hình video
    options = {
        "mirror": data.get("mirror", True),
        "zoom_factor": float(data.get("zoom_factor", 1.08)),
        "speed_factor": float(data.get("speed_factor", 1.0)),
        "contrast": float(data.get("contrast", 1.05)),
        "brightness": float(data.get("brightness", 1.0)),
        "font_size": int(data.get("font_size", 42)),
        "font_color": data.get("font_color", "#FFFFFF"),
        "bg_opacity": int(data.get("bg_opacity", 150)),
        "font_path": "C:\\Windows\\Fonts\\arialbd.ttf",
        "aspect_ratio": data.get("aspect_ratio", "9:16"),
        "enable_subtitles": data.get("enable_subtitles", True),
        "enable_zoom_crop": data.get("enable_zoom_crop", True),
        "enable_ken_burns": data.get("enable_ken_burns", False)
    }
    if str(review_minutes_raw).lower() != "auto":
        try:
            review_minutes = float(review_minutes_raw)
        except (TypeError, ValueError):
            return jsonify({"success": False, "message": "So phut recap khong hop le."}), 400
        if not math.isfinite(review_minutes) or review_minutes <= 0:
            return jsonify({"success": False, "message": "So phut recap phai lon hon 0."}), 400
        options["review_minutes"] = review_minutes
    
    if not video_name:
        return jsonify({"success": False, "message": "Chưa chọn video gốc."})
    if not sentences:
        return jsonify({"success": False, "message": "Kịch bản trống."})

    # Require a durable output destination before spending time on rendering.
    # Capture the exact path in the worker payload so changing settings while a
    # render is running cannot redirect or lose its completed file.
    export_directory = get_export_directory()
    if not export_directory:
        return jsonify({
            "success": False,
            "needs_export_directory": True,
            "message": "Chưa chọn thư mục xuất hợp lệ. Hãy chọn nơi lưu video trước khi render."
        }), 400
    writable, validation_error = check_export_directory_writable(export_directory)
    if not writable:
        return jsonify({
            "success": False,
            "needs_export_directory": True,
            "message": f"Không thể ghi vào thư mục xuất. Hãy chọn thư mục khác: {validation_error}"
        }), 400

    # Validate raw editor ranges before older parsing code can clamp, split or
    # interpolate a malformed timestamp into a plausible-looking wrong scene.
    try:
        from video_renderer import get_video_info
        source_duration, _, _ = get_video_info(os.path.join(WORKSPACE_DIR, video_name))
        planned_minutes = options.get('review_minutes', max(1.5, source_duration / 300.0))
        options['narration_budget'] = narration_budget(planned_minutes, voice_rate)
        validate_timeline_ranges(raw_timeline_text, source_duration)
        validate_render_visual_grounding(raw_timeline_text, source_duration)
    except (OSError, ValueError, RuntimeError) as exc:
        return jsonify({
            "success": False,
            "message": f"Timeline khóa không hợp lệ: {exc}"
        }), 400

    clean_timeline, range_starts, range_ends, visual_groups = parse_sentences_and_visual_beats(sentences)
    previous_end = -1.0
    validation_beats = []
    for index, (text, groups) in enumerate(zip(clean_timeline, visual_groups), start=1):
        if text.startswith(("<VOICE=", "<SFX=")):
            continue
        if groups:
            validation_beats.extend(
                (index, beat["text"], beat.get("start"), beat.get("end"))
                for beat in groups
            )
        else:
            validation_beats.append((index, text, range_starts[index - 1], range_ends[index - 1]))

    for index, text, start, end in validation_beats:
        if start is None or end is None:
            return jsonify({
                "success": False,
                "message": (
                    f"Dòng {index} chưa có TIME_END. Hãy tạo lại kịch bản Gemini theo dạng "
                    "[TIME=mm:ss-mm:ss] trước khi xuất."
                )
            }), 400
        if end <= start:
            return jsonify({
                "success": False,
                "message": f"Dòng {index} có TIME_END không lớn hơn TIME_START."
            }), 400
        if start < previous_end:
            return jsonify({
                "success": False,
                "message": f"Dòng {index} bị đi lùi hoặc chồng lên khoảng thời gian trước."
            }), 400
        previous_end = end
        
    if not project_id:
        return jsonify({
            "success": False,
            "message": "Phien du an khong hop le. Hay tai lai trang va tao lai kich ban Gemini."
        }), 409
    with task_lock:
        project = PROJECTS.get(project_id)
        if project is None and os.path.isfile(os.path.join(WORKSPACE_DIR, video_name)):
            # A server restart clears RAM-only project metadata. The browser
            # still owns the validated script/project id, so restore the local
            # project instead of forcing an expensive Gemini regeneration.
            project = {
                "video_name": video_name,
                "analysis_task_id": "restored-for-render",
                "created_at": time.time(),
            }
            PROJECTS[project_id] = project
    if not project or project.get("video_name") != video_name:
        return jsonify({
            "success": False,
            "message": "Video va kich ban khong cung mot phien. Hay tao lai kich ban cho video dang chon."
        }), 409

    engine = str(data.get('render_engine', 'legacy_v1'))
    if engine not in {'legacy_v1', 'structured_v2'}:
        return jsonify({'success': False, 'message': 'Phiên bản render không hợp lệ.'}), 400
    options['render_engine'] = engine
    analysis_id = str(data.get('analysis_task_id') or project.get('analysis_task_id') or '')
    with task_lock:
        contract_analysis = copy.deepcopy(TASKS.get(analysis_id))
    if contract_analysis is None and re.fullmatch(r'[a-fA-F0-9-]+', analysis_id):
        contract_analysis = load_task_state(analysis_id)
    contract = (contract_analysis or {}).get('sentence_contract')
    if contract:
        try:
            from narrative_contract import verify_script
            if (contract_analysis.get('project_id') != project_id or contract_analysis.get('video_name') != video_name
                    or contract_analysis.get('status') != 'success'):
                raise ValueError('Kịch bản chưa hoàn thành đúng hợp đồng số câu.')
            verify_script('\n'.join(sentences), contract['target_count'], contract['target_words'])
            options['sentence_contract'] = contract
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify({'success': False, 'message': str(exc)}), 400
    if engine == 'structured_v2':
        if data.get('export_mode', 'full_mp4') != 'full_mp4':
            return jsonify({'success': False, 'message': 'Render mới hỗ trợ MP4. Chọn Render cũ để xuất CapCut.'}), 400
        try:
            from structured_render import make_render_plan, prepare_plan_inputs
            analysis_id = str(data.get('analysis_task_id') or project.get('analysis_task_id') or '')
            with task_lock:
                analysis = copy.deepcopy(TASKS.get(analysis_id))
            analysis = analysis or load_task_state(analysis_id)
            if (not analysis or analysis.get('project_id') != project_id
                    or analysis.get('video_name') != video_name):
                raise ValueError('Không tìm thấy phân tích đúng dự án. Tạo recap mới hoặc chọn Render cũ.')
            structured = make_render_plan(analysis.get('matched_narration_payload') or {}, source_duration)
            # The editable legacy text is checked only at this UI boundary.
            # The worker receives the JSON plan, never reparses TIME/BEAT.
            edited, _, _, edited_groups = parse_sentences_and_visual_beats(sentences, strict_source_ranges=True)
            if len(edited) != len(structured['situations']):
                raise ValueError('Số đoạn đã sửa khác dữ liệu phân tích. Chọn Render cũ hoặc tạo lại recap.')
            for situation, text, group in zip(structured['situations'], edited, edited_groups):
                if (len(group) != len(situation['shots']) or any(
                        abs(beat['start'] - shot['start']) > .02 or abs(beat['end'] - shot['end']) > .02
                        for beat, shot in zip(group, situation['shots']))):
                    raise ValueError('Nhóm hình trong trình sửa đã thay đổi. Chọn Render cũ hoặc tạo lại recap.')
                situation['text'] = text
            prepare_plan_inputs(structured)
            options['structured_plan'] = structured
            options['tts_policy'] = 'bounded_v2'
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return jsonify({'success': False, 'message': str(exc)}), 400

    task_id = create_task(
        project_id, video_name, persistent=True, task_kind="render"
    )

    export_mode = data.get("export_mode", "full_mp4")

    # Save inputs first, then launch a detached worker. Rendering is no longer
    # owned by Flask's lifetime, so a web-server restart cannot kill the job.
    save_task_payload(task_id, {
        "video_name": video_name,
        "bgm_name": bgm_name,
        "bgm_volume": bgm_volume,
        "voice_name": voice_name,
        "voice_rate": voice_rate,
        "sentences": sentences,
        "options": options,
        "export_mode": export_mode,
        "export_directory": export_directory,
    })
    worker_script = os.path.join(WORKSPACE_DIR, "render_worker.py")
    worker_log_path = os.path.join(LOG_DIR, f"worker_{task_id}.log")
    worker_log = open(worker_log_path, "a", encoding="utf-8", buffering=1)
    popen_kwargs = {
        "cwd": WORKSPACE_DIR,
        "stdin": subprocess.DEVNULL,
        "stdout": worker_log,
        "stderr": worker_log,
        "close_fds": True,
        "env": {**os.environ, "PYTHONUTF8": "1"},
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        popen_kwargs["start_new_session"] = True
    try:
        worker = subprocess.Popen([sys.executable, worker_script, task_id], **popen_kwargs)
        if isinstance(worker.pid, int):
            update_task_fields(task_id, worker_pid=worker.pid)
    except Exception as exc:
        update_task_fields(
            task_id,
            status="failed",
            message=f"Khong khoi dong duoc worker render: {exc}",
        )
        return jsonify({"success": False, "message": str(exc)}), 500
    finally:
        worker_log.close()
    
    return jsonify({"success": True, "task_id": task_id})

@app.route("/api/task-status/<task_id>", methods=["GET"])
def task_status(task_id):
    # Analysis runs in a thread inside this Flask process, so RAM is the live
    # source of truth. The persisted file is only a checkpoint and may lag
    # behind progress updates. Render runs in a detached process and therefore
    # must continue to be read from disk and monitored by PID/heartbeat.
    with task_lock:
        task = TASKS.get(task_id)
        internal = {'pending_narration_audit_payload', 'narration_draft_payload',
                    'matched_narration_payload', 'story_parts', 'primary_grounding_responses',
                    'grounding_recovery_responses', 'source_writing_batches',
                    'source_writing_candidates', 'visual_writing_batches',
                    'chapter_prose_batches', 'chapter_sentence_candidates',
                    'continuous_recap_drafts', 'grounded_narration_audit',
                    'grounded_gap_responses', 'full_video_outline', 'evidence_writing_batches',
                    'evidence_writing_candidates', 'evidence_text_audit'}
        live_task = copy.deepcopy({k: v for k, v in task.items()
                                   if k not in internal and not k.endswith('_checkpoint')}) if task else None
    if live_task is not None and live_task.get("task_kind") != "render":
        return jsonify(live_task)

    persisted = load_task_state(task_id)
    if persisted is not None:
        if (persisted.get('task_kind') == 'analysis'
                and persisted.get('status') in ('queued', 'running')):
            from render_guard import process_is_alive
            owner_pid = int(persisted.get('analysis_pid') or 0)
            if owner_pid == os.getpid() or not process_is_alive(owner_pid):
                persisted.update(status='failed', message=(
                    'Phan tich bi gian doan khi backend dung. '
                    'Checkpoint da luu; bam Thu lai de tiep tuc.'))
                save_task_state(task_id, persisted)
        heartbeat_at = float(persisted.get("heartbeat_at") or persisted.get("created_at") or 0)
        if (
            persisted.get("task_kind") == "render"
            and persisted.get("status") in ("queued", "running")
            and time.time() - heartbeat_at > 30
        ):
            worker_pid = int(persisted.get("worker_pid") or 0)
            worker_alive = False
            if worker_pid > 0:
                from render_guard import process_is_alive
                worker_alive = process_is_alive(worker_pid)
            if not worker_alive:
                persisted.update({
                    "status": "failed",
                    "message": "Worker render da dung dot ngot. Bam Thu lai de tiep tuc tu cache TTS.",
                })
                save_task_state(task_id, persisted)
        if persisted.get('task_kind') != 'analysis' or persisted.get('status') not in ('queued', 'running'):
            with task_lock:
                TASKS[task_id] = persisted
        return jsonify({k: v for k, v in persisted.items()
                        if k not in internal and not k.endswith('_checkpoint')})

    with task_lock:
        task = TASKS.get(task_id)
        
    if not task:
        return jsonify({"status": "not_found", "message": "Không tìm thấy tác vụ."})
        
    return jsonify(task)



# =====================================================================
# BACKGROUND TASK: AI VIDEO GENERATOR
# =====================================================================
def generate_ai_video_bg(task_id, api_key, script_text, style_prompt, bgm_name, bgm_volume, voice_name, voice_rate, options, export_directory):
    def update_prog(prog, msg):
        with task_lock:
            TASKS[task_id]["progress"] = prog
            TASKS[task_id]["message"] = msg

    try:
        from tts_engine import split_text_into_sentences, generate_all_tts
        
        update_prog(5, "Đang phân tích kịch bản...")
        sentences = split_text_into_sentences(script_text)
        if not sentences:
            raise Exception("Kịch bản trống hoặc quá ngắn.")
            
        temp_dir = os.path.join(WORKSPACE_DIR, f"temp_{task_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Generate Images
        image_paths = generate_all_images(api_key, sentences, style_prompt, temp_dir, options.get("aspect_ratio", "16:9"), update_prog)
        
        # 2. Generate Audio TTS
        update_prog(40, "Đang tổng hợp giọng nói AI (TTS)...")
        audio_paths, durations = generate_all_tts(sentences, voice_name, voice_rate, temp_dir)
        
        # 3. Resolve BGM Path
        bgm_path = ""
        if bgm_name:
            bgm_path = os.path.join(WORKSPACE_DIR, "static", "bgm", bgm_name)
            
        # 4. Render Final Video
        res_url = render_ai_image_video_ffmpeg(
            task_id,
            sentences,
            durations,
            audio_paths,
            image_paths,
            bgm_path,
            bgm_volume,
            options,
            update_prog
        )
        export_path, export_error = copy_result_to_export_directory(res_url, export_directory)
        with task_lock:
            TASKS[task_id]["status"] = "failed" if export_error else "success"
            TASKS[task_id]["progress"] = 100
            TASKS[task_id]["message"] = (
                f"Render đã xong nhưng không lưu được file: {export_error}"
                if export_error else "Tạo Video Hoạt Hình AI thành công!"
            )
            TASKS[task_id]["result"] = res_url
            TASKS[task_id]["export_path"] = export_path
            TASKS[task_id]["export_error"] = export_error

    except Exception as e:
        import traceback
        traceback.print_exc()
        with task_lock:
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["message"] = f"Lỗi: {str(e)}"


@app.route("/ai-video")
def ai_video():
    return render_template("ai_video.html", default_key=DEFAULT_API_KEY)

@app.route("/api/generate-ai-video", methods=["POST"])
def api_generate_ai_video():
    data = request.json or {}
    api_key = data.get("api_key", "").strip() or DEFAULT_API_KEY
    script_text = data.get("script_text", "").strip()
    style_prompt = data.get("style_prompt", "Vẽ nét đơn giản, màu sắc tươi sáng").strip()
    bgm_name = data.get("bgm_name", "")
    bgm_volume = float(data.get("bgm_volume", 12)) / 100.0
    voice_name = data.get("voice_name", "vi-VN-HoaiMyNeural")
    voice_rate = data.get("voice_rate", DEFAULT_VOICE_RATE)
    
    options = {
        "aspect_ratio": data.get("aspect_ratio", "16:9"),
        "font_size": int(data.get("font_size", 42)),
        "font_color": data.get("font_color", "#FFFFFF"),
        "bg_opacity": int(data.get("bg_opacity", 150))
    }
    
    if not script_text:
        return jsonify({"success": False, "message": "Chưa nhập kịch bản."})
    if not api_key:
        return jsonify({"success": False, "message": "Yêu cầu phải có API Key Gemini."})
    export_directory = get_export_directory()
    writable, validation_error = check_export_directory_writable(export_directory)
    if not writable:
        return jsonify({
            "success": False,
            "needs_export_directory": True,
            "message": f"Chưa chọn thư mục xuất hợp lệ. Hãy chọn nơi lưu video trước khi render. {validation_error}"
        }), 400
        
    task_id = create_task()
    
    thread = threading.Thread(
        target=generate_ai_video_bg,
        args=(task_id, api_key, script_text, style_prompt, bgm_name, bgm_volume, voice_name, voice_rate, options, export_directory)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "task_id": task_id})

if __name__ == "__main__":
    print("[*] Server is running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
