"""Opt-in render v2: structured source ownership, audio-led cuts and durable stages."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from semantic_visuals import seconds


def make_render_plan(payload, source_duration):
    situations, seen, cursor = [], set(), -1.0
    registry = {str(c['id']): c for c in payload.get('characters') or [] if c.get('id')}
    for part in payload.get('parts') or []:
        for scene in part.get('scenes') or []:
            sid, text = str(scene.get('scene_id', '')), str(scene.get('tts', '')).strip()
            if not sid or sid in seen or not text:
                raise ValueError('Render mới: ID tình huống trùng/rỗng hoặc thiếu lời kể.')
            if re.search(r'\[(?:TIME|BEAT)=', text, re.I):
                raise ValueError('Render mới: lời kể không được chứa thẻ TIME/BEAT.')
            seen.add(sid)
            shots = []
            locked = scene.get('locked_visual_shots') or []
            links = scene.get('visual_links') or []
            beats = scene.get('visual_beats') or []
            for index, shot in enumerate(locked):
                start = seconds(shot['start'])
                end = seconds(shot['end'])
                if not (math.isfinite(start) and math.isfinite(end)
                        and 0 <= start < end <= source_duration + .01 and start >= cursor - .01):
                    raise ValueError(f'Render mới: khoảng hình sai/chồng lấn ở {sid}.')
                trigger = ''
                if index:
                    if index < len(links):
                        trigger = str(links[index].get('trigger_text') or '')
                    elif index - 1 < len(beats):
                        trigger = str(beats[index - 1].get('trigger_text') or '')
                    if not trigger or len(re.findall(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)', text, re.I)) != 1:
                        raise ValueError(f'Render mới: thiếu mốc lời cho cảnh {index + 1} ở {sid}.')
                shots.append({'shot_id': shot.get('shot_id'), 'start': start, 'end': end,
                              'trigger': trigger})
                cursor = end
            if not shots:
                raise ValueError(f'Render mới: {sid} chưa có nhóm hình đã chọn.')
            characters = list(scene.get('characters') or [])
            situations.append({'id': sid, 'text': text, 'character_ids': characters, 'shots': shots})
    if not situations:
        raise ValueError('Render mới: chưa có dữ liệu phân tích. Hãy tạo recap hoặc chọn Render cũ.')
    return {'version': 2, 'characters': list(registry.values()), 'situations': situations}


def prepare_plan_inputs(plan):
    texts, starts, ends, groups = [], [], [], []
    for situation in plan['situations']:
        text, shots = situation['text'], situation['shots']
        positions = [0]
        for shot in shots[1:]:
            matches = list(re.finditer(r'(?<!\w)' + re.escape(shot['trigger']) + r'(?!\w)', text, re.I))
            if len(matches) != 1 or matches[0].start() <= positions[-1]:
                raise ValueError(f"Mốc lời không đúng thứ tự: {situation['id']}")
            positions.append(matches[0].start())
        positions.append(len(text))
        groups.append([{'text': text[positions[i]:positions[i + 1]].strip(),
                        'start': shot['start'], 'end': shot['end']}
                       for i, shot in enumerate(shots)])
        texts.append(text)
        starts.append(shots[0]['start'])
        ends.append(shots[-1]['end'])
    return texts, starts, ends, groups


def strict_visual_edl(texts, durations, groups, word_cues, source_duration=None):
    from tts_engine import _word_locked_beat_boundaries
    if not (len(texts) == len(durations) == len(groups) == len(word_cues)):
        raise ValueError('Render mới: số đoạn lời, thời lượng và nhóm hình không khớp.')
    rows = []
    for i, (text, duration, group) in enumerate(zip(texts, durations, groups)):
        if not group or not math.isfinite(float(duration)) or float(duration) <= 0:
            raise ValueError(f'Render mới: dữ liệu thời lượng/hình không hợp lệ ở đoạn {i + 1}.')
        if len(group) > 1 and (i >= len(word_cues) or not word_cues[i]):
            raise ValueError(f'Render mới: tình huống {i + 1} thiếu mốc từ; không tự đoán điểm cắt.')
        boundaries = ([0.0, duration] if len(group) == 1 else
                      _word_locked_beat_boundaries(group, word_cues[i], duration))
        if boundaries is None or len(boundaries) != len(group) + 1:
            raise ValueError(f'Render mới: chưa căn được mốc từ của tình huống {i + 1}; không tự đoán điểm cắt.')
        for j, shot in enumerate(group):
            length = boundaries[j + 1] - boundaries[j]
            limit = shot['end']
            if source_duration is not None:
                following = group[j + 1:] + [s for later in groups[i + 1:] for s in later]
                next_start = next((s['start'] for s in following if s['start'] > shot['start']), source_duration)
                limit = min(float(source_duration), float(next_start))
            end = min(limit, max(shot['end'], shot['start'] + length))
            if length <= 0 or length > end - shot['start'] + 1 / 30:
                raise ValueError(f'Render mới: tình huống {i + 1}, cảnh {j + 1} thiếu hình cho '
                                 f'{length:.1f}s lời đọc trước shot kế tiếp hoặc cuối video.')
            rows.append({'text': shot['text'], 'duration': length,
                         'start': shot['start'], 'end': end})
    return rows


def _identity(path):
    if not path:
        return None
    stat = os.stat(path)
    return [os.path.abspath(path), stat.st_size, stat.st_mtime_ns]


def cache_key(plan, source, bgm, volume, voice, rate, options):
    data = [4, plan, _identity(source), _identity(bgm), volume, voice, rate,
            {k: v for k, v in options.items() if not k.startswith('_') and k != 'structured_plan'}]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def render_structured_video(task_id, source, bgm, volume, voice, rate, plan, options, progress, log=None):
    from tts_engine import generate_all_tts, get_audio_duration, get_ffmpeg_exe
    from video_renderer import create_master_audio, create_master_video_raw, get_video_info
    from render_encoding import EncoderSession
    from recap_budget import measured_narration_timing
    texts, starts, ends, groups = prepare_plan_inputs(plan)
    key = cache_key(plan, source, bgm, volume, voice, rate, options)
    work = Path(source).parent / '.render_cache' / key
    work.mkdir(parents=True, exist_ok=True)
    _save(work / 'render_plan.json', plan)
    if log:
        log('INFO', f'[RENDER_V2] {len(texts)} tình huống; cache: {work}')
    tts_file = work / 'tts_complete.json'
    paths = durations = cues = None
    if tts_file.exists():
        try:
            saved = json.loads(tts_file.read_text(encoding='utf-8'))
            if (len(saved['paths']) == len(texts)
                    and len(saved['durations']) == len(texts) and len(saved['cues']) == len(texts)
                    and all(Path(p).exists() and get_audio_duration(p) > 0 for p in saved['paths'])
                    and (work / 'submaker_subtitles.ass').exists()):
                paths, durations, cues = saved['paths'], saved['durations'], saved['cues']
        except (ValueError, OSError, KeyError):
            pass
    if paths is None:
        progress(5, 'Render mới: đang tạo giọng theo đoạn kể...')
        tts_options = dict(options, tts_policy='bounded_v2', _tts_max_items=1,
                           _tts_progress=lambda done, total: progress(
                               5 + int(35 * done / total), f'Giọng đọc: {done}/{total} đoạn đã xong'))
        paths, durations, cues = generate_all_tts(texts, voice, rate, str(work), tts_options, log,
                                                 return_word_cues=True)
        _save(tts_file, {'paths': paths, 'durations': durations, 'cues': cues})
    else:
        progress(40, f'Dùng lại giọng đọc: {len(texts)}/{len(texts)} đoạn')
    audio = work / 'master_audio.mp3'
    audio_marker = work / 'audio_complete.json'
    if not audio.exists() or not audio_marker.exists() or get_audio_duration(str(audio)) <= 0:
        progress(42, 'Đang phối giọng đọc và nhạc nền...')
        natural = create_master_audio(paths, bgm, volume, str(audio), str(work), texts)
        if natural and len(natural) == len(durations) and any(abs(a - b) > .04 for a, b in zip(natural, durations)):
            raise ValueError('Render mới: phối âm làm đổi thời gian giọng; giữ cache để kiểm tra, không xuất lệch hình.')
        _save(audio_marker, {'complete': True})
    edl = strict_visual_edl(texts, durations, groups, cues, source_duration=get_video_info(source)[0])
    _save(work / 'audio_visual_edl.json', edl)
    from narrative_contract import sentences as count_spoken_sentences
    options['_narration_timing'] = measured_narration_timing(sum(len(count_spoken_sentences(t)) for t in texts), durations, rate,
                                                          options.get('narration_budget'))
    raw = work / 'master_video_raw.mp4'
    encoder = EncoderSession(get_ffmpeg_exe(), log)
    render_options = dict(options, _encoder=encoder, _resume_segments=True)
    marker = work / 'video_complete.json'
    if not raw.exists() or not marker.exists():
        progress(50, 'Đang dựng nhóm cảnh theo thời gian giọng đọc...')
        create_master_video_raw(source, str(raw), [r['text'] for r in edl],
                                [r['duration'] for r in edl], render_options, update_fn=progress,
                                timestamps=[r['start'] for r in edl],
                                end_timestamps=[r['end'] for r in edl], diagnostic_log=log)
        _save(marker, {'complete': True})
    else:
        progress(80, 'Đã có hình hoàn chỉnh trong cache; tiếp tục xuất MP4')
    output = Path(source).parent / 'static' / 'output' / f'video_review_{task_id}.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    ass_copy = output.with_suffix('.ass')
    shutil.copy2(work / 'submaker_subtitles.ass', ass_copy)
    # Relative filter paths avoid Windows drive-letter escaping.
    ass_filter = os.path.relpath(ass_copy, Path(source).parent).replace('\\', '/')
    common = [get_ffmpeg_exe(), '-y', '-i', str(raw), '-i', str(audio),
              '-map', '0:v:0', '-map', '1:a:0']
    progress(85, 'Đang xuất MP4 và phụ đề...')
    if options.get('enable_subtitles', True):
        common += ['-vf', "ass='" + ass_filter.replace("'", "'\\''") + "'"]
        encoder.run(common,
                    ['-c:v', 'h264_nvenc', '-preset', 'p1', '-b:v', '8M',
                     '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', str(output)],
                    ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-threads', '4',
                     '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', str(output)],
                    'Xuất MP4 render mới', timeout=900, cwd=str(Path(source).parent))
    else:
        result = subprocess.run(common + ['-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                                         '-shortest', '-movflags', '+faststart', str(output)],
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError('Ghép MP4 thất bại: ' + result.stderr[-1500:])
    ass_copy.unlink(missing_ok=True)
    progress(100, 'Render mới hoàn tất; đã giữ cache để có thể chạy tiếp')
    return f'/static/output/{output.name}'
