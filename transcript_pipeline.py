"""Speech transcript collection for the recap pipeline.

Groq Whisper is the primary provider.  The module deliberately returns a
small provider-neutral structure so the story pipeline never depends on a
vendor response schema.  A completed transcript is cached by source file
fingerprint and can be reused when Gemini or recap writing is retried.
"""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests


GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
TRANSCRIPT_CHUNK_SECONDS = 20 * 60
TRANSCRIPT_OVERLAP_SECONDS = 2.0

PROMOTIONAL_SUBTITLE_PATTERNS = (
    r"\bsubscribe\b",
    r"\bđăng\s*k[ýy]\s+kênh\b",
    r"\bủng\s*hộ\s+kênh\b",
    r"\blike\s*(?:và|and|&)\s*share\b",
    r"\bkhông\s*bỏ\s*lỡ\s+những\s+video\b",
)


class TranscriptQuotaError(RuntimeError):
    pass


def _repair_common_mojibake(value):
    """Repair the common UTF-8-as-Windows-1252 subtitle corruption."""
    text = str(value or "")
    if not any(marker in text for marker in ("Ã", "Â", "Æ", "áº", "á»")):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    old_noise = sum(text.count(marker) for marker in ("Ã", "Â", "Æ", "áº", "á»"))
    new_noise = sum(repaired.count(marker) for marker in ("Ã", "Â", "Æ", "áº", "á»"))
    return repaired if new_noise < old_noise else text


def _subtitle_time_seconds(value):
    raw = str(value or "").strip().replace(",", ".")
    pieces = raw.split(":")
    if len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        raise ValueError("timestamp phu de khong hop le")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _is_promotional_subtitle(text):
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in PROMOTIONAL_SUBTITLE_PATTERNS)


def parse_subtitle_text(subtitle_text, duration_seconds=None):
    """Parse SRT/VTT text into trusted dialogue rows and discard obvious ads.

    SRT is treated as evidence for spoken words and names, never as proof of a
    visible action or of who said a line.
    """
    source = _repair_common_mojibake(subtitle_text).replace("\r\n", "\n").replace("\r", "\n")
    source = source.lstrip("\ufeff")
    if source.startswith("WEBVTT"):
        source = source.split("\n", 1)[1] if "\n" in source else ""
    blocks = re.split(r"\n\s*\n", source)
    segments = []
    removed_promotional = 0
    timestamp_re = re.compile(
        r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
        r"(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{1,3})"
    )
    duration = float(duration_seconds) if duration_seconds is not None else None
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timestamp_index = next(
            (index for index, line in enumerate(lines) if timestamp_re.search(line)),
            None,
        )
        if timestamp_index is None:
            continue
        match = timestamp_re.search(lines[timestamp_index])
        try:
            start = _subtitle_time_seconds(match.group("start"))
            end = _subtitle_time_seconds(match.group("end"))
        except (TypeError, ValueError):
            continue
        if end <= start or start < 0:
            continue
        if duration is not None:
            if start > duration + 2.0:
                continue
            end = min(end, duration)
        text = " ".join(lines[timestamp_index + 1:])
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\{\\[^}]+\}", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" -\t")
        if not text:
            continue
        if _is_promotional_subtitle(text):
            removed_promotional += 1
            continue
        segments.append({
            "start_seconds": start,
            "end_seconds": end,
            "text": text,
            "confidence": 0.98,
            "source": "subtitle",
        })
    segments = _deduplicate_overlap(segments)
    return {
        "provider": "subtitle",
        "model": "user-srt",
        "segments": segments,
        "segment_count": len(segments),
        "removed_promotional": removed_promotional,
        "silent_or_no_speech": not bool(segments),
    }


def merge_preferred_subtitles(subtitle_segments, whisper_segments):
    """Prefer SRT on overlapping dialogue and use Whisper only for gaps."""
    preferred = [dict(row) for row in (subtitle_segments or []) if isinstance(row, dict)]
    fallback = [dict(row) for row in (whisper_segments or []) if isinstance(row, dict)]
    if not preferred:
        return _deduplicate_overlap(fallback)
    merged = list(preferred)
    for row in fallback:
        try:
            start = float(row.get("start_seconds", 0.0))
            end = float(row.get("end_seconds", start))
        except (TypeError, ValueError):
            continue
        overlaps_subtitle = any(
            float(subtitle.get("end_seconds", 0.0)) >= start - 0.35
            and float(subtitle.get("start_seconds", 0.0)) <= end + 0.35
            for subtitle in preferred
        )
        if overlaps_subtitle:
            continue
        row.setdefault("source", "whisper")
        merged.append(row)
    return _deduplicate_overlap(merged)


def _transcript_confidence(row, default=0.75):
    """Map provider-specific Whisper scores to a conservative 0..1 value."""
    if not isinstance(row, dict):
        return float(default)
    try:
        if row.get("confidence") is not None:
            return min(1.0, max(0.0, float(row.get("confidence"))))
        average_logprob = float(row.get("avg_logprob"))
        no_speech = min(1.0, max(0.0, float(row.get("no_speech_prob", 0.0))))
        return min(1.0, max(0.0, math.exp(average_logprob) * (1.0 - no_speech)))
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _source_fingerprint(video_path):
    stat = os.stat(video_path)
    identity = f"{os.path.abspath(video_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()


def _cache_path(cache_dir, video_path):
    return os.path.join(cache_dir, f"{_source_fingerprint(video_path)}.json")


def _load_cache(cache_dir, video_path):
    path = _cache_path(cache_dir, video_path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("source_fingerprint") != _source_fingerprint(video_path):
            return None
        if not isinstance(payload.get("segments"), list):
            return None
        return payload
    except (OSError, ValueError, TypeError):
        return None


def _save_cache(cache_dir, video_path, payload):
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, video_path)
    temporary = f"{path}.tmp"
    body = dict(payload)
    body["source_fingerprint"] = _source_fingerprint(video_path)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def _extract_audio_chunks(video_path, ffmpeg_exe, output_dir, duration_seconds):
    chunks = []
    cursor = 0.0
    index = 0
    duration_seconds = max(0.0, float(duration_seconds))
    while cursor < duration_seconds:
        end = min(duration_seconds, cursor + TRANSCRIPT_CHUNK_SECONDS)
        start = max(0.0, cursor - (TRANSCRIPT_OVERLAP_SECONDS if index else 0.0))
        path = os.path.join(output_dir, f"audio_{index:03d}.mp3")
        command = [
            ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
            "-b:a", "32k", path,
        ]
        result = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=max(180, int((end - start) * 0.75)),
        )
        if result.returncode != 0 or not os.path.isfile(path):
            details = result.stderr.decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"Khong tach duoc am thanh Whisper: {details}")
        chunks.append({"path": path, "start": start, "end": end})
        cursor = end
        index += 1
    return chunks


def _groq_transcribe_chunk(api_key, chunk, model, timeout_seconds=300):
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    last_error = None
    for attempt in range(3):
        try:
            with open(chunk["path"], "rb") as audio:
                response = requests.post(
                    GROQ_TRANSCRIPTION_URL,
                    headers=headers,
                    data=data,
                    files={"file": (os.path.basename(chunk["path"]), audio, "audio/mpeg")},
                    timeout=timeout_seconds,
                )
            if response.status_code in (401, 403):
                raise RuntimeError("Groq API key khong hop le hoac khong co quyen Whisper.")
            if response.status_code == 429:
                raise TranscriptQuotaError("Groq Whisper da het quota hoac vuot toc do.")
            response.raise_for_status()
            return response.json()
        except TranscriptQuotaError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Groq Whisper that bai sau 3 lan: {last_error}")


def _normalize_segments(payload, offset_seconds, chunk_end):
    raw_segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(raw_segments, list):
        text = str((payload or {}).get("text", "")).strip() if isinstance(payload, dict) else ""
        return ([{
            "start_seconds": float(offset_seconds),
            "end_seconds": float(chunk_end),
            "text": text,
        }] if text else [])
    result = []
    for row in raw_segments:
        if not isinstance(row, dict):
            continue
        text = " ".join(str(row.get("text", "")).split()).strip()
        if not text:
            continue
        try:
            start = float(offset_seconds) + float(row.get("start", 0.0))
            end = float(offset_seconds) + float(row.get("end", row.get("start", 0.0)))
        except (TypeError, ValueError):
            continue
        result.append({
            "start_seconds": max(0.0, start),
            "end_seconds": max(start, min(float(chunk_end), end)),
            "text": text,
            "confidence": round(_transcript_confidence(row), 4),
        })
    return result


def _deduplicate_overlap(segments):
    ordered = sorted(segments, key=lambda row: (row["start_seconds"], row["end_seconds"]))
    result = []
    for row in ordered:
        if _is_promotional_subtitle(row.get("text", "")):
            continue
        normalized = " ".join(row["text"].casefold().split())
        duplicate = False
        for previous in result[-4:]:
            previous_text = " ".join(previous["text"].casefold().split())
            overlap = min(row["end_seconds"], previous["end_seconds"]) - max(
                row["start_seconds"], previous["start_seconds"]
            )
            if overlap >= -0.75 and (
                normalized == previous_text
                or (len(normalized) >= 12 and normalized in previous_text)
                or (len(previous_text) >= 12 and previous_text in normalized)
            ):
                duplicate = True
                if len(row["text"]) > len(previous["text"]):
                    previous.update(row)
                break
        if not duplicate:
            result.append(dict(row))
    return result


def _local_transcribe(chunks, model_name="large-v3"):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Groq khong dung duoc va faster-whisper chua duoc cai de fallback local."
        ) from exc
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
    except Exception:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    result = []
    for chunk in chunks:
        segments, _ = model.transcribe(
            chunk["path"], beam_size=5, vad_filter=True, condition_on_previous_text=False
        )
        for segment in segments:
            text = " ".join(str(segment.text).split()).strip()
            if text:
                result.append({
                    "start_seconds": chunk["start"] + float(segment.start),
                    "end_seconds": min(chunk["end"], chunk["start"] + float(segment.end)),
                    "text": text,
                    "confidence": round(_transcript_confidence({
                        "avg_logprob": getattr(segment, "avg_logprob", None),
                        "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                    }), 4),
                })
    return result


def transcribe_video(
    video_path, duration_seconds, ffmpeg_exe, cache_dir, groq_api_key="",
    model=GROQ_TRANSCRIPTION_MODEL, progress=None,
):
    """Return timestamped transcript, using cache then Groq then local Whisper."""
    cached = _load_cache(cache_dir, video_path)
    if cached is not None:
        if progress:
            progress("Da tai transcript Whisper tu cache.")
        return cached

    temp_dir = tempfile.mkdtemp(prefix="whisper_audio_")
    try:
        if progress:
            progress("Whisper dang tach am thanh thanh cac doan nho...")
        chunks = _extract_audio_chunks(
            video_path, ffmpeg_exe, temp_dir, duration_seconds
        )
        segments = []
        provider = "groq"
        if str(groq_api_key or "").strip():
            try:
                for index, chunk in enumerate(chunks, start=1):
                    if progress:
                        progress(f"Groq Whisper dang nghe doan {index}/{len(chunks)}...")
                    payload = _groq_transcribe_chunk(
                        str(groq_api_key).strip(), chunk, model
                    )
                    segments.extend(_normalize_segments(
                        payload, chunk["start"], chunk["end"]
                    ))
            except (TranscriptQuotaError, RuntimeError):
                if progress:
                    progress("Groq khong dung duoc; dang chuyen sang Whisper local...")
                provider = "local"
                segments = _local_transcribe(chunks)
        else:
            provider = "local"
            segments = _local_transcribe(chunks)

        segments = _deduplicate_overlap(segments)
        payload = {
            "provider": provider,
            "model": model if provider == "groq" else "faster-whisper-large-v3",
            "segments": segments,
            "segment_count": len(segments),
            "silent_or_no_speech": not bool(segments),
        }
        _save_cache(cache_dir, video_path, payload)
        return payload
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def format_transcript_window(
    transcript_segments, start_seconds=0.0, end_seconds=None,
    max_characters=160000,
):
    """Return compact, timestamped Whisper text for one Gemini video window.

    Timestamps remain global so the model can reconcile dialogue with the
    whole-film outline.  If an unusually dense transcript exceeds the cap,
    keep evenly distributed rows including both ends instead of truncating the
    ending of the film.
    """
    start = max(0.0, float(start_seconds or 0.0))
    end = float("inf") if end_seconds is None else max(start, float(end_seconds))
    rows = []
    for raw in transcript_segments or []:
        if not isinstance(raw, dict):
            continue
        try:
            row_start = float(raw.get("start_seconds", 0.0))
            row_end = float(raw.get("end_seconds", row_start))
        except (TypeError, ValueError):
            continue
        if row_end < start or row_start > end:
            continue
        text = " ".join(str(raw.get("text", "")).split()).strip()
        if not text:
            continue
        rows.append((row_start, row_end, text, _transcript_confidence(raw)))

    def render(row):
        row_start, row_end, text, confidence = row
        warning = " [DO_TIN_CAY_THAP]" if confidence < 0.55 else ""
        return f"[{row_start:.1f}-{row_end:.1f}]{warning} {text}"

    rendered = [render(row) for row in rows]
    limit = max(1000, int(max_characters or 0))
    if sum(len(line) + 1 for line in rendered) <= limit:
        return "\n".join(rendered)
    if not rows:
        return ""
    average_length = max(24, sum(len(line) + 1 for line in rendered) // len(rendered))
    keep_count = max(2, min(len(rows), limit // average_length))
    indexes = {
        round(index * (len(rows) - 1) / max(1, keep_count - 1))
        for index in range(keep_count)
    }
    return "\n".join(render(rows[index]) for index in sorted(indexes))


def filter_transcript_evidence(segments):
    """Discard promotional/repeated sparse ASR noise without extra requests."""
    from collections import Counter
    counts = Counter(str(s.get('text') or '').strip().casefold() for s in segments or [])
    kept, removed = [], []
    for row in segments or []:
        text = str(row.get('text') or '').strip()
        duration = max(0, float(row.get('end_seconds', 0)) - float(row.get('start_seconds', 0)))
        letters = len(re.findall(r'\w', text))
        sparse = duration >= 15 and letters / max(duration, 1) < .5
        repeated = counts[text.casefold()] >= 3 and duration >= 12 and letters < 60
        if not text or _is_promotional_subtitle(text) or sparse or repeated or _transcript_confidence(row) < .35:
            removed.append(row)
        else:
            kept.append(row)
    return kept, {'removed_noise_segments': len(removed), 'kept_segments': len(kept)}


def attach_transcript_to_evidence(chunk_evidence, transcript_segments, padding=1.5):
    """Attach only time-overlapping dialogue to grounded visual events."""
    segments = [
        row for row in (transcript_segments or [])
        if isinstance(row, dict) and _transcript_confidence(row) >= 0.35
    ]
    attached = 0
    for chunk in chunk_evidence or []:
        if not isinstance(chunk, dict):
            continue
        for event in chunk.get("events") or []:
            if not isinstance(event, dict):
                continue
            try:
                start = float(event.get("start_seconds"))
                end = float(event.get("end_seconds"))
            except (TypeError, ValueError):
                continue
            matches = []
            for segment in segments:
                try:
                    seg_start = float(segment.get("start_seconds"))
                    seg_end = float(segment.get("end_seconds"))
                except (TypeError, ValueError):
                    continue
                if seg_end >= start - padding and seg_start <= end + padding:
                    matches.append({
                        "start_seconds": seg_start,
                        "end_seconds": seg_end,
                        "text": str(segment.get("text", "")).strip(),
                        "confidence": _transcript_confidence(segment),
                    })
            if matches:
                event["transcript_evidence"] = matches
                event["spoken_text"] = " ".join(
                    row["text"] for row in matches if row["text"]
                ).strip()
                attached += len(matches)
            else:
                event["transcript_evidence"] = []
                event["spoken_text"] = ""
            sources = {s.get('shot_id'): s for s in event.get('visual_sources') or []}
            for fact in event.get('grounded_facts') or []:
                windows = [sources[sid] for sid in fact.get('shot_ids') or [] if sid in sources]
                fact['transcript_evidence'] = [dict(line) for line in matches if any(
                    line['end_seconds'] > float(window['start_seconds'])
                    and line['start_seconds'] < float(window['end_seconds'])
                    for window in windows)]
                fact['spoken_text'] = ' '.join(line['text'] for line in fact['transcript_evidence'])
    return chunk_evidence, attached


def promote_uncovered_transcript_events(
    chunk_evidence, transcript_segments, video_duration_seconds,
    maximum_uncovered_seconds=45.0,
):
    """Turn useful dialogue inside long visual gaps into bridge evidence.

    Whisper used to be attached only to events already found by Gemini.  That
    made it impossible for dialogue to repair a scene Gemini had overlooked.
    This function adds a conservative transcript-backed card only when a long
    part of the timeline has no visual event.  The card never guesses a
    speaker or a visible action and remains labelled ``transcript_bridge`` so
    later stages can treat it as weaker than visual evidence.
    """
    chunks = copy.deepcopy(list(chunk_evidence or []))
    segments = []
    for raw in transcript_segments or []:
        if not isinstance(raw, dict):
            continue
        try:
            start = max(0.0, float(raw.get("start_seconds", 0.0)))
            end = max(start, float(raw.get("end_seconds", start)))
        except (TypeError, ValueError):
            continue
        text = " ".join(str(raw.get("text", "")).split()).strip()
        # Very short interjections and recognition noise are poor plot facts.
        if len(text.split()) < 4 or _transcript_confidence(raw) < 0.50:
            continue
        segments.append({"start_seconds": start, "end_seconds": end, "text": text})
    if not chunks or not segments:
        return chunks, 0

    duration = max(1.0, float(video_duration_seconds or 0.0))
    maximum_gap = max(20.0, float(maximum_uncovered_seconds))
    visual_events = sorted(
        (
            event for chunk in chunks for event in (chunk.get("events") or [])
            if isinstance(event, dict)
            and str(event.get("source", "")).lower() != "transcript_bridge"
        ),
        key=lambda row: (
            float(row.get("start_seconds", 0.0)),
            float(row.get("end_seconds", 0.0)),
        ),
    )
    occupied = [
        (
            max(0.0, float(event.get("start_seconds", 0.0))),
            min(duration, float(event.get("end_seconds", 0.0))),
        )
        for event in visual_events
    ]

    gaps = []
    cursor = 0.0
    for start, end in occupied:
        if start - cursor > maximum_gap:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor > maximum_gap:
        gaps.append((cursor, duration))

    additions = []
    seen_text = set()
    for gap_start, gap_end in gaps:
        # Place one bridge in every maximum-gap sized region, but only when
        # Whisper actually heard meaningful speech there.
        window_start = gap_start
        while window_start < gap_end:
            window_end = min(gap_end, window_start + maximum_gap)
            candidates = [
                row for row in segments
                if row["end_seconds"] >= window_start
                and row["start_seconds"] <= window_end
            ]
            if candidates:
                midpoint = (window_start + window_end) / 2.0
                candidate = max(candidates, key=lambda row: (
                    min(len(row["text"].split()), 30)
                    - abs((row["start_seconds"] + row["end_seconds"]) / 2.0 - midpoint) / 8.0
                ))
                normalized = " ".join(candidate["text"].casefold().split())
                if normalized not in seen_text:
                    seen_text.add(normalized)
                    additions.append(dict(candidate))
            window_start = window_end

    added = 0
    for row in additions:
        midpoint = (row["start_seconds"] + row["end_seconds"]) / 2.0
        owner = next((
            chunk for chunk in chunks
            if float(chunk.get("chunk_start_seconds", chunk.get("start_seconds", 0.0)))
            <= midpoint <
            float(chunk.get("chunk_end_seconds", chunk.get("end_seconds", duration)))
        ), None)
        if owner is None:
            owner = min(
                chunks,
                key=lambda chunk: abs(
                    float(chunk.get("chunk_start_seconds", 0.0)) - midpoint
                ),
            )
        added += 1
        owner.setdefault("events", []).append({
            "evidence_id": f"TR{added:05d}",
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "event": "Lời thoại cung cấp thông tin cốt truyện trong khoảng hình ảnh bị bỏ sót.",
            "visual_detail": "",
            "characters": [],
            "importance": "bridge",
            "story_role": "development",
            "narrative_function": "development",
            "event_kind": "explanation",
            "temporal_relation": "continuation",
            "source": "transcript_bridge",
            "source_confidence": 0.62,
            "transcript_evidence": [dict(row)],
            "spoken_text": row["text"],
            "narration_focus": row["text"],
        })
        owner["events"].sort(key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ))
    return chunks, added


def promote_transcript_events_to_target(
    chunk_evidence, transcript_segments, video_duration_seconds, target_count,
):
    """Add evenly distributed, timestamp-grounded dialogue facts to capacity.

    This never invents a visible action or speaker. It only turns real Whisper
    text into compact evidence cards until the requested recap sentence pool is
    available. Existing visual evidence always remains the stronger source.
    """
    chunks = copy.deepcopy(list(chunk_evidence or []))
    target = max(1, int(target_count or 1))
    existing = [
        event for chunk in chunks for event in (chunk.get("events") or [])
        if isinstance(event, dict)
    ]
    deficit = max(0, target - len(existing))
    if not chunks or deficit <= 0:
        return chunks, 0

    duration = max(1.0, float(video_duration_seconds or 0.0))
    rows = []
    for raw in transcript_segments or []:
        if not isinstance(raw, dict):
            continue
        try:
            start = max(0.0, float(raw.get("start_seconds", 0.0)))
            end = min(duration, max(start, float(raw.get("end_seconds", start))))
        except (TypeError, ValueError):
            continue
        text = " ".join(str(raw.get("text", "")).split()).strip()
        if end <= start or len(text.split()) < 4 or _transcript_confidence(raw) < 0.50:
            continue
        rows.append({"start_seconds": start, "end_seconds": end, "text": text})
    if not rows:
        return chunks, 0
    rows.sort(key=lambda row: (row["start_seconds"], row["end_seconds"]))

    # One candidate per time cell keeps coverage even instead of clustering all
    # transcript anchors around dialogue-heavy scenes.
    cell_seconds = max(18.0, duration / max(target, 1))
    grouped = {}
    for row in rows:
        cell = min(target - 1, int(row["start_seconds"] / cell_seconds))
        grouped.setdefault(cell, []).append(row)

    candidates = []
    seen_text = set()
    for cell in sorted(grouped):
        group = grouped[cell]
        words = []
        for row in group:
            for word in row["text"].split():
                if len(words) >= 48:
                    break
                words.append(word)
            if len(words) >= 48:
                break
        text = " ".join(words).strip()
        normalized = " ".join(text.casefold().split())
        if len(text.split()) < 4 or normalized in seen_text:
            continue
        seen_text.add(normalized)
        candidates.append({
            "start_seconds": group[0]["start_seconds"],
            "end_seconds": group[-1]["end_seconds"],
            "text": text,
            "cell": cell,
        })
    if len(candidates) > deficit:
        if deficit == 1:
            candidates = [candidates[len(candidates) // 2]]
        else:
            last = len(candidates) - 1
            indexes = sorted({round(index * last / (deficit - 1)) for index in range(deficit)})
            candidates = [candidates[index] for index in indexes]

    next_id = 1 + sum(
        str(event.get("evidence_id", "")).upper().startswith("TR")
        for event in existing
    )
    added = 0
    for row in candidates[:deficit]:
        midpoint = (row["start_seconds"] + row["end_seconds"]) / 2.0
        owner = next((
            chunk for chunk in chunks
            if float(chunk.get("start_seconds", 0.0)) <= midpoint
            < float(chunk.get("end_seconds", duration))
        ), chunks[-1])
        evidence_id = f"TR{next_id:05d}"
        next_id += 1
        owner.setdefault("events", []).append({
            "evidence_id": evidence_id,
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "event": row["text"],
            "characters": [],
            "importance": "bridge",
            "importance_score": 1,
            "story_role": "development",
            "narrative_function": "development",
            "event_kind": "explanation",
            "temporal_relation": "continuation",
            "event_cluster": f"TRANSCRIPT_{row['cell']:04d}",
            "source": "transcript_anchor",
            "source_confidence": 0.72,
            "transcript_evidence": [{
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "text": row["text"],
            }],
            "spoken_text": row["text"],
            "narration_focus": row["text"],
        })
        added += 1
    for chunk in chunks:
        chunk.setdefault("events", []).sort(key=lambda event: (
            float(event.get("start_seconds", 0.0)),
            float(event.get("end_seconds", 0.0)),
        ))
    return chunks, added
