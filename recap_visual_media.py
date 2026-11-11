"""Small, cached video evidence packs for image-first writing and verification."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import uuid

MEDIA_VERSION = 2


def parse_writing_response(text):
    """Read a complete writing document, preserving every row in an array."""
    source = str(text or '').lstrip('\ufeff').strip()
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', source, re.I)
    if fenced:
        source = fenced.group(1).strip()
    decoder = json.JSONDecoder(strict=False)
    if source.startswith(('{', '[')):
        # Do not rescue an inner scene from a truncated outer document.
        try:
            value, _ = decoder.raw_decode(source)
        except json.JSONDecodeError as exc:
            raise ValueError('VISUAL_WRITING_JSON_INVALID: ' + str(exc)) from exc
    else:
        candidates = []
        for match in re.finditer(r'[\[{]', source):
            try:
                value, end = decoder.raw_decode(source, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                candidates.append((end - match.start(), value))
        if not candidates:
            raise ValueError('VISUAL_WRITING_JSON_INVALID: khong co JSON hoan chinh')
        value = max(candidates, key=lambda row: row[0])[1]
    if isinstance(value, list):
        # Some responses wrap complete scene documents in an outer array.
        # Flatten every explicit container; never select an arbitrary inner row
        # or invent IDs from its position in the requested batch.
        containers = [isinstance(row, dict) and 'event_id' not in row
                      and ('scenes' in row or 'parts' in row) for row in value]
        if any(containers):
            if not all(containers):
                raise ValueError('VISUAL_WRITING_JSON_INVALID: mixed scene rows and containers')
            rows = []
            for container in value:
                normalized = parse_writing_response(json.dumps(container, ensure_ascii=False))
                if not isinstance(normalized.get('scenes'), list):
                    raise ValueError('VISUAL_WRITING_JSON_INVALID: scenes must be a list')
                rows.extend(normalized['scenes'])
            return {'scenes': rows}
        return {'scenes': value}
    if isinstance(value, dict):
        if 'scenes' not in value and isinstance(value.get('parts'), list):
            return parse_writing_response(json.dumps(value['parts'], ensure_ascii=False))
        return value
    raise ValueError('VISUAL_WRITING_JSON_INVALID: can object hoac danh sach cau')


def inspect_writing_batch(payload, expected_ids):
    """Describe strict batch validation failures without repairing model output."""
    rows = payload.get('scenes') if isinstance(payload, dict) else None
    if not isinstance(rows, list) and isinstance(payload, dict):
        parts = payload.get('parts')
        if isinstance(parts, list) and parts and isinstance(parts[0], dict):
            rows = parts[0].get('scenes')
    report = {'expected_ids': list(expected_ids), 'expected_count': len(expected_ids),
              'payload_type': type(payload).__name__, 'rows_type': type(rows).__name__,
              'actual_count': len(rows) if isinstance(rows, list) else None,
              'actual_ids': [], 'missing_ids': [], 'unexpected_ids': [],
              'duplicate_ids': [], 'invalid_rows': [], 'errors': []}
    if not isinstance(rows, list):
        report['errors'].append('scenes_missing_or_not_list')
    else:
        if len(rows) != len(expected_ids):
            report['errors'].append('scene_count_mismatch')
        seen = set()
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                report['invalid_rows'].append({'row': index, 'type': type(row).__name__})
                continue
            eid = row.get('event_id')
            report['actual_ids'].append(eid)
            if not isinstance(eid, str):
                report['invalid_rows'].append({'row': index, 'event_id': eid,
                                              'scene_id': row.get('scene_id'),
                                              'reason': 'event_id_missing_or_not_string'})
                continue
            if eid in seen:
                report['duplicate_ids'].append(eid)
            seen.add(eid)
            if eid not in expected_ids:
                report['unexpected_ids'].append(eid)
        report['missing_ids'] = [eid for eid in expected_ids if eid not in seen]
        if report['invalid_rows']:
            report['errors'].append('invalid_scene_rows')
        if report['missing_ids'] or report['unexpected_ids'] or report['duplicate_ids']:
            report['errors'].append('event_ids_mismatch')
    report['valid'] = not report['errors']
    return rows, report


def seconds(value):
    if isinstance(value, (float, int)):
        return float(value)
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(str(value).split(':'))))


def evidence_manifest(cards, source_offset=0.0):
    """Use backend ranges only, with explicit source and pack coordinates."""
    manifest, cursor = [], 0.0
    for card in cards:
        sid = str(card.get('scene_id') or card.get('event_id') or '')
        if not re.fullmatch(r'[A-Za-z0-9_]+', sid):
            raise ValueError('VISUAL_PACK_SCENE_ID_INVALID')
        sources = ((card.get('visual_sources') or []) if card.get('evidence_scope') == 'situation'
                   else card.get('locked_visual_shots') or card.get('visual_sources') or [])
        if not sources:
            raise ValueError('VISUAL_PACK_SHOTS_MISSING:' + sid)
        for index, source in enumerate(sources, 1):
            a = seconds(source.get('start_seconds', source.get('start'))) + source_offset
            b = seconds(source.get('end_seconds', source.get('end'))) + source_offset
            if not all(math.isfinite(v) for v in (a, b)) or a < 0 or not .05 < b - a <= 15.05:
                raise ValueError('VISUAL_PACK_RANGE_INVALID:' + sid)
            shot_ids = source.get('shot_ids') or ([source['shot_id']] if source.get('shot_id') else [])
            if not shot_ids or any(not re.fullmatch(r'SHOT_[0-9]+', str(s)) for s in shot_ids):
                raise ValueError('VISUAL_PACK_SHOT_ID_INVALID:' + sid)
            visual_id = source.get('visual_id') or f'V{index:03d}'
            if not re.fullmatch(r'V[0-9]+', str(visual_id)):
                raise ValueError('VISUAL_PACK_VISUAL_ID_INVALID:' + sid)
            manifest.append({'scene_id': sid, 'visual_id': visual_id,
                             'shot_ids': shot_ids, 'source_start': a, 'source_end': b,
                             'pack_start': cursor, 'pack_end': cursor + b - a})
            cursor += b - a
    return manifest


def visual_batches(cards, max_scenes=6, max_seconds=90.0, max_shots=18, max_chars=None):
    """Keep normal requests small; never silently omit a sentence's evidence."""
    batch, duration, shots, chars = [], 0.0, 0, 0
    for card in cards:
        rows = evidence_manifest([card])
        length = sum(r['source_end'] - r['source_start'] for r in rows)
        size = len(json.dumps(card, ensure_ascii=False))
        if batch and (len(batch) >= max_scenes or duration + length > max_seconds
                      or shots + len(rows) > max_shots
                      or (max_chars is not None and chars + size > max_chars)):
            yield batch
            batch, duration, shots, chars = [], 0.0, 0, 0
        batch.append(card)
        duration += length
        shots += len(rows)
        chars += size
    if batch:
        yield batch


def evidence_writing_batches(cards, max_scenes=18, max_chars=24000):
    """Batch locked evidence for prose without preparing or uploading media."""
    from semantic_visuals import visual_catalog
    batch, size = [], 0
    for card in cards:
        # Raw plan entries carry coverage registries, neighboring events and
        # other repeated metadata that is never sent in the grounded prompt.
        # Measure the actual writing card instead of that backend state.
        writing_card = {'event_id': card.get('event_id'), 'focus': card.get('narration_focus'),
                        'observed_facts': card.get('observed_facts') or [],
                        'allowed_character_names': card.get('allowed_character_names') or [],
                        'visuals': visual_catalog(card.get('visual_sources') or [])}
        length = len(json.dumps(writing_card, ensure_ascii=False))
        if batch and (len(batch) >= max_scenes or size + length > max_chars):
            yield batch
            batch, size = [], 0
        batch.append(card)
        size += length
    if batch:
        yield batch


def build_visual_pack(source, cache_dir, cards, ffmpeg='ffmpeg', source_offset=0.0):
    """Read actual source frames at 2 FPS; overlay stable scene/shot labels."""
    manifest = evidence_manifest(cards, source_offset)
    if not manifest:
        raise ValueError('VISUAL_PACK_EMPTY')
    stat = os.stat(source)
    identity = [MEDIA_VERSION, os.path.realpath(source), stat.st_size, stat.st_mtime_ns]
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([identity, manifest], sort_keys=True).encode()).hexdigest()
    output = root / (key + '.mp4')
    if output.is_file() and output.stat().st_size:
        return str(output), manifest

    def run(args):
        result = subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y'] + args,
                                capture_output=True, text=True, errors='replace', timeout=120)
        if result.returncode:
            raise RuntimeError('VISUAL_PACK_FAILED: ' + (result.stderr or '')[-1000:])

    clips = []
    font = "fontfile='C\\:/Windows/Fonts/arialbd.ttf':" if os.name == 'nt' else ''
    for row in manifest:
        clip_key = hashlib.sha256(json.dumps([identity, row['scene_id'], row['visual_id'],
            row['shot_ids'], row['source_start'], row['source_end']], sort_keys=True).encode()).hexdigest()
        clip = root / (clip_key + '.clip.mp4')
        if not clip.is_file() or not clip.stat().st_size:
            label = row['scene_id'] + ' ' + row['visual_id'] + ' ' + ','.join(row['shot_ids'])
            # Identifiers are validated above. No model text enters a filter.
            label = label.replace(',', ' ')
            filters = ('scale=640:360:force_original_aspect_ratio=decrease,'
                       'pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=2,setpts=PTS-STARTPTS,'
                       f"drawtext={font}text='{label}':x=8:y=8:fontsize=20:fontcolor=white:"
                       'borderw=3:bordercolor=black')
            temporary = root / (clip_key + '.' + uuid.uuid4().hex + '.tmp.mp4')
            try:
                run(['-ss', f"{row['source_start']:.6f}", '-i', str(source),
                     '-t', f"{row['source_end'] - row['source_start']:.6f}", '-an', '-vf', filters,
                     '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '29', '-pix_fmt', 'yuv420p', str(temporary)])
                os.replace(temporary, clip)
            finally:
                temporary.unlink(missing_ok=True)
        clips.append(clip)
    listing = root / (key + '.' + uuid.uuid4().hex + '.concat.txt')
    listing.write_text(''.join("file '" + p.resolve().as_posix().replace("'", "'\\''") + "'\n"
                               for p in clips), encoding='utf-8')
    temporary = root / (key + '.' + uuid.uuid4().hex + '.tmp.mp4')
    try:
        run(['-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy',
             '-movflags', '+faststart', str(temporary)])
        os.replace(temporary, output)
    finally:
        listing.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return str(output), manifest
