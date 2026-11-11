"""Persistent identity metadata and safe rendering; raw observations stay intact."""
import copy
import math
import re
import unicodedata

VERSION = 3


def needs_identity_rebuild(checkpoint):
    chunks = checkpoint.get('evidence') or [] if isinstance(checkpoint, dict) else []
    return bool(chunks and (checkpoint.get('character_registry_version') != VERSION
        or any(not isinstance(c, dict) or c.get('character_registry_version') != VERSION for c in chunks)))


def confidence(value):
    try:
        value = float(value)
        return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0


def clean_list(value):
    return list(dict.fromkeys(str(v).strip() for v in value if isinstance(v, str) and v.strip())) if isinstance(value, list) else []


def fold(value):
    text = unicodedata.normalize('NFKD', str(value)).casefold().replace('đ', 'd')
    return ' '.join(re.findall(r'\w+', ''.join(c for c in text if not unicodedata.combining(c))))


def canonical_id(value, known=()):
    text = str(value or '').strip().upper()
    if not re.fullmatch(r'P\d+', text):
        return ''
    number = int(text[1:])
    for key in known:
        if re.fullmatch(r'P\d+', str(key)) and int(str(key)[1:]) == number:
            return str(key)
    return f'P{number:02d}'


def dialogue_name_mentions(text, names):
    """Resolve a unique one-letter ASR typo to an existing name, mentions only.

    This never creates identities, proves a speaker, or rewrites the transcript.
    Short words and ambiguous matches are deliberately excluded.
    """
    words = fold(text).split()
    def close(a, b):
        if a == b:
            return True
        if abs(len(a) - len(b)) > 1:
            return False
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b)) == 1
        if len(a) < len(b):
            a, b = b, a
        return any(a[:i] + a[i + 1:] == b for i in range(len(a)))
    normalized = {str(name): fold(name).split() for name in names if name}
    found = set()
    for width in {len(parts) for parts in normalized.values() if parts}:
        for i in range(len(words) - width + 1):
            phrase = ' '.join(words[i:i + width])
            if width == 1 and i and (
                    phrase == 'gian' and words[i - 1] in {'thoi', 'khong', 'trung', 'the', 'nhan'}
                    or phrase == 'nam' and words[i - 1] == 'viet'):
                continue
            exact = [n for n, parts in normalized.items() if phrase == ' '.join(parts)]
            if exact:
                found.update(exact)
                continue
            matches = [n for n, parts in normalized.items() if len(parts) == width
                and len(''.join(parts)) >= 5 and len(phrase.replace(' ', '')) >= 5
                and close(phrase, ' '.join(parts))]
            if len(matches) == 1:
                found.add(matches[0])
    return found


def render_ids(text, labels, allow_bare=True):
    labels = {str(k).upper(): str(v) for k, v in labels.items()}
    text = re.sub(r'\[CHAR=([A-Za-z0-9_]+)\]',
        lambda m: labels.get(m[1].upper(), 'nhân vật chưa xác định'), str(text), flags=re.I)
    if not allow_bare:
        return text
    return re.sub(r'(?<!\w)([CP]\d+)(?!\w)',
        lambda m: labels.get(m[1].upper(), 'nhân vật chưa xác định'), text, flags=re.I)


def safe_name_mentions(text, replacements):
    """Compatibility for old prose: exact proper-name spans, never IGNORECASE.

    New observations use [CHAR=id]. Ordinary words and descriptive phrases are
    not aliases to be globally replaced; substitutions are single-pass.
    """
    pairs = {str(k): str(v) for k, v in replacements.items()
             if k and v and str(k) != str(v) and any(c.isupper() for c in str(k))}
    if not pairs:
        return str(text)
    pattern = r'(?<!\w)(' + '|'.join(re.escape(k) for k in sorted(pairs, key=len, reverse=True)) + r')(?!\w)'
    def replace(match):
        old = match[0]
        prefix = str(text)[:match.start()]
        if old.casefold() == 'gian' and re.search(r'(?:thời|không|trung|thế|nhân)\s+$', prefix, re.I):
            return old
        if old.casefold() == 'nam' and re.search(r'việt\s+$', prefix, re.I):
            return old
        return pairs[old]
    return re.sub(pattern, replace, str(text))


def render_identity_context(text, characters):
    """Render a disposable story-context view; never modify original summaries."""
    labels, candidates = {}, {}
    for c in characters or []:
        label = c.get('display_name') or c.get('name') or c.get('neutral_alias')
        if not label:
            continue
        labels[str(c.get('id') or '').upper()] = label
        if c.get('name_status') == 'candidate' and c.get('proposed_name'):
            candidates[c['proposed_name']] = label
    return render_ids(safe_name_mentions(text, candidates), labels, allow_bare=False)


def verify_name_evidence(name, evidence, transcript=None, valid_shots=None):
    """Validate quote/time/source linkage; third-person mentions do not name a face."""
    if not isinstance(evidence, dict) or not str(name or '').strip():
        return False
    quote = str(evidence.get('quote') or '').strip()
    if not quote or (' ' + fold(name) + ' ') not in (' ' + fold(quote) + ' '):
        return False
    association = evidence.get('association')
    if association not in ('direct_address', 'self_introduction', 'onscreen_label'):
        return False
    try:
        start, end = float(evidence['start_seconds']), float(evidence['end_seconds'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= start + 30):
        return False
    evidence_shots = clean_list(evidence.get('shot_ids'))
    if valid_shots is not None:
        if not evidence_shots or not set(evidence_shots) <= set(valid_shots):
            return False
        # SHOT IDs are source-time windows; a nearby quote must not borrow a
        # face from an unrelated part of the same eight-minute clip.
        from grounded_pipeline import SHOT_SECONDS
        if not any(re.fullmatch(r'SHOT_\d+', sid) and
                   (int(sid[5:]) - 1) * SHOT_SECONDS < end and int(sid[5:]) * SHOT_SECONDS > start
                   for sid in evidence_shots):
            return False
    kind = evidence.get('kind')
    if kind == 'onscreen_text':
        shots = clean_list(evidence.get('shot_ids'))
        return bool(shots and valid_shots is not None and set(shots) <= set(valid_shots)
                    and association == 'onscreen_label')
    if kind != 'dialogue' or transcript is None:
        return False
    matching, named_rows = [], []
    for row in transcript:
        try:
            a = float(row['start_seconds']); b = float(row['end_seconds'])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if a < end and b > start:
            matching.append(str(row.get('text') or ''))
            if (' ' + fold(name) + ' ') in (' ' + fold(row.get('text') or '') + ' '):
                named_rows.append((a, b))
    if valid_shots is not None and not any(
            (int(sid[5:]) - 1) * SHOT_SECONDS < b and int(sid[5:]) * SHOT_SECONDS > a
            for sid in evidence_shots for a, b in named_rows):
        return False
    return bool(matching and fold(quote) in fold(' '.join(matching)))


def profile(character, transcript=None, valid_shots=None, strict=False):
    result = copy.deepcopy(character)
    cid = canonical_id(result.get('canonical_id') or result.get('id'))
    result['id'] = cid
    proposed = str(result.get('name') or result.get('proposed_name') or '').strip()
    evidence = result.get('name_evidence')
    explicit = verify_name_evidence(proposed, evidence, transcript, valid_shots)
    if explicit and valid_shots is None:
        try:
            explicit = any(float(evidence['start_seconds']) - 6 <= float(t) <= float(evidence['end_seconds']) + 6
                           for t in result.get('reference_seconds') or [])
        except (ValueError, TypeError, OverflowError):
            explicit = False
    # Old callers remain readable. Production passes strict=True; a generic
    # "dialogue" claim is retained as a candidate, not falsely verified.
    legacy = not strict and result.get('name_status') != 'candidate' and bool(proposed and isinstance(evidence, str) and evidence.strip()
                                and confidence(result.get('identity_confidence', result.get('confidence'))) >= .9)
    gemini = bool(proposed and result.get('name_source') == 'gemini'
                  and confidence(result.get('confidence', result.get('match_confidence', 0))) >= .9)
    confirmed = gemini or explicit or legacy or (result.get('character_registry_version') == VERSION
        and result.get('name_status') == 'confirmed' and result.get('name_verified') is True)
    result['proposed_name'] = proposed
    result['name'] = proposed if confirmed else ''
    result['name_status'] = 'confirmed' if confirmed else ('candidate' if proposed else 'generated')
    result['name_verified'] = bool(explicit or result.get('name_verified'))
    if strict:
        aliases = clean_list(result.get('aliases'))
        result['name_alias_candidates'] = clean_list(clean_list(result.get('name_alias_candidates')) + aliases)
        quote = evidence.get('quote', '') if isinstance(evidence, dict) else ''
        result['aliases'] = aliases if gemini else [a for a in aliases if confirmed and (' ' + fold(a) + ' ') in (' ' + fold(quote) + ' ')]
    result['name_candidates'] = list(result.get('name_candidates') or [])
    candidate = {'name': proposed, 'evidence': copy.deepcopy(evidence),
                 'status': 'confirmed' if confirmed else 'candidate'}
    if proposed and candidate not in result['name_candidates']:
        result['name_candidates'].append(candidate)
    alias = str(result.get('neutral_alias') or result.get('alias') or '').strip()
    if not alias or fold(alias) == fold(proposed):
        alias = f'nhân vật chưa rõ tên {int(cid[1:]) if cid else 1}'
    result['neutral_alias'] = alias
    result['display_name'] = result['name'] or alias
    for field in ('stable_features', 'face_features', 'body_features', 'distinctive_features',
                  'appearance_variants', 'reference_shot_ids'):
        result[field] = clean_list(result.get(field))
    result['character_registry_version'] = VERSION
    return result


def merge_profile(base, observed):
    result = copy.deepcopy(base)
    # A pending/conflicting view must not add another person's names to this ID.
    if observed.get('identity_conflict') == 'conflicting_name':
        return result
    if observed.get('name_status') == 'confirmed' and observed.get('name'):
        if not result.get('name') or result['name'] == observed['name']:
            for key in ('name', 'name_status', 'name_evidence', 'name_verified', 'name_source', 'display_name'):
                if key in observed:
                    result[key] = copy.deepcopy(observed[key])
    for field in ('stable_features', 'face_features', 'body_features', 'distinctive_features',
                  'appearance_variants', 'reference_shot_ids'):
        if observed.get('identity_match_status') == 'matched':
            result[field] = clean_list(clean_list(result.get(field)) + clean_list(observed.get(field)))
    result['appearances'] = list(result.get('appearances') or [])
    view = {'shot_ids': clean_list((observed.get('identity_evidence') or {}).get('shot_ids')),
            'status': observed.get('identity_match_status', 'pending')}
    if view['shot_ids'] and view not in result['appearances']:
        result['appearances'].append(view)
    result['display_name'] = result.get('name') or result['neutral_alias']
    result['name_candidates'] = list(result.get('name_candidates') or [])
    for candidate in observed.get('name_candidates') or []:
        if candidate not in result['name_candidates']:
            result['name_candidates'].append(copy.deepcopy(candidate))
    if observed.get('identity_match_status') == 'matched':
        from grounded_pipeline import SHOT_SECONDS
        verified = list(result.get('verified_reference_seconds') or [])
        evidence = observed.get('identity_evidence') or {}
        for value in evidence.get('reference_seconds') or []:
            try:
                t = float(value)
            except (ValueError, TypeError, OverflowError):
                continue
            if (math.isfinite(t) and t >= 0
                    and f'SHOT_{math.floor(t / SHOT_SECONDS) + 1:06d}' in view['shot_ids']
                    and t not in verified):
                verified.append(t)
        result['verified_reference_seconds'] = verified
    return result


def compact_registry(rows):
    """Send descriptions, not the complete appearance history or image files."""
    fields = ('id', 'name', 'name_status', 'display_name', 'neutral_alias', 'visual_signature', 'identity_conflict')
    result = []
    for row in rows:
        item = {k: row[k] for k in fields if row.get(k)}
        if row.get('name_status') == 'candidate' and row.get('proposed_name'):
            item['candidate_name'] = row['proposed_name']
        for key, limit in (('stable_features', 4), ('face_features', 3), ('body_features', 3),
                           ('distinctive_features', 3), ('appearance_variants', 3), ('reference_shot_ids', 2)):
            if row.get(key):
                item[key] = row[key][-limit:]
        result.append(item)
    return result


def cache_reference_frames(source, registry, directory, ffmpeg, duration, sampled_clips=None):
    """Store at most two existing 2-FPS source frames per identity, locally."""
    import hashlib
    import json
    import os
    import subprocess
    import uuid
    from grounded_pipeline import SHOT_SECONDS
    stat = os.stat(source)
    key = hashlib.sha256(json.dumps([os.path.realpath(source), stat.st_size,
        stat.st_mtime_ns]).encode()).hexdigest()
    root = os.path.join(directory, key)
    result = copy.deepcopy(registry)
    for person in result:
        cid = canonical_id(person.get('id'))
        if not cid:
            continue
        frames = []
        for t in (person.get('verified_reference_seconds') or [])[:2]:
            if not (isinstance(t, (int, float)) and math.isfinite(t) and 0 <= t < duration):
                continue
            sid = f'SHOT_{math.floor(t / SHOT_SECONDS) + 1:06d}'
            if sid not in person.get('reference_shot_ids', []):
                continue
            input_path, seek = source, t
            if sampled_clips is not None:
                choices = []
                for clip_path, start, end in sampled_clips:
                    if not start <= t < end or not os.path.isfile(clip_path):
                        continue
                    local = round((t - start) * 2) / 2
                    actual = start + local
                    if start <= actual < end and f'SHOT_{math.floor(actual / SHOT_SECONDS) + 1:06d}' == sid:
                        choices.append((abs(actual - t), clip_path, local, actual))
                if not choices:
                    continue
                _, input_path, seek, t = min(choices)
            elif abs(t * 2 - round(t * 2)) >= .001:
                continue
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, f'{cid}_{int(round(t * 1000))}.jpg')
            if not os.path.isfile(path) or not os.path.getsize(path):
                temporary = path + '.' + uuid.uuid4().hex + '.jpg'
                try:
                    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
                    done = subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
                        '-ss', str(seek), '-i', input_path, '-frames:v', '1', '-vf', 'scale=-2:480',
                        '-q:v', '3', temporary], capture_output=True, timeout=30, **kwargs)
                    if done.returncode or not os.path.isfile(temporary) or not os.path.getsize(temporary):
                        continue
                    os.replace(temporary, path)
                except (OSError, subprocess.TimeoutExpired):
                    continue
                finally:
                    try:
                        if os.path.exists(temporary):
                            os.remove(temporary)
                    except OSError:
                        pass
            frames.append({'shot_id': sid, 'source_seconds': t, 'local_path': path})
        person['reference_frames'] = frames
    return result
