"""Backend-timed observations and a proof-carrying narration plan."""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from visual_audit_diagnostics import inspect_audit_response, summarize_audit_scenes

VERSION = 1
SHOT_SECONDS = 6.0


def observation_response_schema():
    string = {'type': 'string'}
    strings = {'type': 'array', 'items': string}
    character = {'type': 'object', 'properties': {
        **{k: string for k in ('id', 'canonical_id', 'name', 'alias', 'visual_signature')},
        **{k: strings for k in ('face_features', 'body_features', 'distinctive_features', 'appearance_variants')},
        'name_evidence': {'type': 'object', 'properties': {
            'kind': {'type': 'string', 'enum': ['dialogue', 'onscreen_text', 'unknown']},
            'quote': string, 'start_seconds': {'type': 'number'}, 'end_seconds': {'type': 'number'},
            'shot_ids': strings,
            'association': {'type': 'string', 'enum': ['direct_address', 'self_introduction', 'onscreen_label', 'third_person', 'uncertain']}},
            'required': ['kind', 'quote', 'start_seconds', 'end_seconds', 'shot_ids', 'association']},
        'match_confidence': {'type': 'number'},
        'identity_evidence': {'type': 'object', 'properties': {
            'matched_features': strings, 'shot_ids': strings,
            'reference_seconds': {'type': 'array', 'maxItems': 2, 'items': {'type': 'number'}}},
            'required': ['matched_features', 'shot_ids']}},
        'required': ['id', 'canonical_id', 'name', 'alias', 'visual_signature',
                     'match_confidence', 'name_evidence', 'identity_evidence']}
    observation = {'type': 'object', 'properties': {'text': string, 'shot_ids': strings, 'actor_ids': strings,
        'story_beat_ids': strings,
        'evidence_kind': {'type': 'string', 'enum': ['visual']},
        'status': {'type': 'string', 'enum': ['confirmed', 'uncertain']}},
        'required': ['text', 'shot_ids', 'actor_ids', 'evidence_kind', 'status']}
    event = {'type': 'object', 'properties': {
        **{k: string for k in ('local_event_id', 'situation_id', 'event', 'sequence_type', 'story_role')},
        'content_kind': {'type': 'string', 'enum': ['story', 'non_story']},
        'scene_context': {'type': 'object', 'properties': {
            k: string for k in ('location', 'time_context', 'goal')},
            'required': ['location', 'time_context', 'goal']},
        'characters': strings, 'depends_on_local': strings,
        'observations': {'type': 'array', 'items': observation}},
        'required': ['local_event_id', 'situation_id', 'event', 'scene_context',
                     'characters', 'content_kind', 'sequence_type', 'story_role', 'depends_on_local', 'observations']}
    return {'type': 'object', 'properties': {'characters': {'type': 'array', 'items': character},
        'events': {'type': 'array', 'items': event},
        'plot_gaps': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'beat_id': string, 'reason': string, 'shot_ids': strings},
            'required': ['beat_id', 'reason', 'shot_ids']}},
        'review_candidates': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'text': string, 'reason': string, 'shot_ids': strings}, 'required': ['text', 'reason', 'shot_ids']}},
        'end_state': {'type': 'object', 'properties': {
            'location': string, 'ongoing_action': string, 'characters': strings},
            'required': ['location', 'ongoing_action', 'characters']}},
        'required': ['characters', 'events', 'review_candidates', 'end_state']}


def parse_observations(text):
    """Read the whole observation document, never an object inside an array."""
    source = str(text or '').lstrip('\ufeff').strip()
    decoder = json.JSONDecoder(strict=False)
    candidates = []
    for match in re.finditer(r'[\[{]', source):
        try:
            value, end = decoder.raw_decode(source, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            candidates.append((end - match.start(), value))
    if not candidates:
        raise ValueError('OBSERVATION_SCHEMA_INVALID: khong co JSON hoan chinh')
    value = max(candidates, key=lambda item: item[0])[1]
    # Gemini sometimes wraps the complete document in a one-item array.
    # Unwrap only an explicit events envelope, retaining all its metadata.
    if (isinstance(value, list) and len(value) == 1
            and isinstance(value[0], dict) and 'events' in value[0]):
        value = value[0]
    if isinstance(value, list):
        # An array contains events, not a single event chosen by object size.
        if not all(isinstance(row, dict) and isinstance(row.get('observations'), list)
                   and isinstance(row.get('event'), str) for row in value):
            raise ValueError('OBSERVATION_SCHEMA_INVALID: danh sach khong phai cac su kien')
        return {'events': value}
    if not isinstance(value.get('events'), list):
        raise ValueError('OBSERVATION_SCHEMA_INVALID: events phai la danh sach')
    return value


def observation_cache_path(directory, source, target_words, contract_version, subtitle_text='', fps=None):
    stat = os.stat(source)
    signature = [os.path.realpath(source), stat.st_size, stat.st_mtime_ns,
                 int(target_words), int(contract_version), subtitle_text]
    if fps is not None:
        signature.append({'analysis_fps': float(fps)})
    key = hashlib.sha256(json.dumps(signature, ensure_ascii=False).encode()).hexdigest()
    return os.path.join(directory, 'grounded_observations', key + '.json')


def load_observation_cache(path):
    try:
        with open(path, encoding='utf-8') as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def save_observation_cache(path, state):
    from task_store import _atomic_write_json
    _atomic_write_json(path, state)


def shot_index(duration):
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('SOURCE_DURATION_INVALID')
    return {f'SHOT_{i + 1:06d}': {'shot_id': f'SHOT_{i + 1:06d}',
            'start_seconds': i * SHOT_SECONDS, 'end_seconds': min(duration, (i + 1) * SHOT_SECONDS)}
            for i in range(math.ceil(duration / SHOT_SECONDS))}


def indexed_clip(source, output, start, end, ffmpeg='ffmpeg', fps=1.0):
    """Overlay stable source IDs. Sampling changes neither speed nor offsets."""
    stat = os.stat(source)
    identity = [2, os.path.realpath(source), stat.st_size, stat.st_mtime_ns,
                float(start), float(end), float(fps)]
    stamp = output + '.source.json'
    cached = load_observation_cache(stamp)
    if (cached.get('identity') == identity and os.path.isfile(output)
            and os.path.getsize(output) > 0 and cached.get('bytes') == os.path.getsize(output)):
        return output
    probe = subprocess.run([ffmpeg, '-hide_banner', '-filters'], capture_output=True, text=True, errors='replace', timeout=30)
    if 'drawtext' not in (probe.stdout or '') + (probe.stderr or ''):
        ffmpeg = shutil.which('ffmpeg') or ffmpeg
    font = "fontfile='C\\:/Windows/Fonts/arialbd.ttf':" if os.name == 'nt' else ''
    label = f"SHOT_%{{eif\\:floor((t+{float(start):.6f})/{SHOT_SECONDS})+1\\:d\\:6}}"
    filters = (f'scale=-2:480,fps={float(fps):g},setpts=PTS-STARTPTS,'
               f"drawtext={font}text='{label}':x=12:y=12:fontsize=26:fontcolor=white:borderw=3:bordercolor=black")
    result = subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{start:.6f}', '-i', source, '-t', f'{end-start:.6f}', '-an',
        '-vf', filters, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '29',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output], capture_output=True, text=True, errors='replace', timeout=900)
    if result.returncode or not os.path.isfile(output):
        raise RuntimeError('INDEXED_CLIP_FAILED: ' + (result.stderr or '')[-1500:])
    save_observation_cache(stamp, {'identity': identity, 'bytes': os.path.getsize(output)})
    return output


def non_story(text):
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(text)).casefold() if not unicodedata.combining(c))
    return bool(re.search(
        r'\b(logo|credits?|opening credits|end credits|title card|studio ident|'
        r'opening montage|intro montage|opening sequence|title sequence|'
        r'opening song|theme song|nhac mo dau|bai hat mo dau|'
        r'doan gioi thieu nhan vat|montage gioi thieu|phim gioi thieu|'
        r'danh sach doan lam phim|bieu tuong hang phim)\b', text))


def observation_prompt(chunk, start, end, desired_facts, context=None):
    first = math.floor(float(start) / SHOT_SECONDS) + 1
    last = math.ceil(float(end) / SHOT_SECONDS)
    return (
        f'Xem clip {chunk}, phim nguon {start:.2f}-{end:.2f}s. '
        f'Nhan hop le trong clip: SHOT_{first:06d} den SHOT_{last:06d}. '
        'ID danh so lien tuc theo TOAN PHIM, KHONG danh so lai tu 1 khi sang clip moi. '
        'Doc dung nhan SHOT tren hinh; khong lay so thu tu frame hay giay trong clip lam ID. '
        'NHIEM VU: ghi du kien nhin thay CUNG ID SHOT tren hinh; CHUA viet recap. '
        'Nguoi trong observation phai tham chieu bang [CHAR=C1] (ID trong characters), '
        'kem actor_ids; KHONG chep ten rieng vao text de backend phai tim-thay chu. '
        'Neu chi nhac den nguoi khac trong loi thoai, khong gan nguoi do thanh actor dang hien tren hinh. '
        'Ghi ro chu the, hanh dong cu the va doi tuong khi thuc su nhin thay. '
        'Khong nang "dung truoc cua" thanh "mo cua", khong bien bieu cam thanh noi tam. '
        'Moi observation co evidence_kind="visual", status="confirmed" chi khi hinh '
        'chung minh; chua ro ghi status="uncertain" va ly do trong review_candidates. '
        'Transcript la ngu canh rieng; khong dua dieu chi nghe duoc vao observation hinh. '
        'Doi chieu cac story beat trong outline: tim mat xich muc tieu, tro ngai, '
        'buoc ngoat, cao trao va ket qua. Observation nao thuc su ho tro beat thi '
        'ghi story_beat_ids=["GB..."]; khong gan chi vi trung thoi gian hay ten nguoi. '
        'Khong chep summary tong quan thanh fact. Neu khong thay thi de trong, khong bia. '
        'Neu co mat xich trong outline chua tim thay hoac chi ho tro duoc mot phan, '
        'ghi plot_gaps gom beat_id, reason va shot_ids vung can doi chieu; '
        'khong danh dau thieu chi vi ket qua nam ngoai clip hien tai. '
        f'Co gang ghi khoang {desired_facts} du kien khac nhau tren TOAN BO clip, khong chi dau clip. '
        'So luong la muc tieu tim bang chung, khong duoc bia, lap cam xuc hay doan frame de du so. '
        'Quan sat CHI TIET TUNG SU KIEN: ghi rieng hanh dong bat dau, cac thao tac noi tiep, '
        'tro ngai nhin thay, phan ung cu the cua nhan vat va ket qua xuat hien tren hinh. '
        'Moi su kien co the gom 1-8 hoac nhieu hon du kien neu that su co dien bien moi, '
        'bang tieng Viet va theo thu tu quan sat. Khong chia mot hanh dong thanh cac cau dong nghia. '
        'Giu nguyen nhan-hanh dong-ket qua; danh dau hanh dong dang tiep dien qua bien clip. '
        'DON VI EVENT LA TINH HUONG, KHONG PHAI TUNG DONG TAC: cung nhan vat, muc tieu, '
        'boi canh thi gom boi/leo/bat ca vao mot event, cac chi tiet nam trong observations. '
        'Tach event khi doi tuyen nhan vat, thoi gian, dia diem hay muc tieu; '
        'tuyet doi khong ghep cong bien mat voi me mang Nobita. '
        'Gan situation_id on dinh cho cung tinh huong trong clip; khong tao event de du so cau. '
        'Moi tinh huong ghi scene_context gom location, time_context, goal; '
        'hai tuyen khac dia diem/nhan vat phai dung situation_id khac du cung chu de bo nha di. '
        'Chi dung shot_ids THUC SU DOC DUOC tren hinh. KHONG tu viet timestamp. '
        'Neu khong ro ID hoac khong chac hanh dong, de vao review_candidates, khong tao observation. '
        'Logo, credits, title card, outro khong phai du kien cot truyen; danh dau content_kind="non_story" va bo. '
        'Ca doan nhac mo dau/opening, montage gioi thieu nhan vat, canh nhan vat tao dang, '
        'bay luon hoac sinh vat xuat hien trong intro deu la non_story, DU khong co chu tren hinh. '
        'Khong ke cac canh intro nhu su kien dang xay ra, khong ghep chung voi canh cot truyen. '
        'Neu mot shot vua co title/logo toan man hinh vua co cot truyen, bo shot tron do; '
        'chi lay shot cot truyen sach tiep theo. Logo kenh nho o goc (watermark) '
        'khong tu no bien canh cot truyen thanh intro. Canh mo phim co dien bien that van la story. '
        'non_story_ranges trong context CHI LA GOI Y CUA BAN TONG QUAN, co the sai. '
        'Xem TOAN BO clip ke ca vung do; tu xac dinh ranh gioi story/non_story bang nhan SHOT. '
        'Khong mac dinh vai phut dau la opening. Giu cold open, gioi thieu boi canh, '
        'nguyen nhan bo nha va cac su kien truoc bai hat. Non_story van ghi observations '
        'voi shot_ids da doc de danh dau vung loai chinh xac. Khong tru thoi gian opening '
        'khoi ID/timestamp; moi ID luon theo phim goc. '
        'Khong suy dien noi tam, loi thoai, bao boi hay ket qua tu tri nho phim. Transcript chi giup hieu ngu canh. '
        'Mot observation la mot menh de cu the, khong phai nhieu cach dien dat lai mot quyet dinh. '
        'Dung ten da xac nhan; chua chac dung biet danh ngan. '
        'HO SO NHAN VAT trong CONTEXT la danh ba goc: KHONG doi ten/ID, KHONG dung lai ID '
        'cho nguoi khac. Doi chieu it nhat hai dac diem ben vung (mat, dang nguoi, mau long, '
        'dau hieu rieng); quan ao co the thay doi. Khong nhan dang chi dua vao vi tri hay vai tro. '
        'Moi nhan vat quan sat phai khai bao trong characters; events.characters chi tham chieu ID da khai bao. '
        'matched_features ghi dac diem that su thay, doi chieu tong hop stable_features, '
        'visual_signature va appearance_variants; khong can chep nguyen hai cum stable_features. '
        'Mot dac diem ben vung kem dac diem ngoai hinh phu hop co the chung minh cung nguoi. '
        'identity_evidence.shot_ids duoc dung moi shot trong clip da xem, khong bat buoc chon vao recap. '
        'Neu thieu bang chung cho ID da biet: GIU canonical_id goi y va ha match_confidence; '
        'khong tu doi thanh nguoi moi, khong dat ID moi chi vi doi quan ao. '
        'Neu khop ho so: canonical_id la ID P da co, match_confidence>=0.9 chi khi chac chan. '
        'Neu chua khop: canonical_id="", name="", dung alias trung tinh, khong doan theo tri nho phim. '
        'GEMINI la nguon ten chinh: giu ten ban nhan dien chac chan qua hinh/ngu canh/am thanh, '
        'Whisper chi tham khao va khong duoc doi ten da nhan dien chi vi phien am sai. '
        'Neu khong chac ten thi dung alias; khong bia ten. '
        'name_evidence phai la object: kind, quote NGUYEN VAN, start_seconds/end_seconds '
        'theo phim goc, shot_ids va association. Chi direct_address/self_introduction/onscreen_label '
        'moi gan ten voi nguoi; third_person chi la ten duoc nhac, khong phai ten nguoi dang thay. '
        'Chua ro thi kind=unknown, quote="", association=uncertain. '
        'Ghi face_features, body_features, distinctive_features khi that su thay ro; '
        'appearance_variants chi la quan ao/hoa trang. Khong bia cho du hai dac diem. '
        'Neu thay ro mat/toan than, ghi toi da hai identity_evidence.reference_seconds '
        'tren phim goc, dung moc da thay trong clip 2 FPS, nam trong '
        'shot_ids da khai bao. Khong ro thi de rong, khong doan. '
        'reference_seconds chi la moc tham chieu danh tinh; timestamp cua su kien '
        'van do backend suy ra tu SHOT ID, khong tu tao. '
        'Tra JSON {"characters":[{"id":"C1","canonical_id":"","name":"","alias":"",'
        '"visual_signature":"","match_confidence":0.0,"name_evidence":{"kind":"unknown",'
        '"quote":"","start_seconds":0,"end_seconds":0,"shot_ids":[],"association":"uncertain"},'
        '"identity_evidence":{"matched_features":[],"shot_ids":[]}}],'
        '"events":[{"local_event_id":"L001","situation_id":"SIT001","event":"tom tat tinh huong",'
        '"scene_context":{"location":"","time_context":"","goal":""},'
        '"characters":[],"content_kind":"story","sequence_type":"narrative","story_role":"development",'
        '"depends_on_local":[],"observations":[{"text":"Du kien da thay",'
        + '"shot_ids":["SHOT_000001"],"actor_ids":[],"evidence_kind":"visual","status":"confirmed"}]}],"review_candidates":[],"end_state":{}}. '.replace(
            'SHOT_000001', f'SHOT_{first:06d}') +
        'Du kien co the cung shot neu hinh that chung minh cac thay doi khac nhau. '
        'Tim cac SHOT khac nhau o dau, giua va cuoi cung mot su kien de giu day du dien bien; '
        'khong chi ghi mot tom tat chung cho ca su kien. So phut recap se duoc ap dung SAU, '
        'khong bo chi tiet that chi vi recap du kien ngan. '
        'Uu tien tim cac moc SHOT khac nhau tren toan clip; cac du kien chung mot vung hinh '
        'se duoc gom thanh mot don vi ke, khong cat deu hinh de tao them cau. '
        'CONTEXT CHI DE NOI MACH, KHONG PHAI BANG CHUNG HINH: ' + json.dumps(context or {}, ensure_ascii=False)
    )


def recover_observations(text, catalog, start, end, chunk_number, regenerate, save=None):
    """Retry invalid documents against the same video, never infer shot offsets."""
    for attempt in range(3):
        if save:
            save(text, attempt)
        try:
            bound = bind_observations(parse_observations(text), catalog, start, end, chunk_number)
            rejected = bound.get('_grounding_rejections', [])
            invalid = [r for r in rejected if r['reason'] != 'non_story']
            review = bound.get('review_candidates') or []
            # Valid facts survive partial uncertainty. Never replay an entire
            # expensive clip just to fill every review candidate or bad ID.
            if bound['events']:
                return bound
            if invalid:
                problem = json.dumps(invalid[:8], ensure_ascii=False)
            elif review:
                problem = 'CAN TIM VA DOC LAI SHOT CHO CAC HANH DONG CHUA RO: ' + json.dumps(review, ensure_ascii=False)[:6000]
            else:
                problem = 'Khong co su kien hop le.'
        except (ValueError, TypeError, AttributeError) as error:
            problem = str(error)
        if attempt == 2:
            raise ValueError(f'GROUNDED_OBSERVATIONS_INVALID: doan {chunk_number}; {problem}')
        text = regenerate(
            'PHAN HOI TRUOC KHONG HOP LE: ' + problem + '\n'
            'Xem lai video dinh kem va tra LAI TOAN BO JSON. Doc truc tiep nhan SHOT tren hinh. '
            'Khong tu cong offset, khong danh so lai tu 1, khong liet ke dai ID de phu kin clip. '
            'Moi du kien chi gan cac ID thuc su chung minh hanh dong. '
            'Neu khong doc duoc nhan, ghi review_candidates thay vi doan. '
            'Bo cac ID sai, dong thoi giu lai day du cac du kien da xac minh.'
        )


def canonical_shot_id(value, catalog):
    """Normalize numeric padding only; never guess a different shot number."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r'SHOT_([0-9]{1,32})', value.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    sid = f'SHOT_{int(match.group(1)):06d}'
    return sid if sid in catalog else None


def bind_observations(payload, catalog, start, end, chunk_number):
    """Reject guessed timestamps; derive all source ranges from known shot IDs."""
    result = {k: copy.deepcopy(v) for k, v in payload.items() if k != 'events'}
    result['events'], rejected, seen, normalizations = [], [], set(), {}
    result['confirmed_non_story_shots'] = []
    result['review_candidates'] = list(result.get('review_candidates') or []) if isinstance(result.get('review_candidates', []), list) else []
    result['dialogue_context'] = []
    result['plot_gaps'] = []
    for gap in payload.get('plot_gaps') or []:
        if not isinstance(gap, dict) or not re.fullmatch(r'GB\d+', str(gap.get('beat_id') or '')):
            continue
        ids = [canonical_shot_id(sid, catalog) for sid in gap.get('shot_ids') or []]
        if ids and all(sid and catalog[sid]['start_seconds'] < end and catalog[sid]['end_seconds'] > start for sid in ids):
            result['plot_gaps'].append(dict(beat_id=gap['beat_id'], reason=str(gap.get('reason') or ''),
                shot_ids=list(dict.fromkeys(ids))))
    declared_actors = {str(c.get('id')).upper() for c in payload.get('characters') or [] if isinstance(c, dict)}
    for event_number, raw in enumerate(payload.get('events') or [], 1):
        if not isinstance(raw, dict):
            continue
        if (raw.get('content_kind') != 'story' or non_story(raw.get('event', ''))
                or str(raw.get('sequence_type', '')).lower() in {
                    'intro', 'opening_montage', 'title_sequence', 'credits', 'studio_logo'}):
            # Only the detailed view can exclude source frames. Overview
            # ranges are proposals, never instructions to trim a cold open.
            for observation in raw.get('observations') or []:
                if not isinstance(observation, dict):
                    continue
                for value in observation.get('shot_ids') or []:
                    sid = canonical_shot_id(value, catalog)
                    if sid and catalog[sid]['end_seconds'] > start and catalog[sid]['start_seconds'] < end:
                        result['confirmed_non_story_shots'].append(copy.deepcopy(catalog[sid]))
            rejected.append({'event': raw.get('event'), 'reason': 'non_story'})
            continue
        facts, sources = [], {}
        for index, observation in enumerate(raw.get('observations') or [], 1):
            if not isinstance(observation, dict) or not isinstance(observation.get('text'), str):
                continue
            text = re.sub(r'\s+', ' ', observation['text']).strip()
            if 'actor_ids' in observation:
                actor_ids = observation['actor_ids']
                if (not isinstance(actor_ids, list) or any(not isinstance(i, str) or i.upper() not in declared_actors for i in actor_ids)
                        or not {i.upper() for i in re.findall(r'\[CHAR=([A-Za-z0-9_]+)\]', text, re.I)} <= {i.upper() for i in actor_ids}):
                    rejected.append({'event': raw.get('event'), 'reason': 'unknown_actor_reference'})
                    continue
            kind = observation.get('evidence_kind', 'visual')
            status = observation.get('status', 'confirmed')
            if status != 'confirmed' or kind not in ('visual', 'dialogue'):
                result['review_candidates'].append(dict(text=text,
                    shot_ids=copy.deepcopy(observation.get('shot_ids') or []),
                    reason='unconfirmed_visual_observation'))
                continue
            if kind == 'dialogue':
                result['dialogue_context'].append(copy.deepcopy(observation))
                continue
            ids = observation.get('shot_ids')
            if (not text or non_story(text) or observation.get('content_kind', 'story') != 'story'
                    or not isinstance(ids, list) or not ids):
                continue
            canonical_ids = [canonical_shot_id(sid, catalog) for sid in ids]
            if any(sid is None for sid in canonical_ids):
                rejected.append({'event': raw.get('event'), 'reason': 'unknown_shot_id', 'shot_ids': ids})
                continue
            for original, canonical in zip(ids, canonical_ids):
                if original != canonical:
                    normalizations[original] = canonical
            ids = list(dict.fromkeys(canonical_ids))
            shots = [catalog[sid] for sid in ids]
            if any(s['end_seconds'] <= start or s['start_seconds'] >= end for s in shots):
                rejected.append({'event': raw.get('event'), 'reason': 'shot_outside_attached_clip', 'shot_ids': ids})
                continue
            signature = (' '.join(text.casefold().split()), tuple(sorted(ids)))
            if signature in seen:
                continue
            seen.add(signature)
            fid = f'F{chunk_number:03d}_{event_number:04d}_{index:02d}'
            fact = {'fact_id': fid, 'text': text, 'shot_ids': ids,
                    'raw_text': observation['text']}
            if isinstance(observation.get('story_beat_ids'), list):
                fact['story_beat_ids'] = list(dict.fromkeys(v for v in observation['story_beat_ids']
                                                          if isinstance(v, str) and re.fullmatch(r'GB\d+', v)))
            if 'actor_ids' in observation:
                fact['actor_ids'] = copy.deepcopy(observation['actor_ids'])
            # Preserve the explicit contract for new observations while keeping
            # old checkpoints readable without manufacturing certainty fields.
            if 'evidence_kind' in observation or 'status' in observation:
                fact.update(evidence_kind='visual', status='confirmed')
            facts.append(fact)
            for shot in shots:
                sid = shot['shot_id']
                source = sources.setdefault(sid, {'shot_id': sid, 'start_seconds': max(start, shot['start_seconds']) - start,
                    'end_seconds': min(end, shot['end_seconds']) - start, 'description': '', 'fact_ids': []})
                source['description'] = (source['description'] + ' ' + text).strip()
                source['fact_ids'].append(fid)
        if not facts:
            continue
        event = copy.deepcopy(raw)
        event['characters'] = list(dict.fromkeys(list(raw.get('characters') or [])
            + [i for fact in facts for i in fact.get('actor_ids') or []]))
        event.update(event=' '.join(f['text'] for f in facts), granularity='atomic',
                     grounded_evidence_version=VERSION, grounded_facts=facts,
                     start_seconds=min(s['start_seconds'] for s in sources.values()),
                     end_seconds=max(s['end_seconds'] for s in sources.values()),
                     visual_shots=list(sources.values()))
        # Unobserved summaries/causes are context only, never extra sentence facts.
        for field in ('cause', 'action', 'result', 'consequence', 'state_before', 'state_after', 'change'):
            event[field] = ''
        result['events'].append(event)
    result['_grounding_rejections'] = rejected
    result['_shot_id_normalizations'] = normalizations
    return result


def grounded_detail(cluster, events):
    """Allocate actual observations, not model-expanded filler, to exact slots."""
    known = {e['evidence_id']: e for e in events}
    pools = {eid: list(known[eid].get('grounded_facts') or []) for eid in cluster['event_ids']}
    quota = cluster['sentence_count']
    if any(not facts for facts in pools.values()):
        raise ValueError('PLAN_GROUNDING_MISSING:' + cluster['cluster_id'])
    if sum(map(len, pools.values())) < quota:
        raise ValueError(f'PLAN_GROUNDED_FACTS_SHORTAGE:{cluster["cluster_id"]}:need={quota}')
    counts = {eid: 1 for eid in pools}
    remaining = quota - len(counts)
    while remaining > 0:
        progressed = False
        for eid, facts in pools.items():
            if counts[eid] < len(facts):
                counts[eid] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError('PLAN_GROUNDED_FACTS_SHORTAGE:' + cluster['cluster_id'])
    if quota >= len(pools):
        groups = [[(eid, f) for f in facts[len(facts) * i // counts[eid]:len(facts) * (i + 1) // counts[eid]]]
                  for eid, facts in pools.items() for i in range(counts[eid])]
    else:
        event_ids = list(pools)
        groups = [[(eid, f) for eid in event_ids[len(event_ids) * i // quota:len(event_ids) * (i + 1) // quota]
                   for f in pools[eid]] for i in range(quota)]
    return {'cluster_id': cluster['cluster_id'], 'sentences': [
        {'source_event_ids': list(dict.fromkeys(eid for eid, _ in group)),
         'focus': ' '.join(f['text'] for _, f in group),
         'fact_ids': [f['fact_id'] for _, f in group]} for group in groups]}


def deduplicate_observations(chunks):
    result, seen, seen_ids, removed = copy.deepcopy(chunks), set(), set(), 0
    for chunk in result:
        kept = []
        for event in chunk.get('events') or []:
            facts = []
            for fact in event.get('grounded_facts') or []:
                key = (' '.join(fact['text'].casefold().split()), tuple(sorted(fact['shot_ids'])))
                ids = set(fact.get('constituent_fact_ids') or [fact['fact_id']])
                if key in seen or ids <= seen_ids:
                    removed += 1
                    continue
                seen.add(key)
                seen_ids.update(ids)
                facts.append(fact)
            if facts:
                event['grounded_facts'] = facts
                event['event'] = ' '.join(f['text'] for f in facts)
                kept.append(event)
        chunk['events'] = kept
    # Facts sharing source windows form one visual unit. Dividing a six-second
    # window equally would claim that each fragment proves a different action,
    # despite never having observed that fragment in isolation.
    entries = [(event, fact) for chunk in result for event in chunk['events']
               for fact in event['grounded_facts']]
    parents = list(range(len(entries)))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    shot_owner = {}
    for index, (event, fact) in enumerate(entries):
        # Shared timestamps do not prove shared situations. Nor does A/B,
        # B/C imply that A and C illustrate the same action.
        key = (id(event), tuple(sorted(fact['shot_ids'])))
        if key in shot_owner:
            parents[root(index)] = root(shot_owner[key])
        else:
            shot_owner[key] = index
    groups = {}
    for index in range(len(entries)):
        groups.setdefault(root(index), []).append(index)
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        owner, first = entries[indexes[0]]
        first['text'] = ' '.join(dict.fromkeys(entries[i][1]['text'] for i in indexes))
        first['shot_ids'] = sorted({sid for i in indexes for sid in entries[i][1]['shot_ids']})
        first['constituent_fact_ids'] = sorted({fid for i in indexes for fid in
            (entries[i][1].get('constituent_fact_ids') or [entries[i][1]['fact_id']])})
        for index in indexes[1:]:
            event, fact = entries[index]
            event['grounded_facts'].remove(fact)
            existing = {s['shot_id'] for s in owner.get('visual_sources') or []}
            owner.setdefault('visual_sources', []).extend(copy.deepcopy(s) for s in event.get('visual_sources') or []
                                                       if s['shot_id'] in first['shot_ids'] and s['shot_id'] not in existing)
            removed += 1
    for chunk in result:
        chunk['events'] = [e for e in chunk['events'] if e['grounded_facts']]
        for event in chunk['events']:
            event['grounded_facts'].sort(key=lambda f: min(f['shot_ids']))
            event['event'] = ' '.join(f['text'] for f in event['grounded_facts'])
            supported = {sid for f in event['grounded_facts'] for sid in f['shot_ids']}
            event['visual_sources'] = [s for s in event.get('visual_sources') or [] if s['shot_id'] in supported]
            event['start_seconds'] = min(s['start_seconds'] for s in event.get('visual_sources') or [])
            event['end_seconds'] = max(s['end_seconds'] for s in event.get('visual_sources') or [])
        chunk['events'].sort(key=lambda e: e['start_seconds'])
    valid = {e['evidence_id'] for c in result for e in c['events']}
    for chunk in result:
        for event in chunk['events']:
            event['depends_on'] = [key for key in event.get('depends_on') or [] if key in valid]
    return result, removed


def supplement_observations(chunks, duration, target, extract, checkpoint=None, save=None):
    """At most two localized observation passes, before planning/writing."""
    result = copy.deepcopy(chunks)
    state = copy.deepcopy(checkpoint or {'completed': []})
    for saved in state['completed']:
        result[saved['chunk_index']]['events'].extend(copy.deepcopy(saved['events']))
    result, _ = deduplicate_observations(result)
    for attempt in range(len(state['completed']), 2):
        count = sum(len(e.get('grounded_facts') or []) for c in result for e in c['events'])
        if count >= target:
            break
        spans = sorted((float(s['start_seconds']), float(s['end_seconds']))
                       for c in result for e in c['events'] for s in e.get('visual_sources') or [])
        spans = sorted(spans + [(float(s['start']), float(s['end'])) for s in state['completed']])
        gaps, cursor = [], 0.0
        for start, end in spans:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < duration:
            gaps.append((cursor, duration))
        if not gaps:
            break
        left, right = max(gaps, key=lambda span: span[1] - span[0])
        length = min(240.0, max(60.0, (target - count) * 8.0), duration)
        start = max(0.0, min(duration - length, (left + right - length) / 2))
        end = min(duration, start + length)
        center = (start + end) / 2
        owner = min(range(len(result)), key=lambda i: abs(center - (
            float(result[i].get('start_seconds', 0)) + float(result[i].get('end_seconds', duration))) / 2))
        events = extract(start, end, target - count, attempt + 1)
        result[owner]['events'].extend(copy.deepcopy(events))
        state['completed'].append({'chunk_index': owner, 'events': events, 'start': start, 'end': end})
        if save:
            save(copy.deepcopy(state))
        result, _ = deduplicate_observations(result)
    return result, len(state['completed'])


def rebuild_plan_in_source_order(plan, chunks):
    """Keep the exact slot count while regrouping observed facts chronologically."""
    result = copy.deepcopy(plan)
    events = {e['evidence_id']: e for c in chunks for e in c.get('events') or []}
    facts = {f['fact_id']: f for e in events.values() for f in e.get('grounded_facts') or []}
    sources = {s['shot_id']: s for e in events.values() for s in e.get('visual_sources') or []}
    slots = [s for part in result['events_by_part'] for s in part]
    ids = list(dict.fromkeys(fid for s in slots for fid in s.get('planned_fact_ids') or []))
    ids.sort(key=lambda fid: min(sources[sid]['start_seconds'] for sid in facts[fid]['shot_ids']))
    if len(ids) < len(slots):
        raise ValueError('PLAN_GROUNDED_FACTS_SHORTAGE: cannot rebuild exact slots')
    for index, slot in enumerate(slots):
        selected = ids[len(ids)*index//len(slots):len(ids)*(index+1)//len(slots)]
        owned = list(dict.fromkeys(facts[fid]['source_event_id'] for fid in selected))
        focus = ' '.join(facts[fid]['text'] for fid in selected)
        slot.update(planned_fact_ids=selected, narration_focus=focus, event=focus,
                    planning_source_event_ids=owned, merged_evidence_ids=owned,
                    supporting_facts=[copy.deepcopy(events[eid]) for eid in owned],
                    characters=list(dict.fromkeys(str(who) for eid in owned for who in events[eid].get('characters') or [])))
    result['visual_order_repaired'] = True
    result['illustration_selection_mode'] = 'observed_representative'
    return result


def lock_plan_visuals(plan, chunks, allow_rebuild=True, representative=False):
    """Every planned sentence gets observed facts and exact source frames BEFORE prose."""
    result = copy.deepcopy(plan)
    facts, sources, fact_sources = {}, {}, {}
    for chunk in chunks:
        for event in chunk.get('events') or []:
            for fact in event.get('grounded_facts') or []:
                facts[fact['fact_id']] = fact
                fact_sources[fact['fact_id']] = {
                    s['shot_id']: dict(copy.deepcopy(s),
                        event=fact['text'], fact_ids=[fact['fact_id']],
                        source_event_id=event.get('evidence_id'),
                        source_start_seconds=float(s['start_seconds']),
                        source_end_seconds=float(s['end_seconds']))
                    for s in event.get('visual_sources') or []
                    if s.get('shot_id') in fact.get('shot_ids', [])}
            for source in event.get('visual_sources') or []:
                if source.get('shot_id'):
                    sid = source['shot_id']
                    if sid in sources:
                        sources[sid]['start_seconds'] = min(sources[sid]['start_seconds'], source['start_seconds'])
                        sources[sid]['end_seconds'] = max(sources[sid]['end_seconds'], source['end_seconds'])
                        sources[sid]['fact_ids'] = list(dict.fromkeys(
                            list(sources[sid].get('fact_ids') or []) + list(source.get('fact_ids') or [])))
                    else:
                        sources[sid] = copy.deepcopy(source)
    scenes = [s for part in result['events_by_part'] for s in part]
    chosen, used_shots, used_facts = {}, set(), set()
    for index, scene in enumerate(scenes):
        ids = scene.get('planned_fact_ids') or []
        if not ids or any(fid not in facts or fid in used_facts for fid in ids):
            raise ValueError('PLAN_FACT_BINDING_INVALID:' + scene['scene_id'])
        used_facts.update(ids)
        shots = []
        scene_shots = set()
        for fid in ids:
            planned_fact = next((f for f in scene.get('observed_facts') or [] if f.get('fact_id') == fid), facts[fid])
            proof_ids = planned_fact.get('shot_ids') or []
            candidates = [fact_sources[fid][sid] for sid in proof_ids if sid in fact_sources[fid]]
            if not proof_ids or len(candidates) != len(proof_ids):
                raise ValueError('PLAN_SHOT_BINDING_MISSING:' + fid)
            if result.get('situation_plan_version', 0) >= 4 and not representative:
                limit = float(scenes[index + 1]['start_seconds']) if index + 1 < len(scenes) else float(result['boundaries'][-1])
                candidates = [s for s in candidates if
                    float(s['start_seconds']) >= float(scene['start_seconds']) - .01
                    and float(s['end_seconds']) <= limit + .01]
                if not candidates:
                    raise ValueError('PLAN_SITUATION_PROOF_MISSING:' + fid)
            elif representative or result.get('situation_plan_version', 0) >= 2:
                # This is an observed illustration of the fact, not a claim
                # that every window of a long action is needed in the recap.
                candidates = [min(candidates, key=lambda s: float(s['start_seconds']))]
            fact_shots = []
            for source in sorted(candidates, key=lambda s: float(s['start_seconds'])):
                if scene.get('situation_narration') and source['shot_id'] in scene_shots:
                    continue
                if source['shot_id'] in used_shots:
                    raise ValueError('PLAN_SOURCE_REUSED:' + scene['scene_id'])
                used_shots.add(source['shot_id'])
                scene_shots.add(source['shot_id'])
                # Adjacent IDs supporting ONE fact can remain one continuous
                # action clip. Never discard later proof windows or bridge a gap.
                if (fact_shots and abs(fact_shots[-1]['end_seconds'] - source['start_seconds']) < .01
                        and source['end_seconds'] - fact_shots[-1]['start_seconds'] <= 15):
                    fact_shots[-1]['end_seconds'] = source['end_seconds']
                    fact_shots[-1]['shot_ids'].append(source['shot_id'])
                    fact_shots[-1]['fact_ids'] = list(dict.fromkeys(
                        fact_shots[-1].get('fact_ids', []) + source.get('fact_ids', [])))
                else:
                    fact_shots.append(dict(copy.deepcopy(source), shot_ids=[source['shot_id']]))
            shots.extend(fact_shots)
        chosen[index] = sorted(shots, key=lambda s: float(s['start_seconds']))
    cursor = -1.0
    for index, scene in enumerate(scenes):
        visuals = []
        for source in chosen[index]:
            a, b = float(source['start_seconds']), float(source['end_seconds'])
            if a < cursor - .01 or b - a < .06:
                if allow_rebuild and result.get('situation_plan_version', 0) < 3:
                    return lock_plan_visuals(rebuild_plan_in_source_order(plan, chunks), chunks,
                                             allow_rebuild=False, representative=True)
                raise ValueError('PLAN_SOURCE_ORDER_INVALID:' + scene['scene_id'])
            cursor = b
            visuals.append(dict(source, start_seconds=a, end_seconds=b))
        planned_facts = {f['fact_id']: f for f in scene.get('observed_facts') or []}
        bound_facts = [copy.deepcopy(planned_facts.get(fid, facts[fid])) for fid in scene['planned_fact_ids']]
        if result.get('situation_plan_version', 0) >= 4:
            available = {sid for shot in visuals for sid in shot.get('shot_ids') or [shot['shot_id']]}
            for fact in bound_facts:
                fact['shot_ids'] = [sid for sid in fact['shot_ids'] if sid in available]
                if not fact['shot_ids']:
                    raise ValueError('PLAN_FACT_WITHOUT_PROOF:' + fact['fact_id'])
        scene.update(observed_facts=bound_facts,
                     visual_sources=visuals, start_seconds=visuals[0]['start_seconds'],
                     end_seconds=visuals[-1]['end_seconds'], visual_selection_pending=False,
                     grounded_plan_locked=True, tts_min_words=6)
    # Keep each part-local source window nonnegative, including facts close to
    # the old coarse partition. No cut or source position is fabricated.
    parts = result['events_by_part']
    if len(parts) == 3 and all(parts):
        result['boundaries'] = [0.0] + [
            (parts[i][-1]['end_seconds'] + parts[i + 1][0]['start_seconds']) / 2
            for i in range(2)] + [float(result['boundaries'][-1])]
    result['grounded_plan_version'] = VERSION
    return result


def audit_narrations(payload, audit, checkpoint=None, save=None, visual_evidence=False,
                     source_identity=None, visual_source_offset=0.0, max_attempts=3,
                     max_workers=1, diagnostic=None):
    """Only the factual clauses already in a locked plan may be verbalized."""
    result = copy.deepcopy(payload)
    cache = copy.deepcopy(checkpoint or {})
    # Older checkpoints stored absent replies as factual failures. They must be
    # checked again, even when the sentence has not changed.
    cache = {key: row for key, row in cache.items()
             if isinstance(row, dict) and type(row.get('faithful')) is bool
             and row.get('reason') != 'missing_evidence_verdict'}
    scenes = [s for p in result['parts'] for s in p.get('scenes') or [] if s.get('grounded_plan_locked')]
    pending = []
    def emit(event, **data):
        if diagnostic:
            diagnostic(event, data)
    for scene in scenes:
        signature_data = [scene['tts'], scene.get('observed_facts')]
        if visual_evidence:
            from recap_visual_media import evidence_manifest
            try:
                visual_signature = evidence_manifest([scene], visual_source_offset)
                for row in visual_signature:
                    for key in ('source_start', 'source_end', 'pack_start', 'pack_end'):
                        row[key] = round(row[key], 2)
            except (ValueError, TypeError, KeyError):
                visual_signature = [visual_source_offset, scene.get('locked_visual_shots'), scene.get('visual_sources')]
            signature_data += ['video_audit_v1', source_identity, visual_signature]
        signature = hashlib.sha256(json.dumps(signature_data,
                                   sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        scene['_audit_signature'] = signature
        if signature not in cache:
            pending.append(scene)
    emit('AUDIT_CACHE_STATUS', total=len(scenes), cache_hit_ids=[s['scene_id'] for s in scenes
         if s['_audit_signature'] in cache], pending_ids=[s['scene_id'] for s in pending],
         cache_only=max_attempts == 0, max_attempts=max_attempts, max_workers=max_workers)
    def inspect_batch(batch):
        cards = [{'scene_id': s['scene_id'], 'tts': s['tts'], 'observed_facts': s['observed_facts'],
                  **({'locked_visual_shots': copy.deepcopy(s.get('locked_visual_shots') or []),
                      'visual_sources': copy.deepcopy(s.get('visual_sources') or [])}
                     if visual_evidence else {})} for s in batch]
        try:
            return batch, audit(cards)
        except (ValueError, TypeError) as exc:
            emit('AUDIT_CALLBACK_ERROR', expected_ids=[s['scene_id'] for s in batch],
                 error_type=type(exc).__name__, error=str(exc))
            return batch, {}  # Missing/invalid replies are not factual rejections.

    for attempt in range(max(0, int(max_attempts))):
        # Smaller follow-up batches make incomplete responses easier to finish.
        batch_size = 6 if visual_evidence else (24 if attempt == 0 else 8)
        batches = [pending[offset:offset + batch_size] for offset in range(0, len(pending), batch_size)]
        # Workers only perform independent requests. Verdicts/checkpoints are
        # merged on this thread so concurrent completions cannot lose results.
        def outcomes():
            if max_workers <= 1:
                yield from map(inspect_batch, batches)
            else:
                with ThreadPoolExecutor(max_workers=min(2, int(max_workers))) as executor:
                    yield from executor.map(inspect_batch, batches)

        for batch, response in outcomes():
            report, accepted = inspect_audit_response(response, [s['scene_id'] for s in batch])
            emit('AUDIT_BATCH_RESULT', attempt=attempt + 1, **report)
            for scene in batch:
                if scene['scene_id'] in accepted:
                    cache[scene['_audit_signature']] = accepted[scene['scene_id']]
            if save:
                save(copy.deepcopy(cache))
        pending = [s for s in pending if s['_audit_signature'] not in cache]
        if not pending:
            break
        emit('AUDIT_PENDING_AFTER_ATTEMPT', attempt=attempt + 1,
             pending_ids=[s['scene_id'] for s in pending], will_retry=attempt + 1 < max_attempts)
    for scene in scenes:
        signature = scene.pop('_audit_signature')
        verdict = cache.get(signature)
        scene['visual_alignment_errors'] = [error for error in scene.get('visual_alignment_errors') or []
                                            if not str(error).startswith(('unsupported_narration:', 'narration_audit_pending:'))]
        scene['grounding_audit_pending'] = verdict is None
        if verdict is None:
            scene.setdefault('visual_alignment_errors', []).append('narration_audit_pending:missing_evidence_verdict')
        elif not verdict['faithful']:
            scene.setdefault('visual_alignment_errors', []).append('unsupported_narration:' + verdict['reason'])
        scene['grounding_audit_passed'] = verdict is not None and verdict['faithful']
        scene['grounding_audit_method'] = 'video' if visual_evidence else 'text'
        if scene['grounding_audit_passed']:
            scene.pop('visual_fallback', None)
            if visual_evidence:
                for shot in scene.get('locked_visual_shots') or []:
                    shot['visual_match_tier'] = 'verified_video'
        else:
            emit('AUDIT_SCENE_UNRESOLVED', scene_id=scene['scene_id'], signature=signature,
                 status='pending' if verdict is None else 'rejected', tts=scene['tts'],
                 reason=('cache_miss_final_check' if max_attempts == 0 else 'no_valid_verdict')
                        if verdict is None else verdict['reason'],
                 observed_facts=scene.get('observed_facts'),
                 locked_visual_shots=scene.get('locked_visual_shots'),
                 visual_sources=scene.get('visual_sources'), source_offset=visual_source_offset)
    emit('AUDIT_SUMMARY', **summarize_audit_scenes(result))
    return result


def preserve_structure_only_narrations(previous, candidate, requirements):
    """Link-only repairs may change the EDL, never the spoken sentence."""
    prose_errors = {str(r.get('scene_id', '')).upper()
                    for key in ('invalid_tts_scenes', 'invalid_character_mentions')
                    for r in requirements.get(key) or []}
    structural = {str(r.get('scene_id', '')).upper()
                  for r in requirements.get('invalid_timestamp_scenes') or []
                  if r.get('reasons') and not any(str(reason).startswith(
                      ('unsupported_narration:', 'narration_audit_pending:'))
                      for reason in r['reasons'])} - prose_errors
    old = {str(s.get('scene_id', '')).upper(): s for p in previous.get('parts') or []
           for s in p.get('scenes') or []}
    result = copy.deepcopy(candidate)
    for part in result.get('parts') or []:
        for scene in part.get('scenes') or []:
            key = str(scene.get('scene_id', '')).upper()
            if key in structural and key in old:
                scene['tts'] = old[key]['tts']
    return result


def preserve_verified_scenes(previous, candidate):
    """A failed repair must not replace an already verified narration/shot pair."""
    result = copy.deepcopy(candidate)
    old = {s.get('scene_id'): s for p in previous.get('parts') or []
           for s in p.get('scenes') or [] if s.get('grounding_audit_passed')
           and not s.get('visual_alignment_errors')}
    for part in result.get('parts') or []:
        part['scenes'] = [copy.deepcopy(old[s['scene_id']])
                          if s.get('scene_id') in old and
                          (not s.get('grounding_audit_passed') or s.get('visual_alignment_errors'))
                          else s for s in part.get('scenes') or []]
    return result


def apply_local_visual_fallbacks(payload, log=None):
    """Use an observed illustration inside the sentence's allocated window.

    Similarity is a text heuristic over observed descriptions, not a new visual
    verification. Preserve the failed verdict explicitly for manual review.
    """
    from recap_pipeline import _time_seconds, _format_time, build_backend_visual_plan
    result = copy.deepcopy(payload)
    for part in result.get('parts') or []:
        scenes = part.get('scenes') or []
        for index, scene in enumerate(scenes):
            errors = list(scene.get('visual_alignment_errors') or [])
            if not errors and scene.get('grounding_audit_passed') and scene.get('locked_visual_shots'):
                continue
            sources = scene.get('visual_sources') or []
            if not scene.get('grounded_plan_locked') or not sources:
                continue  # Missing source geometry/identity is not a similarity failure.
            try:
                lo = min(float(s['start_seconds']) if 'start_seconds' in s else _time_seconds(s['start']) for s in sources)
                hi = max(float(s['end_seconds']) if 'end_seconds' in s else _time_seconds(s['end']) for s in sources)
                if index:
                    lo = max(lo, _time_seconds(scenes[index - 1]['end']))
                if index + 1 < len(scenes):
                    hi = min(hi, _time_seconds(scenes[index + 1]['start']))
            except (KeyError, TypeError, ValueError):
                continue
            def tokens(text):
                normalized = ''.join(c for c in unicodedata.normalize('NFKD', str(text)).casefold()
                                     if not unicodedata.combining(c))
                return set(re.findall(r'\w+', normalized))
            wanted = tokens(scene.get('tts', ''))
            candidates = []
            for source in sources:
                try:
                    a = float(source['start_seconds']) if 'start_seconds' in source else _time_seconds(source['start'])
                    b = float(source['end_seconds']) if 'end_seconds' in source else _time_seconds(source['end'])
                except (KeyError, TypeError, ValueError):
                    continue
                if (not all(math.isfinite(v) for v in (a, b, lo, hi))
                        or a < 0 or a < lo - .001 or b > hi + .001 or not .05 < b - a <= 15
                        or not (source.get('shot_id') or source.get('shot_ids'))):
                    continue
                described = tokens(source.get('description') or source.get('event') or source.get('action') or '')
                score = len(wanted & described) / max(1, len(wanted | described))
                candidates.append((score, a, dict(source, start_seconds=a, end_seconds=b)))
            if not candidates:
                continue
            score, _, chosen = max(candidates, key=lambda row: (row[0], -row[1]))
            source = copy.deepcopy(chosen)
            source.update(start=_format_time(source['start_seconds']), end=_format_time(source['end_seconds']))
            source['visual_match_tier'] = 'approximate_unverified'
            visual_plan = build_backend_visual_plan(scene.get('tts', ''), [source])
            if visual_plan.get('visual_alignment_errors') or not visual_plan.get('locked_visual_shots'):
                continue
            report = {
                'scene_id': scene.get('scene_id'), 'part': part.get('part'),
                'tts': scene.get('tts'), 'status': 'approximate_unverified',
                'reasons': errors or ['no_verified_visual'],
                'shot_ids': source.get('shot_ids') or [source.get('shot_id')],
                'start': source['start'], 'end': source['end'],
                'scope_seconds': [lo, hi],
                'description': source.get('description') or source.get('event') or source.get('action') or '',
                'similarity_score': round(score, 4), 'selection_method': 'observed_description_token_overlap',
            }
            scene.update(visual_plan)
            scene['visual_selection_pending'] = False
            scene['visual_fallback'] = report
            # Do not claim the substitute passed verification.
            scene['grounding_audit_passed'] = False
            if log:
                log(report)
    return result
