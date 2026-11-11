"""Normalize event-keyed ideas without guessing ownership or inventing facts."""

import copy
import json
import re


def clean_id(value):
    return value.strip().upper() if isinstance(value, str) else ''


def clean_focus(value):
    return re.sub(r'\s+', ' ', value).strip() if isinstance(value, str) else value


def inspect_event_ideas(payload, cluster):
    """Return a canonical draft plus exact field errors, retaining good ideas."""
    cid = cluster['cluster_id']
    issues = []
    supplied = payload.get('cluster_id') if isinstance(payload, dict) else None
    if supplied is not None and supplied != cid:
        issues.append({'code': 'cluster_id_mismatch', 'path': 'cluster_id', 'supplied': supplied})
    raw = payload.get('ideas_by_event') if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raw = {}
        issues.append({'code': 'ideas_map_required', 'path': 'ideas_by_event'})
    known = set(cluster['event_ids'])
    ideas = {}
    for key, values in raw.items():
        eid = clean_id(key)
        if not eid:
            issues.append({'code': 'invalid_event_id', 'path': f'ideas_by_event.{key}', 'event_id': str(key)})
            continue
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            normalized, seen = [], set()
            for focus in values:
                focus = clean_focus(focus)
                if isinstance(focus, str) and focus:
                    if focus.casefold() in seen:
                        continue
                    seen.add(focus.casefold())
                normalized.append(focus)
            values = normalized
        if eid in ideas:
            if ideas[eid] != values:
                ideas[eid] = None
                issues.append({'code': 'conflicting_normalized_id', 'path': f'ideas_by_event.{eid}', 'event_id': eid})
            continue
        ideas[eid] = values
    for eid in ideas:
        if eid not in known:
            issues.append({'code': 'unknown_event_id', 'path': f'ideas_by_event.{eid}', 'event_id': eid})
    for eid in cluster['event_ids']:
        values = ideas.get(eid)
        if eid not in ideas:
            issues.append({'code': 'missing_event_id', 'path': f'ideas_by_event.{eid}', 'event_id': eid})
        elif not isinstance(values, list) or not values:
            issues.append({'code': 'nonempty_ideas_list_required', 'path': f'ideas_by_event.{eid}', 'event_id': eid})
        else:
            for index, value in enumerate(values, 1):
                if not isinstance(value, str) or not value:
                    issues.append({'code': 'nonempty_text_required', 'path': f'ideas_by_event.{eid}[{index}]',
                                   'event_id': eid, 'focus_index': index})
    return {'cluster_id': cid, 'ideas_by_event': ideas}, issues


def event_ideas_to_detail(payload, cluster):
    draft, issues = inspect_event_ideas(payload, cluster)
    if issues:
        raise ValueError('DETAIL_EVENT_IDEAS: ' + json.dumps(issues, ensure_ascii=False))
    return {'cluster_id': cluster['cluster_id'], 'sentences': [
        {'source_event_ids': [eid], 'focus': focus}
        for eid in cluster['event_ids'] for focus in draft['ideas_by_event'][eid]]}


def event_ideas_repair_prompt(draft, cluster, issues, facts):
    known = set(cluster['event_ids'])
    targets = [eid for eid in cluster['event_ids']
               if any(i.get('event_id') == eid for i in issues)]
    unknown = [eid for eid in draft['ideas_by_event'] if eid not in known]
    read_only = {k: v for k, v in draft['ideas_by_event'].items() if k in known and k not in targets}
    return (
        f"SUA RIENG ID LOI CUA CUM {cluster['cluster_id']}. Khong tao lai cum hay doi ID dung.\n"
        'Chi tra ideas_by_event cho cac ID_CAN_SUA; moi ID co danh sach y ke rieng. '
        'Giu cac y hop le da co trong ID do; chi bo sung/thay o rong hoac sai kieu. '
        'Khong tu gan ID la cho du kien. Xac nhan loai cac khoa la bang discard_unknown_ids. '
        'Backend giu nguyen READ_ONLY; khong tra lai cac ID nay. Khong them su kien hay lap y.\n'
        'ID_CAN_SUA: ' + json.dumps(targets) + '\nKHOA_LA_CAN_LOAI: ' + json.dumps(unknown)
        + '\nLOI_CU_THE: ' + json.dumps(issues, ensure_ascii=False)
        + '\nBAN_NHAP: ' + json.dumps(draft, ensure_ascii=False)
        + '\nREAD_ONLY: ' + json.dumps(read_only, ensure_ascii=False)
        + '\nBANG_CHUNG: ' + json.dumps([f for f in facts if f['id'] in targets], ensure_ascii=False)
        + '\nChi tra JSON {"cluster_id":"' + cluster['cluster_id']
        + '","ideas_by_event":{"EV... ":["y ke da sua"]},"discard_unknown_ids":[]}.'
    )


def apply_event_idea_repairs(draft, patch, cluster, issues):
    if not isinstance(patch, dict) or patch.get('cluster_id', cluster['cluster_id']) != cluster['cluster_id']:
        raise ValueError('DETAIL_IDEA_PATCH_CLUSTER_ID')
    targets = {i['event_id'] for i in issues if i.get('event_id') in cluster['event_ids']}
    raw = patch.get('ideas_by_event')
    if not isinstance(raw, dict):
        raise ValueError('DETAIL_IDEA_PATCH_MAP_REQUIRED')
    keys = [clean_id(k) for k in raw]
    if len(set(keys)) != len(keys) or not targets.issubset(keys):
        raise ValueError('DETAIL_IDEA_PATCH_IDS: requested=' + json.dumps(sorted(targets))
                         + '; supplied=' + json.dumps(keys))
    normalized_patch, _ = inspect_event_ideas({'ideas_by_event': raw}, cluster)
    for eid in set(keys) - targets:
        if (eid not in cluster['event_ids'] or eid not in draft['ideas_by_event']
                or normalized_patch['ideas_by_event'][eid] != draft['ideas_by_event'][eid]):
            raise ValueError('DETAIL_IDEA_PATCH_IDS: changed read_only ID ' + eid)
    unknown = set(draft['ideas_by_event']) - set(cluster['event_ids'])
    removals = patch.get('discard_unknown_ids', [])
    if not isinstance(removals, list) or {clean_id(k) for k in removals} != unknown:
        raise ValueError('DETAIL_IDEA_PATCH_UNKNOWN_IDS: ' + json.dumps(sorted(unknown)))
    result = copy.deepcopy(draft)
    for eid in unknown:
        result['ideas_by_event'].pop(eid)
    for key, values in raw.items():
        eid = clean_id(key)
        if eid not in targets:
            continue  # Identical read-only echoes do not rewrite locked content.
        # A structural repair must not erase already valid ideas in this ID.
        existing = draft['ideas_by_event'].get(eid)
        good = [clean_focus(v) for v in existing if isinstance(v, str) and clean_focus(v)] if isinstance(existing, list) else []
        replacement = values if isinstance(values, list) else [values]
        if not {v.casefold() for v in good}.issubset(
                {clean_focus(v).casefold() for v in replacement if isinstance(v, str)}):
            raise ValueError('DETAIL_IDEA_PATCH_CHANGED_VALID_FOCUS:' + eid)
        result['ideas_by_event'][eid] = values
    normalized, _ = inspect_event_ideas(result, cluster)
    return normalized


def legacy_event_idea_draft(payload, cluster):
    """Preserve unambiguous single-event groups for a targeted schema repair."""
    ideas = {}
    for group in payload.get('event_plans') or []:
        if not isinstance(group, dict):
            continue
        ids = group.get('source_event_ids')
        ids = [ids] if isinstance(ids, str) else ids
        if not isinstance(ids, list):
            continue
        ids = list(dict.fromkeys(clean_id(key) for key in ids))
        if len(ids) != 1:
            continue  # Multiple IDs + multiple ideas cannot be assigned by position.
        values = group.get('focuses', group.get('focus'))
        values = [values] if isinstance(values, str) else values
        if isinstance(values, list):
            ideas.setdefault(ids[0], []).extend(values)
    return inspect_event_ideas({'cluster_id': cluster['cluster_id'], 'ideas_by_event': ideas}, cluster)
