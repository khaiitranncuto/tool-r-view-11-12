"""Two-stage text-only planning with durable per-cluster progress."""

import copy
import hashlib
import json
import re

PLANNING_VERSION = 2
WORKFLOW_VERSION = 8


def repair_spine_ownership(candidate, events, duration):
    """Reconcile grouping preferences against the backend's ordered event ledger."""
    candidate = copy.deepcopy(candidate) if isinstance(candidate, dict) else {}
    known = {e['evidence_id']: e for e in events}
    aliases = {}
    for key in known:
        match = re.fullmatch(r'EV([0-9]+)_([0-9]+)', key)
        if match:
            aliases.setdefault(tuple(map(int, match.groups())), []).append(key)
    owners, originals, diagnostics = {}, {}, []
    raw = candidate.get('selected_event_clusters')
    for index, cluster in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(cluster, dict):
            continue
        originals[index] = cluster
        ids = cluster.get('event_ids')
        for value in ids if isinstance(ids, list) else []:
            key = str(value).strip().upper()
            if key not in known:
                match = re.fullmatch(r'EV([0-9]+)_([0-9]+)', key)
                matches = aliases.get(tuple(map(int, match.groups())), []) if match else []
                key = matches[0] if len(matches) == 1 else ''
            if not key or key in owners:
                diagnostics.append({'cluster_id': cluster.get('cluster_id'), 'event_id': value,
                                    'reason': 'duplicate' if key else 'unknown'})
                continue
            owners[key] = index
    groups, current, previous_owner = [], [], None
    for event in events:
        key = event['evidence_id']
        owner = owners.get(key, -1)
        if key not in owners:
            diagnostics.append({'event_id': key, 'reason': 'restored_missing_event'})
        if current and owner != previous_owner:
            groups.append((previous_owner, current))
            current = []
        current.append(key)
        previous_owner = owner
    if current:
        groups.append((previous_owner, current))
    clusters, used = [], set()
    for index, (owner, ids) in enumerate(groups):
        original = originals.get(owner, {})
        cid = str(original.get('cluster_id') or f'BACKEND_{index+1:04d}')
        while cid in used:
            cid += '_R'
        used.add(cid)
        count = original.get('sentence_count')
        clusters.append({'cluster_id': cid, 'event_ids': ids,
            'summary': str(original.get('summary') or ' → '.join(known[k].get('event', k) for k in ids)),
            'required_context_ids': [k for k in original.get('required_context_ids') or [] if k in known],
            'sentence_count': count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else len(ids),
            'importance': 'main'})
    candidate['selected_event_clusters'] = clusters
    try:
        first, second = float(candidate['boundary_1_seconds']), float(candidate['boundary_2_seconds'])
        if not 0 < first < second < duration:
            raise ValueError('boundaries')
    except (KeyError, ValueError, TypeError):
        candidate.update(boundary_1_seconds=duration/3, boundary_2_seconds=duration*2/3)
    if not isinstance(candidate.get('parts'), list) or len(candidate['parts']) != 3:
        candidate['parts'] = [{'part': p, 'summary': ''} for p in range(1, 4)]
    return candidate, diagnostics


def refine_story_clusters(plan, events, max_events=4, max_capacity=10, max_span=180):
    """Bound writer context to a situation, without deleting/reordering facts."""
    from recap_pipeline import event_narration_capacity
    result = copy.deepcopy(plan)
    known = {str(row['evidence_id']).upper(): row for row in events}
    clusters = []
    reserved = {c['cluster_id'] for c in plan['selected_event_clusters']}
    for cluster in plan['selected_event_clusters']:
        pieces, current, capacity = [], [], 0
        for key in cluster['event_ids']:
            row = known[key]
            amount = event_narration_capacity(row)
            span = float(row.get('end_seconds', 0)) - float(known[current[0]].get('start_seconds', 0)) if current else 0
            if current and (len(current) >= max_events or capacity + amount > max_capacity or span > max_span):
                pieces.append(current)
                current, capacity = [], 0
            current.append(key)
            capacity += amount
        if current:
            pieces.append(current)
        if len(pieces) == 1:
            clusters.append(cluster)
            continue
        remaining = max(len(pieces), cluster['sentence_count'])
        for index, ids in enumerate(pieces):
            cid = f"{cluster['cluster_id']}_P{index + 1:02d}"
            while cid in reserved:
                cid += '_'
            reserved.add(cid)
            item = copy.deepcopy(cluster)
            count = max(1, round(remaining / (len(pieces) - index)))
            remaining -= count
            item.update(cluster_id=cid, event_ids=ids, sentence_count=count,
                        summary=' -> '.join(str(known[key].get('action') or known[key].get('event') or key)[:120]
                                            for key in ids))
            clusters.append(item)
    result['selected_event_clusters'] = clusters
    return result


def validate_event_roles(payload, events):
    """Optional narrative roles; source chronology and ownership remain mandatory."""
    known = {str(row['evidence_id']).upper(): row for row in events}
    raw = payload.get('editorial_anchors')
    if not isinstance(raw, dict):
        raise ValueError('SPINE_ROLES_MISSING')
    anchors = {}
    for role in ('opening', 'inciting_incident', 'protagonist_goal', 'climax', 'resolution', 'ending'):
        values = raw.get(role, [])
        if not isinstance(values, list) or any(not isinstance(key, str) or key not in known for key in values):
            raise ValueError(f'SPINE_ROLE_INVALID:{role}')
        anchors[role] = list(dict.fromkeys(values))
    # One source event can contain both the decisive action and its outcome.
    # Reject reversed distinct events, rather than comparing coarse event ends.
    for left, right in (('opening', 'climax'), ('climax', 'resolution'),
                        ('resolution', 'ending'), ('climax', 'ending')):
        if anchors[left] and anchors[right]:
            left_time = max(float(known[key].get('start_seconds', 0)) for key in anchors[left])
            right_time = min(float(known[key].get('start_seconds', 0)) for key in anchors[right])
            if right_time < left_time:
                raise ValueError(f'SPINE_ROLE_ORDER:{left}->{right}')
    importance = payload.get('event_importance') or {}
    if not isinstance(importance, dict) or any(key not in known or value not in {'critical', 'main', 'support'}
                                               for key, value in importance.items()):
        raise ValueError('SPINE_ROLE_IMPORTANCE_INVALID')
    return {'editorial_contract_version': 2, 'editorial_anchors': anchors,
            'story_contract': copy.deepcopy(payload.get('story_contract') or {}),
            'turning_points': [key for key in payload.get('turning_points', []) if key in known],
            'event_importance': copy.deepcopy(importance)}


def validate_backbone(payload, events):
    """Validate only global story anchors and importance, without clustering."""
    known = {str(row["evidence_id"]).upper(): row for row in events}
    anchors = payload.get("editorial_anchors")
    roles = ("opening", "inciting_incident", "protagonist_goal", "climax", "resolution", "ending")
    if not isinstance(anchors, dict):
        raise ValueError("SPINE_BACKBONE_ANCHORS_MISSING")
    for role in roles:
        values = anchors.get(role)
        if not isinstance(values, list) or not values or any(key not in known for key in values):
            raise ValueError(f"SPINE_BACKBONE_ANCHOR_INVALID:{role}")
    first = lambda role: min(float(known[key].get("start_seconds", 0)) for key in anchors[role])
    last = lambda role: max(float(known[key].get("end_seconds", 0)) for key in anchors[role])
    if first("opening") >= first("climax") or first("resolution") < last("climax") - 2 or first("ending") < first("resolution"):
        raise ValueError("SPINE_BACKBONE_ORDER: check climax, resolution and ending IDs")
    importance = payload.get("event_importance") or {}
    if not isinstance(importance, dict) or any(key not in known or value not in {"critical", "main", "support"}
                                               for key, value in importance.items()):
        raise ValueError("SPINE_BACKBONE_IMPORTANCE_INVALID")
    return {"editorial_contract_version": 1, "editorial_anchors": copy.deepcopy(anchors),
            "story_contract": copy.deepcopy(payload.get("story_contract") or {}),
            "turning_points": [key for key in payload.get("turning_points", []) if key in known],
            "event_importance": copy.deepcopy(importance)}


class SpineBudgetError(ValueError):
    """A valid story structure whose numeric sentence allocation needs repair."""

    def __init__(self, report):
        self.report = report
        exceeded = [row for row in report["clusters"] if row["excess"]]
        if exceeded:
            row = exceeded[0]
            message = (f"SPINE_CAPACITY:{row['cluster_id']}: allocated={row['allocated']} "
                       f"capacity={row['capacity']} excess={row['excess']}")
        else:
            message = f"SPINE_SENTENCE_TOTAL:{report['allocated_total']}/{report['target_count']}"
        super().__init__(message)


def spine_budget_report(plan, events, target_count):
    from recap_pipeline import event_narration_capacity, event_narration_foci
    known = {str(row.get("evidence_id", "")).strip().upper(): row for row in events}
    clusters = []
    for cluster in plan["selected_event_clusters"]:
        facts = [{"event_id": key, "capacity": event_narration_capacity(known[key]),
                  "facts": event_narration_foci(known[key])}
                 for key in cluster["event_ids"]]
        estimated_facts = sum(row["capacity"] for row in facts)
        exact_target = plan.get('sentence_budget_mode') == 'exact_target'
        # Fact estimates guide allocation; they are not a four-sentence ceiling.
        # The planner must substantiate the assigned quota with distinct ideas.
        capacity = target_count if exact_target and not plan.get('grounding_required') else estimated_facts
        count = cluster["sentence_count"]
        clusters.append({
            "cluster_id": cluster["cluster_id"], "summary": cluster.get("summary", ""),
            "importance": cluster.get("importance", "main"),
            "allocated": count, "capacity": capacity,
            "excess": max(0, count - capacity), "spare": max(0, capacity - count),
            "event_facts": facts,
            "estimated_facts": estimated_facts,
        })
    exact_target = plan.get('sentence_budget_mode') == 'exact_target'
    return {"target_count": target_count, "allocated_total": sum(row["allocated"] for row in clusters),
            "capacity_total": target_count if exact_target and not plan.get('grounding_required') else sum(row["capacity"] for row in clusters),
            "estimated_facts_total": sum(row['estimated_facts'] for row in clusters),
            "capacity_is_limit": not exact_target or bool(plan.get('grounding_required')), "clusters": clusters}


def balance_spine_budget(plan, events, target_count, log=None, fixed_counts=None):
    """Treat cluster counts as preferences; backend owns exact global arithmetic."""
    balanced = copy.deepcopy(plan)
    fixed_counts = fixed_counts or {}
    for cluster in balanced.get("selected_event_clusters") or []:
        if cluster.get("cluster_id") in fixed_counts:
            cluster["sentence_count"] = fixed_counts[cluster["cluster_id"]]
    try:
        return validate_spine(balanced, events, target_count)
    except SpineBudgetError as exc:
        report = exc.report
    fixed_indexes = {i for i, row in enumerate(report["clusters"]) if row["cluster_id"] in fixed_counts}
    supported_total = sum(row["allocated"] if i in fixed_indexes else row["capacity"]
                          for i, row in enumerate(report["clusters"]))
    minimum_total = sum(row["allocated"] if i in fixed_indexes else 1
                        for i, row in enumerate(report["clusters"]))
    if (supported_total < target_count or target_count < minimum_total
            or any(report["clusters"][i]["excess"] for i in fixed_indexes)):
        raise ValueError(
            f"SPINE_CAPACITY_TOTAL: backend uoc luong {supported_total}/{target_count} cau; "
            "can kiem tra lai fact va suc chua, khong tu giam muc tieu hay chen y lap."
        )
    rows = report["clusters"]
    original = [row["allocated"] for row in rows]
    counts = [min(row["capacity"], max(1, row["allocated"])) for row in rows]
    priorities = {"critical": 3, "main": 2, "support": 1}
    while sum(counts) < target_count:
        candidates = [i for i, row in enumerate(rows) if i not in fixed_indexes and counts[i] < row["capacity"]]
        chosen = max(candidates, key=lambda i: (
            (original[i] - counts[i]) / max(1, original[i]),
            priorities.get(rows[i]["importance"], 2),
            (rows[i]["capacity"] - counts[i]) / max(1, rows[i]["capacity"]), -i,
        ))
        counts[chosen] += 1
    while sum(counts) > target_count:
        candidates = [i for i in range(len(rows)) if i not in fixed_indexes and counts[i] > 1]
        chosen = max(candidates, key=lambda i: (
            -priorities.get(rows[i]["importance"], 2),
            counts[i] / max(1, original[i]), counts[i], -i,
        ))
        counts[chosen] -= 1
    for cluster, row, count in zip(balanced["selected_event_clusters"], rows, counts):
        cluster["sentence_count"] = count
        if log and balanced.get('sentence_budget_mode') == 'exact_target':
            log(f"Cum {row['cluster_id']}: khoa {count} o cau; uoc luong fact {row['estimated_facts']} chi la goi y, khong phai tran.")
        elif log:
            log(f"Cum {row['cluster_id']}: Gemini de xuat {row['allocated']}, "
                f"backend can bang {count}, suc chua uoc luong {row['capacity']}; giu nguyen su kien.")
    if log:
        log(f"Backend da can bang tong {sum(counts)}/{target_count} cau; khong goi Gemini tinh lai phan bo.")
    return validate_spine(balanced, events, target_count)


def evidence_rows(chunks):
    return sorted([copy.deepcopy(row) for chunk in chunks for row in chunk.get("events", [])
                   if isinstance(row, dict)], key=lambda row: float(row.get("start_seconds", 0)))


def validate_spine(plan, events, target_count):
    """Lock ownership and quotas before expanding any cluster into sentences."""
    known = {str(row.get("evidence_id", "")).upper(): row for row in events}
    order = {key: index for index, key in enumerate(known)}
    clusters = plan.get("selected_event_clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("SPINE_CLUSTERS_MISSING")
    seen_ids, seen_clusters, previous = set(), set(), -1
    total = 0
    for cluster in clusters:
        if not isinstance(cluster, dict):
            raise ValueError("SPINE_CLUSTER_INVALID: expected object")
        cid = str(cluster.get("cluster_id", "")).strip()
        ids = [str(value).strip().upper() for value in cluster.get("event_ids") or []]
        count = cluster.get("sentence_count")
        if (not cid or cid in seen_clusters or not ids or isinstance(count, bool)
                or not isinstance(count, int) or count < 1):
            raise ValueError(f"SPINE_CLUSTER_INVALID:{cid}")
        if len(set(ids)) != len(ids) or any(key not in known or key in seen_ids for key in ids):
            raise ValueError(f"SPINE_EVENT_IDS_INVALID:{cid}")
        positions = [order[key] for key in ids]
        if positions != sorted(positions) or positions[0] <= previous:
            raise ValueError(f"SPINE_CLUSTER_ORDER:{cid}")
        context = [str(value).strip().upper() for value in cluster.get("required_context_ids") or []]
        if any(key not in known for key in context):
            raise ValueError(f"SPINE_CONTEXT_UNKNOWN:{cid}")
        cluster["event_ids"] = ids
        previous = positions[-1]
        seen_ids.update(ids)
        seen_clusters.add(cid)
        total += count
    if seen_ids != set(known):
        raise ValueError("SPINE_EVENTS_MISSING:" + ",".join(sorted(set(known) - seen_ids)))
    report = spine_budget_report(plan, events, target_count)
    if total != target_count or any(row["excess"] for row in report["clusters"]):
        raise SpineBudgetError(report)
    return plan


def validate_cluster_detail(detail, cluster, allow_flexible_count=False):
    """Only the requested cluster can be repaired; all its events stay covered."""
    cid = cluster["cluster_id"]
    if not isinstance(detail, dict) or detail.get("cluster_id") != cid:
        raise ValueError(f"DETAIL_CLUSTER_ID:{cid}")
    rows = detail.get("sentences")
    if (not isinstance(rows, list) or not rows
            or (not allow_flexible_count and len(rows) != cluster["sentence_count"])):
        actual = len(rows) if isinstance(rows, list) else 0
        raise ValueError(f"DETAIL_SENTENCE_COUNT:{cid}:expected={cluster['sentence_count']}:actual={actual}")
    positions = {key: index for index, key in enumerate(cluster["event_ids"])}
    covered, foci, previous = set(), set(), -1
    focus_owners = []
    groups_by_id = {}
    for row_index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"DETAIL_EVENT_OR_FOCUS:{cid}:row={row_index}:expected object")
        if not isinstance(row.get('source_event_ids'), list):
            raise ValueError(f'DETAIL_EVENT_IDS_REQUIRED:{cid}:row={row_index}')
        ids = [str(value).strip().upper() for value in row.get("source_event_ids") or []]
        if not isinstance(row.get('focus'), str):
            raise ValueError(f'DETAIL_FOCUS_TYPE:{cid}:row={row_index}:expected text')
        focus = re.sub(r"\s+", " ", row['focus']).strip()
        if not ids or len(set(ids)) != len(ids) or any(key not in positions for key in ids) or not focus:
            raise ValueError(f"DETAIL_EVENT_OR_FOCUS:{cid}:row={row_index}:ids={ids}:focus_empty={not bool(focus)}")
        indexes = [positions[key] for key in ids]
        if indexes != sorted(indexes) or indexes[0] < previous:
            raise ValueError(f"DETAIL_EVENT_ORDER:{cid}")
        # Repeated single events can own several facts; a merged group cannot
        # also be reused as another sentence with overlapping event ownership.
        for key in ids:
            group = tuple(ids)
            if key in groups_by_id and (len(group) > 1 or groups_by_id[key] != group):
                raise ValueError(f"DETAIL_EVENT_OVERLAP:{cid}:{key}")
            groups_by_id[key] = group
        focus_key = tuple(re.findall(r"\w+", focus.casefold()))
        if focus_key in foci:
            raise ValueError(f"DETAIL_REPEATED_FOCUS:{cid}")
        tokens = set(focus_key)
        for old_ids, old_tokens in focus_owners:
            smaller = min(len(tokens), len(old_tokens))
            if (tuple(ids) == old_ids and smaller >= 5
                    and len(tokens & old_tokens) / smaller >= .85
                    and len(tokens ^ old_tokens) <= 2):
                raise ValueError(f"DETAIL_REPEATED_FOCUS:{cid}: chi doi tu dem, khong co fact moi")
        foci.add(focus_key)
        focus_owners.append((tuple(ids), tokens))
        covered.update(ids)
        previous = indexes[-1]
        row["source_event_ids"], row["focus"] = ids, focus
    if covered != set(positions):
        raise ValueError(f"DETAIL_EVENT_MISSING:{cid}")
    return detail


def expand_event_plans(detail, cluster):
    """Flatten new per-event idea groups in backend-owned source order.

    Flat drafts/patches keep their existing order and still require validation.
    Only the explicitly grouped writer format grants order to the backend.
    """
    from event_ideas import event_ideas_to_detail, clean_id, clean_focus
    if isinstance(detail, dict) and 'ideas_by_event' in detail:
        return event_ideas_to_detail(detail, cluster)
    if not isinstance(detail, dict) or "event_plans" not in detail:
        if isinstance(detail, dict) and isinstance(detail.get('sentences'), list):
            detail = copy.deepcopy(detail)
            detail.setdefault('cluster_id', cluster['cluster_id'])
            for row in detail['sentences']:
                if not isinstance(row, dict):
                    continue
                ids = row.get('source_event_ids')
                ids = [ids] if isinstance(ids, str) else ids
                if isinstance(ids, list):
                    row['source_event_ids'] = list(dict.fromkeys(clean_id(key) for key in ids))
                if isinstance(row.get('focus'), str):
                    row['focus'] = clean_focus(row['focus'])
        return detail
    if detail.get("cluster_id") != cluster["cluster_id"] or not isinstance(detail["event_plans"], list):
        raise ValueError("DETAIL_EVENT_PLANS_SCHEMA")
    if not detail['event_plans']:
        raise ValueError('DETAIL_EVENT_PLANS_EMPTY:' + cluster['cluster_id'])
    order = {key: index for index, key in enumerate(cluster["event_ids"])}
    groups, owners = {}, {}
    for group_index, group in enumerate(detail["event_plans"], 1):
        if not isinstance(group, dict):
            raise ValueError("DETAIL_EVENT_PLANS_SCHEMA")
        ids, focuses = group.get("source_event_ids"), group.get("focuses")
        ids = [ids] if isinstance(ids, str) else ids
        if not isinstance(ids, list) or not ids:
            raise ValueError(f'DETAIL_EVENT_IDS_REQUIRED:group={group_index}')
        ids = list(dict.fromkeys(clean_id(key) for key in ids))
        unknown = [key for key in ids if key not in order]
        if unknown:
            raise ValueError(f'DETAIL_EVENT_UNKNOWN_ID:group={group_index}:ids={unknown}')
        focuses = [focuses] if isinstance(focuses, str) else focuses
        if not isinstance(focuses, list) or not focuses:
            raise ValueError(f'DETAIL_EVENT_FOCUSES_REQUIRED:group={group_index}:ids={ids}')
        focuses = [clean_focus(focus) for focus in focuses]
        for focus_index, focus in enumerate(focuses, 1):
            if not isinstance(focus, str) or not focus:
                raise ValueError(f'DETAIL_EVENT_FOCUS_INVALID:group={group_index}:focus={focus_index}:ids={ids}')
        if len(ids) > 1 and len(focuses) > 1:
            raise ValueError(f'DETAIL_EVENT_AMBIGUOUS_MAPPING:group={group_index}:ids={ids}; use ideas_by_event')
        ids = sorted(ids, key=order.get)
        key = tuple(ids)
        if any(eid in owners and (owners[eid] != key or len(ids) > 1) for eid in ids):
            raise ValueError(f'DETAIL_EVENT_OVERLAP:group={group_index}:ids={ids}')
        for eid in ids:
            owners[eid] = key
        rows = groups.setdefault(key, [])
        existing = {row['focus'].casefold() for row in rows}
        for focus in focuses:
            if focus.casefold() not in existing:
                rows.append({'source_event_ids': ids, 'focus': focus})
                existing.add(focus.casefold())
    return {"cluster_id": detail["cluster_id"], "sentences": [row for key in sorted(groups, key=lambda ids: order[ids[0]]) for row in groups[key]]}


def materialize_sentence_plan(plan, events, require_verified_visuals=False, defer_visual_selection=False):
    """Resolve AI sentence ownership into backend IDs, ranges and visual data."""
    from recap_pipeline import expand_evidence_to_visual_slots
    known = {str(row.get("evidence_id", "")).upper(): row for row in events}
    results = []
    for cluster in plan["selected_event_clusters"]:
        detail = validate_cluster_detail(copy.deepcopy(plan["cluster_details"][cluster["cluster_id"]]), cluster)
        rows = detail["sentences"]
        cursor = 0
        while cursor < len(rows):
            ids = rows[cursor]["source_event_ids"]
            end = cursor + 1
            if len(ids) == 1:
                while end < len(rows) and rows[end]["source_event_ids"] == ids:
                    end += 1
            if defer_visual_selection:
                # Narration slots own facts, not guessed slices of a source
                # window. Exact pictures are selected AFTER prose exists.
                owned = [known[key] for key in ids]
                slots = []
                for position in range(end - cursor):
                    slot = copy.deepcopy(owned[0])
                    slot.update(
                        source_evidence_id=ids[0],
                        start_seconds=min(float(e['start_seconds']) for e in owned),
                        end_seconds=max(float(e['end_seconds']) for e in owned),
                        visual_sources=[], visual_selection_pending=True,
                        narration_slot_index=position + 1,
                        narration_slot_count=end - cursor,
                        fact_id=f"{ids[0]}_NARRATION_{cursor + position + 1}",
                        merged_evidence_ids=list(ids),
                        supporting_facts=[copy.deepcopy(e) for e in owned],
                        planned_fact_ids=list(rows[cursor + position].get('fact_ids') or []),
                    )
                    slots.append(slot)
            else:
                slots = expand_evidence_to_visual_slots(
                    [known[key] for key in ids], end - cursor,
                    require_verified_visuals=require_verified_visuals,
                )
            if len(slots) != end - cursor:
                raise ValueError(f"DETAIL_CAPACITY:{cluster['cluster_id']}")
            for slot, row in zip(slots, rows[cursor:end]):
                slot["narration_focus"] = row["focus"]
                slot["planning_cluster_id"] = cluster["cluster_id"]
                slot["selection_cluster_id"] = cluster["cluster_id"]
                slot["planning_source_event_ids"] = row["source_event_ids"]
                results.append(slot)
            cursor = end
    return results


def run_two_stage_planning(chunks, duration, target_words, base_prompt, generate,
                           parse, normalize, checkpoint=None, save=None, log=None,
                           global_outline=None, defer_visual_selection=False,
                           lightweight=False):
    """Chronological grouping, optional roles, budget, then grounded expansions.

    generate(prompt, label) returns (text, model). Capacity retries and model
    fallback remain the caller's responsibility. Transport errors preserve all
    completed work; validation repairs are bounded and local.
    """
    from recap_pipeline import required_tts_scene_count, event_narration_capacity, event_narration_foci
    rows = evidence_rows(chunks)
    if log:
        log(f"Buoc 1/5: da sap {len(rows)} su kien theo thoi gian; giu nguyen bang chung va quan he nguon.")
    target_count = required_tts_scene_count(target_words)
    exact_target = bool(defer_visual_selection)
    grounding_required = bool(rows and all(row.get('grounded_evidence_version') for row in rows))
    fingerprint = hashlib.sha256(json.dumps({"version": WORKFLOW_VERSION,
        "events": rows, "duration": duration, "words": target_words,
        "prompt": base_prompt, "sentence_first": defer_visual_selection,
        "lightweight": lightweight}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    state = copy.deepcopy(checkpoint or {})
    if state.get("fingerprint") != fingerprint:
        state = {"fingerprint": fingerprint, "version": WORKFLOW_VERSION, "details": {}}

    def persist():
        if save:
            save(copy.deepcopy(state))

    if lightweight and not state.get('spine') and not state.get('grouping'):
        # The backend already owns chronology, evidence, quotas and IDs. One
        # optional short editorial pass may change emphasis, never geometry.
        grouping, _ = repair_spine_ownership({}, rows, duration)
        grouping = refine_story_clusters(grouping, rows)
        editorial = {}
        if not state.get('brief_attempted'):
            state['brief_attempted'] = True
            persist()
            ledger = [{'id': r['evidence_id'], 'event': str(r.get('event', ''))[:240],
                       'depends_on': (r.get('depends_on') or [])[:8]} for r in rows]
            prompt = (
                'BIEN TAP NGAN: thu tu, shot, quota va cum do backend quan ly; KHONG lap lai ke hoach. '
                'Chi chon diem nhan va noi mach ba phan theo bang chung. '
                'Khong bia cao trao/ket thuc. Khong tra lai danh sach su kien hay timestamp. '
                'Tra JSON {"event_importance":{"EV...":"critical/main/support"},'
                '"parts":[{"part":1,"summary":"","handoff_to_next":""},'
                '{"part":2,"summary":"","handoff_to_next":""},'
                '{"part":3,"summary":"","handoff_to_next":""}]}.'
                '\nOUTLINE (context): ' + json.dumps(global_outline or {}, ensure_ascii=False)[:8000]
                + '\nSU KIEN: ' + json.dumps(ledger, ensure_ascii=False, separators=(',', ':')))
            try:
                text, model = generate(prompt, 'Bien tap ngan toan phim (tuy chon, mot luot)')
                editorial = parse(text)
                if not isinstance(editorial, dict):
                    raise ValueError('EDITORIAL_BRIEF_OBJECT_REQUIRED')
                state.update(brief_raw=text, model=model)
            except Exception as exc:
                editorial = {}
                state['brief_error'] = str(exc)
                if log:
                    log('Bien tap ngan gian doan; dung outline va thu tu nguon, tiep tuc: ' + str(exc))
        editorial['editorial_anchors'] = {k: [] for k in (
            'opening', 'inciting_incident', 'protagonist_goal', 'climax', 'resolution', 'ending')}
        try:
            state['backbone'] = validate_event_roles(editorial, rows)
        except (ValueError, TypeError, KeyError):
            state['backbone'] = validate_event_roles({'editorial_anchors': {}}, rows)
        parts = editorial.get('parts')
        if (isinstance(parts, list) and len(parts) == 3
                and all(isinstance(p, dict) and p.get('part') == i for i, p in enumerate(parts, 1))):
            grouping['parts'] = [{k: p[k] for k in ('part', 'summary', 'handoff_to_next') if k in p}
                                 for p in parts]
        else:
            # Existing outline is context only. Do not manufacture source events.
            grouping['parts'] = [{'part': i, 'summary': ''} for i in range(1, 4)]
        state['spine_draft'] = grouping
        persist()
        if log:
            log('Ke hoach nhe: backend gom cum theo thoi gian, giu fact+shot; bo hai luot AI chia cum/gan vai tro.')

    def assign_roles(grouping):
        if state.get("backbone"):
            return
        if lightweight:
            state['backbone'] = validate_event_roles({'editorial_anchors': {}}, rows)
            persist()
            return
        prompt = (
            "BUOC 3/5: Gan vai tro SAU KHI da gom cum theo mach su kien. Chi doc bang chung text.\n"
            "Khong doi cum hay them su kien. Danh gia importance critical/main/support.\n"
            "Moi vai tro co the de [] neu chua du bang chung; khong ep phim co ket thuc.\n"
            "Mot event co the mang nhieu vai tro neu chua ca hanh dong quyet dinh va ket qua.\n"
            "Khong tu lay event cuoi lam ending. Khong suy dien resolution khi xung dot dang do.\n"
            "Cac vai tro khac event phai theo thu tu start_seconds; neu mau thuan, xem lai ca climax.\n"
            "SU KIEN: " + json.dumps(rows, ensure_ascii=False) +
            "\nCUM DA KHOA: " + json.dumps(grouping["selected_event_clusters"], ensure_ascii=False) +
            '\nTra JSON {"editorial_anchors":{"opening":[],"inciting_incident":[],'
            '"protagonist_goal":[],"climax":[],"resolution":[],"ending":[]},'
            '"event_importance":{},"turning_points":[],'
            '"story_contract":{"premise":"","protagonist_goal":"","central_conflict":"","ending_meaning":""}}'
        )
        error = ""
        for attempt in range(3):
            text, model = generate(prompt + error, "Gemini lap ke hoach 3/5: gan vai tro sau gom cum")
            role_record = {'text': text, 'model': model, 'attempt': attempt + 1}
            state.setdefault('role_raw_responses', []).append(role_record)
            persist()
            try:
                role_payload = parse(text)
                if not isinstance(role_payload, dict):
                    raise ValueError('SPINE_ROLES_OBJECT_REQUIRED')
                state["backbone"] = validate_event_roles(role_payload, rows)
                state["model"] = model
                persist()
                if log:
                    log("Da gan vai tro tu bang chung; vai tro chua xac minh de rong, khong tu tao ket thuc.")
                return
            except (ValueError, TypeError, KeyError) as exc:
                role_record['error'] = str(exc)
                persist()
                error = "\nCHI SUA VAI TRO, GIU NGUYEN CUM: " + str(exc)
                if attempt == 2:
                    if grounding_required:
                        state['backbone'] = {'editorial_contract_version': 2,
                            'editorial_anchors': {k: [] for k in ('opening', 'inciting_incident',
                                'protagonist_goal', 'climax', 'resolution', 'ending')},
                            'story_contract': {}, 'turning_points': [], 'event_importance': {}}
                        persist()
                        if log:
                            log('Vai tro chua hop le: giu rong nhan chua xac minh va tiep tuc tu su kien/shot da co.')
                        return
                    raise ValueError("SPINE_ROLES_INVALID: " + str(exc)) from exc

    if not state.get("spine"):
        capacities = [{"id": row["evidence_id"], "time": [row.get("start_seconds"), row.get("end_seconds")],
                       "event": row.get("event"), "cause": row.get("cause"),
                       "action": row.get("action"), "change": row.get("change"),
                       "result": row.get("consequence") or row.get("result"),
                       "state_before": row.get("state_before"), "state_after": row.get("state_after"),
                       "depends_on": row.get("depends_on") or [],
                       "temporal_relation": row.get("temporal_relation"),
                       "role": row.get("story_role") or row.get("narrative_function"),
                       "estimated_facts": event_narration_capacity(row)} for row in rows]
        # Stage two is grouping only. Do not resend the old instruction bundle
        # asking this request to choose anchors/importance a second time.
        header = (
            "BUOC 1-2/5: Dung mach dien bien theo thoi gian, roi gom cum ke va ba phan.\n"
            "Chi dung su kien da xac minh; khong gan nhan cao trao hay ket thuc luc nay.\n"
            "Moi cum gom su kien lien tiep, co summary, event_ids, required_context_ids va sentence_count du kien.\n"
            "Moi cum la MOT tinh huong cu the, toi da 4 event va khoang 10 cau; khong gom ca phan phim vao mot cum.\n"
            "Gop cac mo ta cung ket qua/cam xuc vao mot y, uu tien thay doi trang thai va hanh dong moi.\n"
            "Phu tat ca su kien, khong lap ID. Tra hai ranh gioi giay va parts summary/end_state/handoff_to_next.\n"
            "Giu chuoi nguyen nhan-hanh dong-ket qua; depends_on chi dung ID co that truoc event.\n"
            "Khong them quan he nhan qua khi bang chung khong xac nhan.\n"
        )
        # Keep only the event ledger and grouping schema from the previous
        # schema; the locked backbone is always restored by the backend below.
        instruction = (
            f"Muc tieu DUNG {target_count} cau; de xuat sentence_count theo do quan trong va thong tin, "
            "backend chot tong chinh xac. Mot event co the co nhieu cau; estimated_facts KHONG PHAI tran so cau.\n"
            "SU KIEN VA SUC CHUA: " + json.dumps(capacities, ensure_ascii=False, separators=(",", ":")) +
            '\nChi tra JSON {"boundary_1_seconds":0,"boundary_2_seconds":0,"part_weights":[1,1,1],'
            '"selected_event_clusters":[{"cluster_id":"C001","summary":"","event_ids":["EV..."],'
            '"required_context_ids":[],"sentence_count":1}],"parts":[{"part":1,"summary":"","end_state":"","handoff_to_next":""},'
            '{"part":2,"summary":"","end_state":"","handoff_to_next":""},'
            '{"part":3,"summary":"","end_state":"","handoff_to_next":""}]}'
        )
        error = ""
        draft = copy.deepcopy(state.get("spine_draft") or state.get("grouping"))
        for attempt in range(3):
            budget_repair = draft is not None
            if budget_repair:
                # Older failed tasks already contain a usable story skeleton.
                # Reconcile its numbers locally instead of spending another call.
                text, model = json.dumps(draft, ensure_ascii=False), state.get("model", "")
            elif lightweight:
                local_grouping, _ = repair_spine_ownership({}, rows, duration)
                local_grouping = refine_story_clusters(local_grouping, rows)
                text, model = json.dumps(local_grouping, ensure_ascii=False), state.get('model', '')
            else:
                request_prompt = header + instruction + error
                label = "Gemini lap ke hoach 2/5: chia cum ke"
                text, model = generate(request_prompt, label)
            record = {'text': text, 'model': model, 'attempt': attempt + 1}
            state.setdefault('grouping_raw_responses', []).append(record)
            state['grouping_raw_responses'] = state['grouping_raw_responses'][-3:]
            persist()
            try:
                try:
                    candidate = parse(text)
                    if not isinstance(candidate, dict):
                        raise ValueError('SPINE_OBJECT_REQUIRED')
                except (ValueError, TypeError):
                    if not grounding_required:
                        raise
                    candidate, diagnostics = repair_spine_ownership({}, rows, duration)
                    state['grouping_repairs'] = diagnostics
                if exact_target:
                    candidate['sentence_budget_mode'] = 'exact_target'
                    candidate['target_sentence_count'] = target_count
                    candidate['grounding_required'] = grounding_required
                else:
                    candidate.pop('sentence_budget_mode', None)
                # Lock complete ordered ownership before assigning narrative roles.
                try:
                    validate_spine(candidate, rows, target_count)
                except SpineBudgetError:
                    pass
                except (ValueError, TypeError, KeyError) as exc:
                    if not grounding_required:
                        raise
                    candidate, diagnostics = repair_spine_ownership(candidate, rows, duration)
                    state['grouping_repairs'] = diagnostics
                    record['ownership_error'] = str(exc)
                    persist()
                    if log:
                        log(f'Backend da sua {len(diagnostics)} muc ID/cum tu so su kien nguon; tiep tuc, khong xem lai video.')
                candidate = refine_story_clusters(candidate, rows)
                state["grouping"] = copy.deepcopy(candidate)
                persist()
                assign_roles(candidate)
                for key in ("editorial_anchors", "story_contract", "editorial_contract_version", "turning_points"):
                    if key in candidate and candidate[key] != state["backbone"][key]:
                        raise ValueError(f"SPINE_BACKBONE_CHANGED:{key}")
                    candidate[key] = copy.deepcopy(state["backbone"][key])
                priority = {"critical": 3, "main": 2, "support": 1}
                for cluster in candidate.get("selected_event_clusters") or []:
                    cluster["importance"] = max((state["backbone"]["event_importance"].get(key, "main")
                                                 for key in cluster.get("event_ids") or []),
                                                key=lambda value: priority[value], default="main")
                if log:
                    log("Buoc 4/5: phan bo cau theo do quan trong va suc chua fact da xac minh.")
                spine = balance_spine_budget(candidate, rows, target_count, log)
                if not exact_target:
                    normalize(json.dumps(spine, ensure_ascii=False))
                else:
                    first = float(spine.get('boundary_1_seconds', 0))
                    second = float(spine.get('boundary_2_seconds', 0))
                    if not 0 < first < second < float(duration):
                        raise ValueError('SPINE_BOUNDARIES_INVALID: can 0 < boundary_1 < boundary_2 < duration')
                state.update(spine=spine, spine_model=model, model=model)
                state.pop("spine_draft", None)
                state.pop("spine_budget_diagnostics", None)
                persist()
                if log:
                    log(f"Da khoa khung toan phim: {len(spine['selected_event_clusters'])} cum, {target_count} cau; model={model}.")
                break
            except (ValueError, TypeError, KeyError) as exc:
                if str(exc).startswith("SPINE_ROLES_"):
                    raise
                if str(exc).startswith("SPINE_CAPACITY_TOTAL"):
                    state["spine_draft"] = copy.deepcopy(candidate)
                    state["spine_budget_diagnostics"] = spine_budget_report(candidate, rows, target_count)
                    persist()
                    raise
                # A malformed story structure still needs a semantic repair.
                # Numeric balancing never changes anchors or event ownership.
                draft = None
                error = "\nLAN TRUOC SAI: " + str(exc)
                if attempt == 2:
                    raise ValueError("SPINE_INVALID: " + str(exc)) from exc
    spine = validate_spine(copy.deepcopy(state["spine"]), rows, target_count)
    known = {str(row["evidence_id"]).upper(): row for row in rows}
    clusters = spine["selected_event_clusters"]
    global_guide = {key: spine.get(key) for key in ("editorial_anchors", "story_contract", "parts", "turning_points")}
    global_guide["clusters"] = [{"id": c["cluster_id"], "summary": c.get("summary", ""),
                                "sentence_count": c["sentence_count"]} for c in clusters]
    for index, cluster in enumerate(clusters):
        cid = cluster["cluster_id"]
        if cid in state["details"]:
            validate_cluster_detail(state["details"][cid], cluster)
            continue
        owned = [known[key] for key in cluster["event_ids"]]
        done_count = sum(len(value.get("sentences") or []) for value in state["details"].values())
        future_capacity = sum(event_narration_capacity(known[key])
                              for later in clusters[index+1:] for key in later["event_ids"])
        minimum_count = max(1, target_count - done_count - future_capacity)
        maximum_count = min(sum(event_narration_capacity(row) for row in owned),
                            target_count - done_count - len(clusters[index+1:]))
        if exact_target:
            minimum_count = maximum_count = cluster['sentence_count']
        global_guide["clusters"] = [{"id": c["cluster_id"], "summary": c.get("summary", ""),
                                    "sentence_count": c["sentence_count"]} for c in clusters]
        facts = [{"id": row["evidence_id"], "event": row.get("event"),
                  "facts": event_narration_foci(row), "cause": row.get("cause"),
                  "action": row.get("action"), "result": row.get("consequence") or row.get("change"),
                  "estimated_facts": event_narration_capacity(row)} for row in owned]
        previous = state["details"].get(clusters[index-1]["cluster_id"], {}) if index else {}
        context_ids = cluster.get("required_context_ids") or []
        context = [{"id": key, "event": known[key].get("event"),
                    "cause": known[key].get("cause"), "result": known[key].get("consequence")}
                   for key in context_ids if key in known]
        prompt = (
            "LUOT 5/5: Trien khai DUNG MOT CUM cua khung da khoa thanh cac O CAU, CHUA VIET TTS.\n"
            "Khong doi khung, su kien hay vai tro da xac minh. Vai tro rong la chua du bang chung.\n"
            "Khong tu viet them ket thuc. Ngan sach tung cum la DU KIEN, chi dung facts duoc cung cap.\n"
            "Moi o co source_event_ids va focus la y du kien can ke bang tieng Viet.\n"
            "Moi focus phai them fact moi; 'quyet dinh', 'the hien quyet tam', 'san sang bat dau' cung mot viec khong phai ba fact.\n"
            "Khong chia mot hanh dong thanh nhieu cam xuc/dien dat dong nghia de du chi tieu.\n"
            "Mot event co nhieu o neu moi o khai thac fact khac nhau. Mot o co the gom cac event lien tiep.\n"
            "Tat ca event_ids cua cum phai duoc phu; khong them ID ngoai cum. Khong lap focus.\n"
            "Giu nguyen nhan -> hanh dong -> dao chieu -> ket qua, noi tu cum truoc sang cum sau.\n"
            "KHUNG TOAN PHIM: " + json.dumps(global_guide, ensure_ascii=False) +
            "\nCUM CAN LAM: " + json.dumps(cluster, ensure_ascii=False) +
            "\nSU KIEN: " + json.dumps(facts, ensure_ascii=False) +
            "\nCONTEXT CHI DE NOI MACH, KHONG KE LAI: " + json.dumps(context, ensure_ascii=False) +
            "\nHAI O CUOI CUM TRUOC, KHONG LAP LAI: " + json.dumps(previous.get("sentences", [])[-2:], ensure_ascii=False) +
            '\nChi tra JSON: {"cluster_id":"' + cid + '","ideas_by_event":{"EV...":["y thu nhat","y thu hai"]}}.'
            "\nMoi ID trong CUM CAN LAM la MOT khoa ideas_by_event, moi khoa co danh sach y rieng. "
            "Khong gom nhieu ID trong mot khoa, khong tu them ID. Backend so huu thu tu, tu ghep cac y theo event_ids."
            f"\nDu kien {cluster['sentence_count']} o cau; co the linh hoat {minimum_count}-{maximum_count} o "
            "neu moi o co y moi. Backend can bang cac cum chua viet de giu tong muc tieu."
        )
        if exact_target:
            prompt = prompt.replace('Ngan sach tung cum la DU KIEN', 'So o cua cum da KHOA')
            prompt = prompt.rsplit('\nDu kien ', 1)[0]
            prompt += (
                f'\nCHI TIEU DA KHOA: phai tra DUNG {cluster["sentence_count"]} y ke cho cum nay. '
                f'Tong toan phim DUNG {target_count} cau. Khong tang/giam quota cua cum. '
                'Moi su kien duoc co NHIEU cau, khong gioi han 4; phan bo theo thong tin va do quan trong. '
                'Moi y lam ro mot chi tiet, nguyen nhan, hanh dong, dien bien hoac ket qua CO BANG CHUNG. '
                'Khong tach cau bang tu dem, lap y hay tu tao hanh dong de du chi tieu. '
                'Uoc luong so fact chi la goi y, khong phai gioi han so cau.'
            )
        error = ""
        if grounding_required:
            from grounded_pipeline import grounded_detail
            detail = grounded_detail(cluster, owned)
            validate_cluster_detail(detail, cluster)
            state['details'][cid] = detail
            persist()
            if log:
                log(f'Cum {cid}: da khoa {len(detail["sentences"])} o tu fact+shot quan sat; khong goi AI tao them y.')
            continue
        for attempt in range(3):
            from detail_repair import inspect_detail_rows, build_detail_repair_prompt, apply_detail_repairs
            from event_ideas import (inspect_event_ideas, event_ideas_repair_prompt,
                                     apply_event_idea_repairs, legacy_event_idea_draft)
            draft_detail = copy.deepcopy((state.get("detail_drafts") or {}).get(cid))
            row_issues = inspect_detail_rows(draft_detail, cluster) if draft_detail else []
            idea_draft = copy.deepcopy((state.get('event_idea_drafts') or {}).get(cid))
            idea_issues = copy.deepcopy((state.get('event_idea_issues') or {}).get(cid) or [])
            targeted_ideas = bool(idea_draft is not None and idea_issues)
            count_mismatch = bool(exact_target and draft_detail
                                  and len(draft_detail.get('sentences') or []) != cluster['sentence_count'])
            targeted = bool(row_issues) and not targeted_ideas and not count_mismatch
            request_prompt = (event_ideas_repair_prompt(idea_draft, cluster, idea_issues, facts)
                              if targeted_ideas else build_detail_repair_prompt(draft_detail, cluster, row_issues, facts)
                              if targeted else prompt) + error
            if count_mismatch and not targeted_ideas:
                request_prompt += ('\nPHAN BO LAI DUNG SO O CHO RIENG CUM NAY; GIU DU ID NGUON. '
                                   'BAN NHAP CHUA DUNG SO CAU: ' + json.dumps(draft_detail, ensure_ascii=False))
            label = (f"Gemini lap ke hoach 5/5: sua rieng ID loi cum {cid}" if targeted_ideas else
                     f"Gemini lap ke hoach 5/5: sua rieng {len(row_issues)} o cum {cid}" if targeted else
                     f"Gemini lap ke hoach 5/5: cum {index+1}/{len(clusters)} {cid}")
            text, model = generate(request_prompt, label)
            raw_record = {'text': text, 'model': model, 'attempt': attempt + 1, 'label': label}
            raw_history = state.setdefault('detail_raw_responses', {}).setdefault(cid, [])
            raw_history.append(raw_record)
            del raw_history[:-3]
            persist()  # Save BEFORE JSON parsing or any validator can fail.
            candidate_detail = None
            parsed = None
            try:
                parsed = parse(text)
                if targeted_ideas:
                    parsed = apply_event_idea_repairs(idea_draft, parsed, cluster, idea_issues)
                if isinstance(parsed, dict) and 'ideas_by_event' in parsed:
                    idea_draft, idea_issues = inspect_event_ideas(parsed, cluster)
                    state.setdefault('event_idea_drafts', {})[cid] = idea_draft
                    state.setdefault('event_idea_issues', {})[cid] = idea_issues
                    persist()
                candidate_detail = (apply_detail_repairs(draft_detail, parsed,
                                     [item["row_index"] for item in row_issues]) if targeted
                                    else expand_event_plans(parsed, cluster))
                if (isinstance(candidate_detail, dict) and candidate_detail.get("cluster_id") == cid
                        and isinstance(candidate_detail.get("sentences"), list) and candidate_detail["sentences"]):
                    state.setdefault("detail_drafts", {})[cid] = copy.deepcopy(candidate_detail)
                    diagnostics = inspect_detail_rows(candidate_detail, cluster)
                    state.setdefault("detail_diagnostics", {})[cid] = diagnostics
                    state["model"] = model
                    persist()
                    if diagnostics and log:
                        log(f"Cum {cid}: da giu ban nhap; cac o can sua: "
                            + json.dumps(diagnostics, ensure_ascii=False))
                detail = validate_cluster_detail(candidate_detail, cluster, allow_flexible_count=not exact_target)
                actual_count = len(detail["sentences"])
                if not minimum_count <= actual_count <= maximum_count:
                    raise ValueError(f"DETAIL_GLOBAL_BUDGET:{cid}: can {minimum_count}-{maximum_count} o de du tong {target_count}")
                actual_cluster = dict(cluster, sentence_count=actual_count)
                # Verify that per-event allocations can actually be materialized
                # before committing this cluster to the durable checkpoint.
                materialize_sentence_plan({"selected_event_clusters": [actual_cluster], "cluster_details": {cid: detail}}, owned,
                                          defer_visual_selection=defer_visual_selection)
                fixed = {key: len(value["sentences"]) for key, value in state["details"].items()}
                fixed[cid] = actual_count
                adjusted = balance_spine_budget(spine, rows, target_count, log, fixed_counts=fixed)
                for current, updated in zip(clusters, adjusted["selected_event_clusters"]):
                    current["sentence_count"] = updated["sentence_count"]
                state["spine"] = copy.deepcopy(spine)
                state["details"][cid] = detail
                state.get("detail_drafts", {}).pop(cid, None)
                state.get("detail_diagnostics", {}).pop(cid, None)
                state.get('event_idea_drafts', {}).pop(cid, None)
                state.get('event_idea_issues', {}).pop(cid, None)
                state.get('detail_last_errors', {}).pop(cid, None)
                state["model"] = model
                persist()
                if log:
                    log(f"Da luu cum {cid}: {len(detail['sentences'])} o cau; model={model}.")
                break
            except (ValueError, TypeError, KeyError) as exc:
                raw_record['error'] = str(exc)
                state.setdefault('detail_last_errors', {})[cid] = {'message': str(exc), 'attempt': attempt + 1}
                if (candidate_detail is None and isinstance(parsed, dict) and 'event_plans' in parsed):
                    fallback_draft, fallback_issues = legacy_event_idea_draft(parsed, cluster)
                    state.setdefault('event_idea_drafts', {})[cid] = fallback_draft
                    state.setdefault('event_idea_issues', {})[cid] = fallback_issues
                persist()
                if log:
                    log(f'Cum {cid}, lan {attempt + 1}/3: ' + str(exc))
                error = "\nCUM NAY SAI: " + str(exc) + "; chi sua cum nay, khong sua khung hay cum khac."
                if attempt == 2:
                    raise ValueError(f"DETAIL_INVALID:{cid}: " + str(exc)) from exc
    result = copy.deepcopy(spine)
    result["planning_version"] = PLANNING_VERSION
    if defer_visual_selection:
        result['visual_selection_mode'] = 'sentence_first'
    result["cluster_details"] = copy.deepcopy(state["details"])
    if exact_target:
        allocated = sum(len(detail['sentences']) for detail in result['cluster_details'].values())
        if allocated != target_count:
            raise ValueError(f'PLAN_SENTENCE_TOTAL_MISMATCH:{allocated}/{target_count}')
    result_text = json.dumps(result, ensure_ascii=False)
    try:
        normalize(result_text)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("DETAIL_MERGE_INVALID: " + str(exc)) from exc
    return result_text, state.get("model", "")
