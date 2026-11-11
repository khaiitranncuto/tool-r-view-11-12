"""Inspect and patch defective sentence-plan rows without rewriting a cluster."""

import copy
import json
import re


def _safe_ids(row):
    ids = row.get('source_event_ids') if isinstance(row, dict) else None
    return ids if isinstance(ids, list) else []


def inspect_detail_rows(detail, cluster):
    rows = detail.get("sentences") if isinstance(detail, dict) else None
    if not isinstance(rows, list) or not rows:
        return []
    order = {key: index for index, key in enumerate(cluster["event_ids"])}
    issues = {}

    def flag(index, code, **extra):
        issues.setdefault(index, []).append({"code": code, **extra})

    previous = None
    foci, owners, covered = {}, {}, set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            flag(index, "invalid_row")
            continue
        raw_ids = row.get("source_event_ids")
        if not isinstance(raw_ids, list):
            flag(index, "invalid_source_ids")
            continue
        ids = [str(key).strip().upper() for key in raw_ids]
        if not isinstance(row.get('focus'), str):
            flag(index, 'invalid_focus_type')
            continue
        focus = re.sub(r"\s+", " ", str(row.get("focus") or "")).strip()
        if not ids or len(set(ids)) != len(ids) or any(key not in order for key in ids) or not focus:
            flag(index, "unknown_id_or_empty_focus", supplied_ids=ids)
            continue
        positions = [order[key] for key in ids]
        expected = sorted(ids, key=order.get)
        if positions != sorted(positions):
            # Do not silently reorder prose or relabel its action. AI must
            # check the actual focus as well as the metadata in this row.
            flag(index, "ids_out_of_order", supplied_ids=ids, expected_id_order=expected)
        if previous is not None and min(positions) < previous[1]:
            left = previous[0]
            flag(left, "event_order_conflict", conflicting_row=index + 1,
                 expected_id_order=sorted(set(ids + previous[2]), key=order.get))
            flag(index, "event_order_conflict", conflicting_row=left + 1,
                 expected_id_order=sorted(set(ids + previous[2]), key=order.get))
        for key in ids:
            if key in owners:
                earlier, group = owners[key]
                if len(ids) > 1 or group != tuple(ids):
                    flag(earlier, "overlapping_event_group", conflicting_row=index + 1)
                    flag(index, "overlapping_event_group", conflicting_row=earlier + 1)
            owners[key] = (index, tuple(ids))
        focus_key = tuple(re.findall(r"\w+", focus.casefold()))
        if focus_key in foci:
            flag(index, "repeated_focus", conflicting_row=foci[focus_key] + 1)
        else:
            tokens = set(focus_key)
            for old_focus, earlier in foci.items():
                old_ids = [str(key).strip().upper() for key in rows[earlier].get('source_event_ids') or []]
                old_tokens = set(old_focus)
                smaller = min(len(tokens), len(old_tokens))
                if (ids == old_ids and smaller >= 5 and len(tokens & old_tokens) / smaller >= .85
                        and len(tokens ^ old_tokens) <= 2):
                    flag(index, 'near_repeated_focus', conflicting_row=earlier + 1)
                    break
        foci[focus_key] = index
        covered.update(ids)
        previous = (index, max(positions), ids)
    missing = set(order) - covered
    if missing:
        # Replacing an adjacent row can cover the missing event with a causal
        # merged focus while preserving the global sentence count.
        key = min(missing, key=order.get)
        nearest = min(range(len(rows)), key=lambda i: min((
            abs(order[str(value).strip().upper()] - order[key]) for value in
            _safe_ids(rows[i]) if str(value).strip().upper() in order
        ), default=len(order)) if isinstance(rows[i], dict) else len(order))
        flag(nearest, "missing_event_coverage", missing_ids=sorted(missing, key=order.get))
    return [{"row_index": index + 1, "issues": codes,
             "previous_ids": _safe_ids(rows[index - 1])
                             if index and isinstance(rows[index - 1], dict) else [],
             "next_ids": _safe_ids(rows[index + 1])
                         if index + 1 < len(rows) and isinstance(rows[index + 1], dict) else []}
            for index, codes in sorted(issues.items())]


def build_detail_repair_prompt(detail, cluster, issues, facts):
    rows = detail["sentences"]
    targets = {item["row_index"] for item in issues}
    context_indexes = {i for target in targets for i in (target - 2, target - 1, target)
                       if 0 <= i < len(rows)}
    context = [{"row_index": i + 1, "read_only": i + 1 not in targets, "row": rows[i]}
               for i in sorted(context_indexes)]
    relevant_ids = {str(key).strip().upper() for row in context if isinstance(row["row"], dict)
                    for key in _safe_ids(row['row'])}
    for issue in issues:
        for code in issue["issues"]:
            relevant_ids.update(code.get("missing_ids") or [])
    evidence = [fact for fact in facts if fact["id"] in relevant_ids]
    return (
        f"SUA RIENG CAC O LOI CUA CUM {cluster['cluster_id']}, KHONG TAO LAI CA CUM.\n"
        f"Giu nguyen tong {len(rows)} o, giu row_index; khong chen, xoa hay di chuyen o.\n"
        "Chi sua source_event_ids va focus cua cac o duoc yeu cau. Context read_only khong duoc sua.\n"
        "Kiem tra noi dung y ke va quan he nguyen nhan-hanh dong-ket qua, khong chi doi ID cho qua validator.\n"
        "Su kien co the co nhieu o neu moi o la mot y moi. Khong lap focus, khong them du kien.\n"
        "THU TU SU KIEN DA KHOA: " + json.dumps(cluster["event_ids"]) +
        "\nCAC O CAN SUA: " + json.dumps(issues, ensure_ascii=False) +
        "\nCONTEXT LIEN KE: " + json.dumps(context, ensure_ascii=False) +
        "\nBANG CHUNG LIEN QUAN: " + json.dumps(evidence, ensure_ascii=False) +
        '\nChi tra JSON {"cluster_id":"' + cluster["cluster_id"] +
        '","repairs":[{"row_index":1,"source_event_ids":["EV..."],"focus":"y ke da sua"}]}. '
        "Tra dung cac row_index: " + json.dumps(sorted(targets))
    )


def apply_detail_repairs(detail, patch, requested_rows):
    if not isinstance(patch, dict) or patch.get("cluster_id") != detail.get("cluster_id"):
        raise ValueError("DETAIL_PATCH_CLUSTER_ID")
    repairs = patch.get("repairs")
    if not isinstance(repairs, list):
        raise ValueError("DETAIL_PATCH_SCHEMA")
    result = copy.deepcopy(detail)
    expected, seen = set(requested_rows), set()
    for row in repairs:
        if not isinstance(row, dict) or set(row) != {"row_index", "source_event_ids", "focus"}:
            raise ValueError("DETAIL_PATCH_FIELDS")
        index = row["row_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index not in expected or index in seen:
            raise ValueError(f"DETAIL_PATCH_ROW:{index}")
        if not isinstance(row["source_event_ids"], list) or not isinstance(row["focus"], str):
            raise ValueError("DETAIL_PATCH_CONTENT")
        seen.add(index)
        result["sentences"][index-1] = {"source_event_ids": copy.deepcopy(row["source_event_ids"]),
                                       "focus": row["focus"]}
    if seen != expected:
        raise ValueError("DETAIL_PATCH_ROWS_MISSING")
    return result
