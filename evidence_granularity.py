"""Keep chapter summaries out of the chronological sentence-event ledger."""

import copy
import re


def _context_rows(value):
    return list(value) if isinstance(value, list) else ([value] if isinstance(value, (dict, str)) else [])


def separate_event_granularity(chunks, review_legacy=True, preserve_unlocated_events=False):
    result = copy.deepcopy(chunks or [])
    all_events = [row for chunk in result for row in chunk.get("events", []) if isinstance(row, dict)]
    context_ids, review_ids = set(), set()
    for chunk in result:
        contexts = _context_rows(chunk.get("story_context"))
        reviews = _context_rows(chunk.get("event_review_candidates"))
        kept = []
        for row in chunk.get("events") or []:
            if not isinstance(row, dict):
                continue
            kind = str(row.get("granularity") or "").strip().lower()
            evidence_id = str(row.get("evidence_id") or "")
            if kind == "summary":
                context_ids.add(evidence_id)
                row["narration_eligible"] = False
                contexts.append(row)
                continue
            if review_legacy and kind != "atomic" and kind != "uncertain":
                start = float(row.get("start_seconds", 0) or 0)
                end = float(row.get("end_seconds", start) or start)
                children = [other for other in all_events if other is not row
                            and str(other.get("evidence_id")) != evidence_id
                            and start <= float(other.get("start_seconds", -1))
                            < float(other.get("end_seconds", -1)) <= end
                            and float(other.get("end_seconds", 0)) - float(other.get("start_seconds", 0)) < (end-start)/3]
                meanings = {re.sub(r"\W+", " ", str(row.get(key) or "")).strip().casefold()
                            for key in ("event", "cause", "action", "change", "consequence", "result")}
                meanings.discard("")
                # Legacy broad items need review, not guessed subdivision.
                # Duration alone never demotes an event. Explicitly atomic
                # continuous actions are preserved regardless of duration.
                if end-start >= 120 and len(meanings) >= 3 and (len(children) >= 2 or (
                        not row.get("visual_sources") and not preserve_unlocated_events)):
                    kind = "uncertain"
                    row["granularity_review_reason"] = "broad_multi_fact_event_without_atomic_contract"
                    row["contained_event_ids"] = [other["evidence_id"] for other in children]
            if kind == "uncertain":
                review_ids.add(evidence_id)
                row["granularity"] = "uncertain"
                row["narration_eligible"] = False
                reviews.append(row)
                continue
            row["narration_eligible"] = True
            kept.append(row)
        # Retain source records, including visuals, for audit/review. They
        # simply stop competing with concrete events for sentence slots.
        chunk["events"] = kept
        chunk["story_context"] = contexts
        chunk["event_review_candidates"] = reviews
    unavailable = {str(row.get("evidence_id")) for chunk in result
                   for row in (chunk.get("story_context") or []) + (chunk.get("event_review_candidates") or [])
                   if isinstance(row, dict) and row.get("evidence_id")}
    for chunk in result:
        for event in chunk["events"]:
            dependencies = event.get("depends_on") or []
            event["context_dependencies"] = list(dict.fromkeys(
                list(event.get("context_dependencies") or []) + [key for key in dependencies if key in unavailable]
            ))
            event["depends_on"] = [key for key in dependencies if key not in unavailable]
    return result, {"summary_ids": sorted(context_ids), "review_ids": sorted(review_ids),
                    "atomic_count": sum(len(chunk["events"]) for chunk in result)}
