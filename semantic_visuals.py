"""Validate narration-to-action links against backend-owned source ranges."""

import math
import re

VISUAL_ALIGNMENT_CONTRACT_VERSION = 1
MAX_ACTION_SECONDS = 15.0


def seconds(value):
    parts = str(value).split(":")
    result = sum(float(part) * 60 ** index for index, part in enumerate(reversed(parts)))
    if not math.isfinite(result) or result < 0:
        raise ValueError("Invalid visual time")
    return result


def timecode(value):
    return f"{int(value // 3600):02d}:{int(value % 3600 // 60):02d}:{value % 60:05.2f}"


def visual_catalog(sources):
    """Keep stable IDs from the supplied list; the writer never invents times."""
    rows = []
    seen = set()
    for index, source in enumerate(sources or [], 1):
        if not isinstance(source, dict):
            continue
        try:
            start, end = seconds(source.get("start")), seconds(source.get("end"))
        except (TypeError, ValueError, OverflowError):
            continue
        key = (round(start, 2), round(end, 2))
        if not 0.05 < end - start <= MAX_ACTION_SECONDS + 0.05 or key in seen:
            continue
        seen.add(key)
        rows.append({
            "visual_id": f"V{index:03d}",
            "start": timecode(start), "end": timecode(end),
            "start_seconds": start, "end_seconds": end,
            "action": str(source.get("event") or source.get("action") or "").strip(),
            "visual_match_tier": source.get("visual_match_tier") or "verified_beat",
            "shot_id": source.get("shot_id"),
            "shot_ids": list(source.get("shot_ids") or ([source['shot_id']] if source.get('shot_id') else [])),
            "fact_ids": list(source.get("fact_ids") or []),
        })
    return rows


def semantic_visual_plan(narration, sources, links=None, legacy_beats=None):
    """Lock complete action windows to exact spoken triggers, never word ratios.

    Missing/invalid links are exposed for targeted repair. A provisional primary
    image is retained so that a defective row can be parsed and repaired alone.
    """
    catalog = visual_catalog(sources)
    if not catalog:
        raise ValueError("VISUAL_GROUNDING_MISSING: Cau recap khong co vung hanh dong da xem.")
    lookup = {row["visual_id"]: row for row in catalog}
    text = re.sub(r"\s+", " ", str(narration or "")).strip()
    errors, selected, beats, normalized_links = [], [], [], []
    raw_links = links if isinstance(links, list) else []
    # Older responses may contain useful literal triggers. Accept only ranges
    # wholly inside a known action; discard arbitrary model timestamps.
    if not raw_links and legacy_beats:
        raw_links = [{"visual_id": catalog[0]["visual_id"], "trigger_text": ""}]
        for beat in legacy_beats:
            if not isinstance(beat, dict):
                continue
            try:
                start, end = seconds(beat.get("start")), seconds(beat.get("end"))
            except (TypeError, ValueError, OverflowError):
                continue
            owner = next((row for row in catalog if
                          row["start_seconds"] <= start < end <= row["end_seconds"]), None)
            if owner and owner["visual_id"] != raw_links[-1]["visual_id"]:
                raw_links.append({"visual_id": owner["visual_id"],
                                  "trigger_text": beat.get("trigger_text", "")})
    if not raw_links:
        raw_links = [{"visual_id": catalog[0]["visual_id"], "trigger_text": ""}]
        if len(catalog) > 1:
            errors.append("missing_visual_links: link spoken actions to visual_id")
    previous_end, previous_text_end = -1.0, 0
    for index, link in enumerate(raw_links):
        if not isinstance(link, dict):
            errors.append("invalid_visual_link")
            continue
        visual_id = str(link.get("visual_id") or "").strip().upper()
        shot = lookup.get(visual_id)
        trigger = re.sub(r"\s+", " ", str(link.get("trigger_text") or "")).strip()
        if shot is None:
            errors.append(f"unknown_visual_id:{visual_id}")
            continue
        if index == 0:
            # Opening starts with speech and needs no model-generated trigger.
            trigger = ""
        else:
            matches = list(re.finditer(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", text, re.I)) if trigger else []
            if (not 2 <= len(trigger.split()) <= 12 or len(matches) != 1
                    or matches[0].start() < previous_text_end
                    or not text[:matches[0].start()].strip()):
                errors.append(f"invalid_spoken_trigger:{trigger}")
                continue
            # Cuts are ordered by their START. A backend-derived literal cue
            # may include lookahead words from the following short sentence.
            previous_text_end = matches[0].start() + 1
        if shot["start_seconds"] < previous_end - 0.01:
            errors.append(f"visual_order_or_overlap:{visual_id}")
            continue
        if not selected and index:
            errors.append("missing_opening_visual")
            continue
        selected.append(dict(shot))
        normalized_links.append({"visual_id": visual_id, "trigger_text": trigger})
        if index:
            beats.append({"trigger_text": trigger, "start": shot["start"], "end": shot["end"],
                          "match_type": "literal", "confidence": 1.0, "backend_locked": True,
                          "semantic_action": True})
        previous_end = shot["end_seconds"]
    if not selected:
        selected = [catalog[0]]
    return {
        "start": selected[0]["start"], "end": selected[0]["end"],
        "visual_beats": beats, "locked_visual_shots": selected,
        "visual_links": (normalized_links if not errors else (links if isinstance(links, list) else [])),
        "visual_alignment_errors": errors,
        "visual_alignment_version": VISUAL_ALIGNMENT_CONTRACT_VERSION,
    }
