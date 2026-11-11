"""Trace overview milestones to observed source facts without inventing coverage."""
import math
import copy


def merge_plot_events(chunks, events):
    """Keep supplemental facts inside existing chapter checkpoints, once only."""
    for event in events:
        ids = {f.get('fact_id') for f in event.get('grounded_facts') or []}
        known = {f.get('fact_id') for c in chunks for e in c.get('events') or []
                 for f in e.get('grounded_facts') or []}
        if not ids or ids <= known or not chunks:
            continue
        center = (float(event['start_seconds']) + float(event['end_seconds'])) / 2
        owner = min(chunks, key=lambda c: (
            not float(c.get('start_seconds', 0)) <= center < float(c.get('primary_end_seconds', c.get('end_seconds', 0))),
            abs(float(c.get('start_seconds', 0)) - center)))
        owner.setdefault('events', []).append(copy.deepcopy(event))


def plot_coverage(outline, chunks, duration, reviewed=()):
    facts = {}
    for chunk in chunks:
        for event in chunk.get('events') or []:
            sources = {sid: source for source in event.get('visual_sources') or []
                       for sid in source.get('shot_ids') or [source.get('shot_id')] if sid}
            for fact in event.get('grounded_facts') or []:
                if fact.get('status', 'confirmed') != 'confirmed' or fact.get('evidence_kind', 'visual') != 'visual':
                    continue
                windows = [sources[sid] for sid in fact.get('shot_ids') or [] if sid in sources]
                if not windows or not fact.get('fact_id'):
                    continue
                facts[fact['fact_id']] = (fact, min(float(w['start_seconds']) for w in windows),
                                        max(float(w['end_seconds']) for w in windows))
    milestones = []
    gaps = [g for c in chunks for g in c.get('plot_gaps') or [] if isinstance(g, dict)]
    for beat in (outline or {}).get('story_beats') or []:
        try:
            start, end = max(0., float(beat['start_seconds'])), min(float(duration), float(beat['end_seconds']))
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start + end) or end <= start:
            continue
        bid = str(beat.get('beat_id') or '')
        ending = beat.get('story_role') == 'ending' or beat.get('narrative_function') == 'ending'
        overlaps = [fid for fid, (_, a, b) in facts.items() if a < end and b > start]
        explicit = [fid for fid, (f, _, _) in facts.items() if bid and bid in (f.get('story_beat_ids') or [])]
        declared_gaps = [g for g in gaps if g.get('beat_id') == bid]
        status = ('reviewed_unresolved' if declared_gaps and bid in reviewed else
                  'partial_sources' if declared_gaps else
                  'source_linked' if explicit else 'candidate_sources' if overlaps else
                  'reviewed_not_observed' if bid in reviewed else 'missing_sources')
        milestones.append(dict(beat_id=bid, summary=beat.get('summary', ''),
            start_seconds=start, end_seconds=end, priority='low' if ending else beat.get('importance', 'main'),
            status=status, fact_ids=explicit or overlaps,
            gaps=declared_gaps,
            overview_inferred=bool(beat.get('ending_inferred_by_backend'))))
    return dict(milestones=milestones,
        missing_main_ids=[r['beat_id'] for r in milestones if r['status'] in {'missing_sources', 'partial_sources'} and r['priority'] != 'low'],
        uncertain_ids=[r['beat_id'] for r in milestones if r['status'] in {'candidate_sources', 'reviewed_not_observed', 'reviewed_unresolved'}],
        all_milestones_verified=False,
        note='Temporal overlap is a candidate link, not semantic proof that the whole milestone occurred.')


def missing_plot_windows(report, budget=2, max_seconds=60):
    rows = [r for r in report['milestones'] if r['status'] in {'missing_sources', 'partial_sources'} and r['priority'] != 'low'
            and r.get('beat_id')]
    rows.sort(key=lambda r: (r['priority'] != 'main', r['start_seconds']))
    windows = []
    for row in rows[:max(0, budget)]:
        from grounded_pipeline import SHOT_SECONDS
        starts = [(int(sid[5:]) - 1) * SHOT_SECONDS for g in row['gaps'] for sid in g.get('shot_ids') or []
                  if str(sid).startswith('SHOT_') and str(sid)[5:].isdigit()]
        start = max(row['start_seconds'], min(starts)) if starts else row['start_seconds']
        end = min(row['end_seconds'], start + max_seconds)
        if end > start:
            windows.append(dict(start_seconds=start, end_seconds=end, beat_id=row['beat_id'],
                summary=row['summary'], reason='missing_main_plot_source'))
    return windows
