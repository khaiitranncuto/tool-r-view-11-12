"""One source timeline, written as consecutive small batches."""
import copy


def global_writing_plan(plan):
    result = copy.deepcopy(plan)
    events = [e for group in result.get('events_by_part') or [] for e in group]
    if not events:
        raise ValueError('GLOBAL_WRITING_PLAN_EMPTY')
    duration = float(result['boundaries'][-1])
    summaries = list(dict.fromkeys(str(s).strip() for s in result.get('summaries') or [] if str(s).strip()))
    beats = [b for group in result.get('story_beats_by_part') or [] for b in group]
    result.update(events_by_part=[events], boundaries=[0.0, duration],
        summaries=summaries or [''], story_beats_by_part=[beats],
        writing_layout='global_batches_v1')
    contract = result.setdefault('scene_contract', {})
    contract['part_scene_counts'] = [len(events)]
    contract['part_word_targets'] = [int(contract.get('target_words') or 0)]
    return result
