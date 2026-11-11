import copy
import json
import unittest

from global_writing import global_writing_plan
from evidence_writer import build_sentence_plan, writing_batches, write_batch
from narrative_contract import sentences
from recap_pipeline import build_part_story_context, coerce_single_part_response, merge_single_part_responses
from source_grounded_writer import is_source_grounded


class GlobalWritingTests(unittest.TestCase):
    def plan(self):
        events = [dict(event_id=f'E{i:04d}', scene_id=f'S{i:04d}',
            start_seconds=(i - 1) * 12, end_seconds=(i - 1) * 12 + 6,
            sentence_quota=1, grounded_plan_locked=True, situation_narration=True,
            source_link_contract_version=1, characters=[],
            observed_facts=[dict(fact_id=f'F{i}', text='Nobita mở cửa.',
                shot_ids=[f'SHOT_{i:06d}'], evidence_kind='visual')],
            visual_sources=[dict(shot_id=f'SHOT_{i:06d}', fact_ids=[f'F{i}'],
                start_seconds=(i - 1) * 12, end_seconds=(i - 1) * 12 + 6)]) for i in range(1, 8)]
        return dict(events_by_part=[events[:2], events[2:4], events[4:]], boundaries=[0, 24, 48, 84],
            summaries=['Đầu truyện', 'Giữa truyện', 'Cuối truyện'], story_beats_by_part=[[], [], []],
            characters=[], sentence_contract={'target_count': 7},
            scene_contract=dict(scene_count=7, target_words=140, part_scene_counts=[2, 2, 3]))

    def test_old_plan_becomes_one_timeline_without_changing_events_or_quotas(self):
        plan = self.plan()
        original = copy.deepcopy(plan)
        result = global_writing_plan(plan)
        self.assertEqual(result['boundaries'], [0, 84])
        self.assertEqual(len(result['events_by_part']), 1)
        self.assertEqual(result['events_by_part'][0], sum(plan['events_by_part'], []))
        self.assertEqual(result['scene_contract']['part_word_targets'], [140])
        self.assertEqual(result['summaries'], original['summaries'])
        self.assertEqual(global_writing_plan(result), result)
        self.assertEqual(plan, original)

    def test_groups_merge_directly_to_one_timeline_and_resume_without_generation(self):
        plan = build_sentence_plan(global_writing_plan(self.plan()))
        context = build_part_story_context(plan, 1)
        events = context['required_events']
        # Context uses formatted global times because the only offset is zero.
        for e in events:
            for s in e['visual_sources']:
                from semantic_visuals import seconds
                s['start_seconds'], s['end_seconds'] = seconds(s['start']), seconds(s['end'])
        rows, calls = [], []
        for batch in writing_batches(events, max_sentences=2):
            def generate(missing, prompt):
                calls.append([e['event_id'] for e in missing])
                return json.dumps({'scenes': [dict(event_id=e['event_id'], paragraph='Nobita mở cửa.',
                    sentence_evidence=[dict(fact_ids=[e['observed_facts'][0]['fact_id']], evidence_kind='visual')])
                    for e in missing]})
            drafts = []
            bound = write_batch(batch, context, generate, save=lambda d: drafts.extend(d))
            def forbidden(*_):
                self.fail('Validated saved groups must not be generated again')
            self.assertEqual(write_batch(batch, context, forbidden, candidates=drafts), bound)
            rows.extend(bound)
        self.assertEqual(len(calls), 4)
        response = coerce_single_part_response(json.dumps({'scenes': rows}), 1, 84, 140,
            locked_event_by_id={e['event_id']: e for e in events})
        merged = merge_single_part_responses([response], [0, 84], 140, single_timeline=True)
        self.assertEqual(len(merged['parts']), 1)
        scenes = merged['parts'][0]['scenes']
        self.assertEqual([s['scene_id'] for s in scenes], [f'S{i:04d}' for i in range(1, 8)])
        self.assertEqual(sum(len(sentences(s['tts'])) for s in scenes), 7)
        self.assertTrue(all(is_source_grounded(s) for s in scenes))
        self.assertEqual(scenes[-1]['locked_visual_shots'][0]['start_seconds'], 72)
