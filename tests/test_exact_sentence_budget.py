import copy
import json
import unittest

from recap_pipeline import (event_narration_capacity, normalize_story_plan_from_evidence,
                            build_part_story_context, coerce_single_part_response,
                            validate_single_part_response)
from story_planning import balance_spine_budget, run_two_stage_planning, validate_spine, spine_budget_report


class ExactSentenceBudgetTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        fields = ['state_before', 'cause', 'action', 'event', 'result', 'state_after']
        for i in range(30):
            # Reproduce the reported distribution: 10*4 + 7*5 + 13*6 = 153
            # initial fact cards. The plan still has to allocate exactly 156.
            count = 4 if i < 10 else 5 if i < 17 else 6
            event = {'evidence_id': f'EV{i:03d}', 'start_seconds': i * 20,
                     'end_seconds': (i + 1) * 20, 'sentence_first': True}
            for j, field in enumerate(fields[:count]):
                event[field] = f'{field}{i} detail{j}'
            self.events.append(event)
        self.chunks = [{'events': self.events}]
        self.spine = {'boundary_1_seconds': 200, 'boundary_2_seconds': 400,
                      'parts': [{'part': i, 'summary': f'Part {i}'} for i in range(1, 4)],
                      'selected_event_clusters': [{'cluster_id': f'C{i:03d}', 'event_ids': [e['evidence_id']],
                                                   'sentence_count': 4} for i, e in enumerate(self.events)]}

    def test_event_can_have_more_than_four_facts_in_sentence_first_mode(self):
        self.assertEqual(event_narration_capacity(self.events[-1]), 6)
        self.assertEqual(sum(event_narration_capacity(e) for e in self.events), 153)

    def test_global_allocation_reaches_156_without_treating_estimates_as_ceiling(self):
        plan = dict(copy.deepcopy(self.spine), sentence_budget_mode='exact_target')
        balanced = balance_spine_budget(plan, self.events, 156)
        self.assertEqual(sum(c['sentence_count'] for c in balanced['selected_event_clusters']), 156)
        self.assertTrue(any(c['sentence_count'] > 4 for c in balanced['selected_event_clusters']))
        self.assertEqual([c['event_ids'] for c in balanced['selected_event_clusters']],
                         [c['event_ids'] for c in self.spine['selected_event_clusters']])
        report = spine_budget_report(balanced, self.events, 156)
        self.assertEqual(report['estimated_facts_total'], 153)
        self.assertFalse(report['capacity_is_limit'])
        validate_spine(balanced, self.events, 156)

    def normalize(self, text):
        return normalize_story_plan_from_evidence(text, self.chunks, 600, 3120,
                                                  require_editorial_contract=True)

    def generator(self, bad_once=None):
        calls = []
        def generate(prompt, label):
            calls.append(label)
            if '2/5:' in label:
                return json.dumps(self.spine), 'model'
            if '3/5:' in label:
                return json.dumps({'editorial_anchors': {}, 'event_importance': {}}), 'model'
            cluster = json.loads(prompt.split('\nCUM CAN LAM: ', 1)[1].split('\nSU KIEN: ', 1)[0])
            count = cluster['sentence_count']
            if bad_once is not None and cluster['cluster_id'] == 'C001' and not bad_once:
                bad_once.append(True)
                count -= 1
            eid = cluster['event_ids'][0]
            focuses = [f'ObservedDimension{j} {eid}' for j in range(count)]
            return json.dumps({'cluster_id': cluster['cluster_id'], 'ideas_by_event': {eid: focuses}}), 'model'
        return generate, calls

    def test_complete_plan_has_exactly_156_source_linked_slots_before_writer_starts(self):
        generate, calls = self.generator()
        text, _ = run_two_stage_planning(self.chunks, 600, 3120, 'plan', generate,
                                         json.loads, self.normalize, defer_visual_selection=True)
        raw = json.loads(text)
        self.assertEqual(raw['sentence_budget_mode'], 'exact_target')
        self.assertEqual(raw['target_sentence_count'], 156)
        self.assertEqual(sum(c['sentence_count'] for c in raw['selected_event_clusters']), 156)
        for c in raw['selected_event_clusters']:
            self.assertEqual(len(raw['cluster_details'][c['cluster_id']]['sentences']), c['sentence_count'])
        plan = self.normalize(text)
        self.assertEqual(sum(len(p) for p in plan['events_by_part']), 156)
        self.assertEqual(plan['scene_contract']['covered_source_event_count'], 30)
        # Each writer part inherits its exact share; it cannot return fewer rows.
        written = 0
        for part in range(1, 4):
            context = build_part_story_context(plan, part)
            required = context['required_events']
            response = coerce_single_part_response(json.dumps({'scenes': [
                {'event_id': e['event_id'], 'tts': 'Noi dung tu su kien da quan sat.'} for e in required]}),
                part, 200, 1040, locked_event_by_id={e['event_id']: e for e in required})
            validate_single_part_response(response, part, 1040, required_event_ids=[e['event_id'] for e in required],
                                          required_scene_count=len(required))
            written += len(json.loads(response)['parts'][0]['scenes'])
        self.assertEqual(written, 156)

    def test_short_cluster_draft_is_replanned_locally_not_rebalanced_into_other_clusters(self):
        generate, calls = self.generator(bad_once=[])
        saved = []
        text, _ = run_two_stage_planning(self.chunks, 600, 3120, 'plan', generate, json.loads,
                                         self.normalize, defer_visual_selection=True, save=saved.append)
        self.assertEqual(sum(label.endswith('C000') for label in calls), 1)
        self.assertEqual(sum(label.endswith('C001') for label in calls), 2)
        self.assertEqual(sum(len(d['sentences']) for d in json.loads(text)['cluster_details'].values()), 156)
        error = saved[-1]['detail_raw_responses']['C001'][0]['error']
        self.assertIn('DETAIL_SENTENCE_COUNT', error)
        self.assertIn('expected=', error)
        self.assertIn('actual=', error)


if __name__ == '__main__':
    unittest.main()
