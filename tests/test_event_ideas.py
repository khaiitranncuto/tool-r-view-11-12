import copy
import json
import unittest

from event_ideas import (inspect_event_ideas, event_ideas_to_detail,
                         apply_event_idea_repairs, event_ideas_repair_prompt)
from story_planning import expand_event_plans, run_two_stage_planning, validate_cluster_detail
from recap_pipeline import normalize_story_plan_from_evidence
from detail_repair import inspect_detail_rows, build_detail_repair_prompt


class EventIdeasTests(unittest.TestCase):
    def setUp(self):
        self.cluster = {'cluster_id': 'C2', 'event_ids': ['EV2', 'EV3'], 'sentence_count': 4}

    def test_backend_owns_order_and_normalizes_ids_strings_and_exact_duplicates(self):
        payload = {'ideas_by_event': {' ev3 ': ' Ket qua 3 ', 'ev2': ['Y mot', '  Y   mot ', 'Y hai']}}
        original = copy.deepcopy(payload)
        detail = event_ideas_to_detail(payload, self.cluster)
        self.assertEqual(detail['sentences'], [
            {'source_event_ids': ['EV2'], 'focus': 'Y mot'},
            {'source_event_ids': ['EV2'], 'focus': 'Y hai'},
            {'source_event_ids': ['EV3'], 'focus': 'Ket qua 3'},
        ])
        self.assertEqual(payload, original)

    def test_reports_unknown_id_missing_id_and_precise_empty_focus(self):
        payload = {'ideas_by_event': {'EV2': ['Dung', ''], 'TYPO': ['Unknown']}}
        draft, issues = inspect_event_ideas(payload, self.cluster)
        self.assertIn({'code': 'nonempty_text_required', 'path': 'ideas_by_event.EV2[2]',
                       'event_id': 'EV2', 'focus_index': 2}, issues)
        self.assertIn('missing_event_id', [i['code'] for i in issues])
        self.assertIn('unknown_event_id', [i['code'] for i in issues])
        with self.assertRaisesRegex(ValueError, 'ideas_by_event.EV2'):
            event_ideas_to_detail(draft, self.cluster)

    def test_structural_patch_keeps_good_events_and_good_ideas(self):
        draft, issues = inspect_event_ideas({'ideas_by_event': {'EV2': ['Good', ''], 'EV3': ['Locked']}}, self.cluster)
        patched = apply_event_idea_repairs(draft, {'ideas_by_event': {'ev2': ['Good', 'New']}}, self.cluster, issues)
        self.assertEqual(patched['ideas_by_event']['EV3'], ['Locked'])
        self.assertEqual(patched['ideas_by_event']['EV2'], ['Good', 'New'])
        echo = apply_event_idea_repairs(draft, {'ideas_by_event': {'EV2': ['Good', 'New'], 'EV3': ['Locked']}}, self.cluster, issues)
        self.assertEqual(echo, patched)
        with self.assertRaisesRegex(ValueError, 'CHANGED_VALID_FOCUS'):
            apply_event_idea_repairs(draft, {'ideas_by_event': {'EV2': ['Replacement']}}, self.cluster, issues)
        with self.assertRaisesRegex(ValueError, 'PATCH_IDS'):
            apply_event_idea_repairs(draft, {'ideas_by_event': {'EV2': ['Good', 'New'], 'EV3': ['Changed']}}, self.cluster, issues)

    def test_unknown_id_requires_explicit_removal_not_guessing_a_replacement(self):
        draft, issues = inspect_event_ideas({'ideas_by_event': {'EV2': ['Good'], 'EV3': ['Locked'], 'OTHER': ['X']}}, self.cluster)
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_IDS'):
            apply_event_idea_repairs(draft, {'ideas_by_event': {}}, self.cluster, issues)
        patched = apply_event_idea_repairs(draft, {'ideas_by_event': {}, 'discard_unknown_ids': ['other']}, self.cluster, issues)
        self.assertEqual(patched['ideas_by_event'], {'EV2': ['Good'], 'EV3': ['Locked']})

    def test_conflicting_aliases_of_one_id_are_not_silently_overwritten(self):
        _, issues = inspect_event_ideas({'ideas_by_event': {'EV2': ['One'], ' ev2 ': ['Two'], 'EV3': ['Three']}}, self.cluster)
        self.assertIn('conflicting_normalized_id', [i['code'] for i in issues])

    def test_legacy_single_owner_groups_can_be_merged_without_semantic_guessing(self):
        detail = expand_event_plans({'cluster_id': 'C2', 'event_plans': [
            {'source_event_ids': ' ev2 ', 'focuses': 'Y mot'},
            {'source_event_ids': ['EV3'], 'focuses': ['Y ba']},
            {'source_event_ids': ['EV2', 'ev2'], 'focuses': ['Y mot', 'Y hai']},
        ]}, self.cluster)
        self.assertEqual([row['focus'] for row in detail['sentences']], ['Y mot', 'Y hai', 'Y ba'])
        validate_cluster_detail(detail, self.cluster, allow_flexible_count=True)

    def test_legacy_many_ids_many_ideas_is_explicitly_ambiguous(self):
        with self.assertRaisesRegex(ValueError, 'DETAIL_EVENT_AMBIGUOUS_MAPPING:group=1'):
            expand_event_plans({'cluster_id': 'C2', 'event_plans': [
                {'source_event_ids': ['EV2', 'EV3'], 'focuses': ['Y mot', 'Y hai']}]}, self.cluster)

    def test_flat_format_normalizes_without_reordering_prose(self):
        detail = expand_event_plans({'sentences': [
            {'source_event_ids': ' ev3 ', 'focus': ' Y  ba '},
            {'source_event_ids': ['ev2'], 'focus': 'Y hai'}]}, self.cluster)
        self.assertEqual(detail['sentences'][0], {'source_event_ids': ['EV3'], 'focus': 'Y ba'})
        with self.assertRaisesRegex(ValueError, 'DETAIL_EVENT_ORDER'):
            validate_cluster_detail(detail, self.cluster, allow_flexible_count=True)

    def test_malformed_flat_ids_can_be_diagnosed_and_repaired_without_crashing_inspector(self):
        detail = {'cluster_id': 'C2', 'sentences': [{'source_event_ids': 123, 'focus': None}]}
        issues = inspect_detail_rows(detail, self.cluster)
        self.assertIn('invalid_source_ids', [i['code'] for row in issues for i in row['issues']])
        prompt = build_detail_repair_prompt(detail, self.cluster, issues, [
            {'id': 'EV2', 'facts': ['F2']}, {'id': 'EV3', 'facts': ['F3']}])
        self.assertIn('missing_event_coverage', prompt)


class EventIdeaRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{'events': [{'evidence_id': f'EV{i}', 'start_seconds': (i - 1) * 40,
            'end_seconds': i * 40, 'event': f'Hanh dong {i}', 'cause': f'Nguyen nhan {i}',
            'consequence': f'Ket qua {i}'} for i in range(1, 4)]}]
        self.spine = {'boundary_1_seconds': 40, 'boundary_2_seconds': 80,
            'parts': [{'part': i, 'summary': f'Part {i}'} for i in range(1, 4)],
            'selected_event_clusters': [
                {'cluster_id': 'C1', 'event_ids': ['EV1'], 'sentence_count': 2},
                {'cluster_id': 'C2', 'event_ids': ['EV2', 'EV3'], 'sentence_count': 4}]}
        self.good = {f'EV{i}': [f'Nguyen nhan {i}', f'Ket qua {i}'] for i in range(1, 4)}

    def run_plan(self, generate, **kwargs):
        def wrapped(prompt, label):
            if '2/5:' in label:
                return json.dumps(self.spine), 'model'
            if '3/5:' in label:
                return json.dumps({'editorial_anchors': {}, 'event_importance': {}}), 'model'
            return generate(prompt, label), 'model'
        return run_two_stage_planning(self.chunks, 120, 120, 'plan', wrapped, json.loads,
            lambda text: normalize_story_plan_from_evidence(text, self.chunks, 120, 120),
            defer_visual_selection=True, **kwargs)

    def test_reported_ambiguous_group_is_repaired_by_event_without_replanning_completed_cluster(self):
        calls, saved = [], []
        def generate(prompt, label):
            calls.append(label)
            if label.endswith('C1'):
                return json.dumps({'ideas_by_event': {'EV1': self.good['EV1']}})
            if 'sua rieng ID loi' in label:
                self.assertIn('ID_CAN_SUA: ["EV2", "EV3"]', prompt)
                return json.dumps({'ideas_by_event': {k: self.good[k] for k in ['EV2', 'EV3']}})
            return json.dumps({'cluster_id': 'C2', 'event_plans': [
                {'source_event_ids': ['EV2', 'EV3'], 'focuses': ['Y 1', 'Y 2', 'Y 3', 'Y 4']}]})
        text, _ = self.run_plan(generate, save=saved.append)
        details = json.loads(text)['cluster_details']
        self.assertEqual(len(details['C1']['sentences']), 2)
        self.assertEqual(len(details['C2']['sentences']), 4)
        self.assertEqual(sum(label.endswith('C1') for label in calls), 1)
        raw = saved[-1]['detail_raw_responses']['C2']
        self.assertIn('AMBIGUOUS_MAPPING', raw[0]['error'])
        self.assertIn('event_plans', json.loads(raw[0]['text']))

    def test_invalid_json_is_saved_before_parse_and_completed_cluster_survives(self):
        saved = []
        def generate(prompt, label):
            return json.dumps({'ideas_by_event': {'EV1': self.good['EV1']}}) if label.endswith('C1') else 'broken JSON'
        with self.assertRaisesRegex(ValueError, 'DETAIL_INVALID:C2'):
            self.run_plan(generate, save=saved.append)
        self.assertEqual(len(saved[-1]['detail_raw_responses']['C2']), 3)
        self.assertEqual(saved[-1]['detail_raw_responses']['C2'][-1]['text'], 'broken JSON')
        self.assertIn('error', saved[-1]['detail_raw_responses']['C2'][-1])
        self.assertIn('C1', saved[-1]['details'])

    def test_interrupted_targeted_repair_resumes_only_bad_event_and_locks_good_event(self):
        saved = []
        def generate(prompt, label):
            if label.endswith('C1'):
                return json.dumps({'ideas_by_event': {'EV1': self.good['EV1']}})
            if 'sua rieng ID loi' in label:
                raise RuntimeError('503 disconnected')
            return json.dumps({'ideas_by_event': {'EV2': self.good['EV2'], 'EV3': [self.good['EV3'][0], '']}})
        with self.assertRaisesRegex(RuntimeError, '503'):
            self.run_plan(generate, save=saved.append)
        calls = []
        def resume(prompt, label):
            calls.append(label)
            self.assertIn('ID_CAN_SUA: ["EV3"]', prompt)
            self.assertIn('READ_ONLY: {"EV2"', prompt)
            return json.dumps({'ideas_by_event': {'EV3': self.good['EV3']}})
        text, _ = self.run_plan(resume, checkpoint=saved[-1])
        self.assertEqual(len(calls), 1)
        self.assertEqual([row['focus'] for row in json.loads(text)['cluster_details']['C2']['sentences']],
                         self.good['EV2'] + self.good['EV3'])


if __name__ == '__main__':
    unittest.main()
