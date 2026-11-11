import copy
import json
import unittest

from plot_coverage import plot_coverage, missing_plot_windows, merge_plot_events
from narrative_contract import allocate_sentences
from grounded_pipeline import bind_observations, shot_index
from recap_pipeline import normalize_chunk_evidence


class PlotCoverageTests(unittest.TestCase):
    def chunks(self):
        return [{'start_seconds': 0, 'end_seconds': 120, 'primary_end_seconds': 120,
            'events': [dict(evidence_id='EV1', start_seconds=0, end_seconds=6,
                grounded_facts=[dict(fact_id='F1', text='Cậu mở cửa.', shot_ids=['SHOT_000001'])],
                visual_sources=[dict(shot_id='SHOT_000001', start_seconds=0, end_seconds=6)])]}]

    def outline(self):
        return {'story_beats': [dict(beat_id='GB001', start_seconds=0, end_seconds=20, importance='main'),
            dict(beat_id='GB002', start_seconds=20, end_seconds=100, importance='main'),
            dict(beat_id='GB003', start_seconds=100, end_seconds=120, story_role='ending')]}

    def test_temporal_overlap_does_not_claim_semantic_verification(self):
        report = plot_coverage(self.outline(), self.chunks(), 120)
        self.assertEqual(report['milestones'][0]['status'], 'candidate_sources')
        self.assertFalse(report['all_milestones_verified'])
        self.assertEqual(report['missing_main_ids'], ['GB002'])

    def test_review_windows_are_bounded_and_skip_ending(self):
        report = plot_coverage(self.outline(), self.chunks(), 120)
        windows = missing_plot_windows(report)
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0]['start_seconds'], windows[0]['end_seconds']), (20, 80))
        self.assertEqual(missing_plot_windows(report, budget=0), [])
        reviewed = plot_coverage(self.outline(), self.chunks(), 120, reviewed=['GB002'])
        self.assertEqual(reviewed['milestones'][1]['status'], 'reviewed_not_observed')
        self.assertFalse(reviewed['all_milestones_verified'])

    def test_partial_beat_can_trigger_targeted_review_even_with_existing_facts(self):
        chunks = self.chunks()
        chunks[0]['plot_gaps'] = [dict(beat_id='GB001', reason='chưa thấy kết quả', shot_ids=['SHOT_000002'])]
        report = plot_coverage(self.outline(), chunks, 120)
        self.assertEqual(report['milestones'][0]['status'], 'partial_sources')
        self.assertEqual(missing_plot_windows(report)[0]['start_seconds'], 6)
        after = plot_coverage(self.outline(), chunks, 120, reviewed=['GB001'])
        self.assertEqual(after['milestones'][0]['status'], 'reviewed_unresolved')

    def test_observation_beat_links_and_gap_shots_survive_normalization(self):
        raw = {'characters': [], 'plot_gaps': [
            dict(beat_id='GB001', reason='unclear', shot_ids=['SHOT_000001']),
            dict(beat_id='GB002', reason='invalid source', shot_ids=['SHOT_999999'])],
            'events': [dict(event='Cậu mở cửa', content_kind='story', observations=[
                dict(text='Cậu mở cửa', shot_ids=['SHOT_000001'], story_beat_ids=['GB001', 'invalid'])])]}
        bound = bind_observations(raw, shot_index(120), 0, 30, 1)
        self.assertEqual(len(bound['plot_gaps']), 1)
        normalized = normalize_chunk_evidence(json.dumps(bound, ensure_ascii=False), 0, 30, chunk_number=1)
        fact = normalized['events'][0]['grounded_facts'][0]
        self.assertEqual(fact['story_beat_ids'], ['GB001'])
        self.assertEqual(len(normalized['plot_gaps']), 1)

    def test_restoring_supplement_does_not_add_chapters_or_duplicate_facts(self):
        chunks = self.chunks()
        extra = dict(evidence_id='EVNEW', start_seconds=24, end_seconds=30,
            grounded_facts=[dict(fact_id='FNEW', text='Cậu chạy ra ngoài.', shot_ids=['SHOT_000005'])],
            visual_sources=[dict(shot_id='SHOT_000005', start_seconds=24, end_seconds=30)])
        original = copy.deepcopy(extra)
        merge_plot_events(chunks, [extra])
        merge_plot_events(chunks, [extra])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]['events']), 2)
        self.assertEqual(extra, original)
        report = plot_coverage(self.outline(), chunks, 120)
        self.assertEqual(report['missing_main_ids'], [])

    def test_ending_gets_one_sentence_without_lowering_total_or_capping_climax(self):
        events = [dict(event_id=f'E{i}', story_role=role,
            observed_facts=[dict(fact_id=f'F{i}', text='Cậu mở cửa.', shot_ids=['SHOT_000001'])])
            for i, role in enumerate(['development', 'climax', 'resolution', 'ending', 'ending'])]
        plan = dict(events_by_part=[events], scene_contract={}, situation_plan_version=4)
        result = allocate_sentences(plan, 30, 600)
        counts = [e['sentence_quota'] for e in result['events_by_part'][0]]
        self.assertEqual(sum(counts), 30)
        self.assertEqual(counts[-2:], [1, 1])
        self.assertGreater(counts[1], counts[0])
        self.assertNotIn('sentence_quota', events[0])
