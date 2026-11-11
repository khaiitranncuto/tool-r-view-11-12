import copy
import json
import unittest
from types import SimpleNamespace

from targeted_visual_review import repair_missing_visuals
from app import generate_gemini_text_with_recovery, generate_story_plan_with_model_fallback
from recap_pipeline import expand_evidence_to_visual_slots


class TargetedVisualReviewTests(unittest.TestCase):
    def event(self, index, has_shot=False, span=10):
        start = index * 100
        event = {'evidence_id': f'EV{index}', 'event': f'Hanh dong {index}',
                 'start_seconds': start, 'end_seconds': start + span}
        if has_shot:
            event['visual_sources'] = [{'start_seconds': start, 'end_seconds': start + 5,
                                        'event': f'Hanh dong {index}', 'verified_visual_shot': True}]
        return event

    def response(self, windows):
        return json.dumps({'repairs': [{'evidence_id': w['evidence_id'], 'found': True,
            'visual_shots': [{'start_offset': 30, 'end_offset': 34, 'description': 'Hanh dong da thay'}]}
            for w in windows]})

    def test_primary_pass_shots_are_used_without_any_video_review(self):
        original = [{'events': [self.event(1, True), self.event(2, True)]}]
        def review(*args):
            self.fail('Valid shots must never be re-uploaded')
        result, report = repair_missing_visuals(original, 1000, review)
        self.assertEqual(report, {'review_calls': 0, 'unlocated_ids': []})
        self.assertTrue(all(e['visual_ranges_locked'] for e in result[0]['events']))
        self.assertNotIn('visual_ranges_locked', original[0]['events'][0])

    def test_only_missing_shots_are_reviewed_and_valid_sources_do_not_change(self):
        chunks = [{'events': [self.event(1, True), self.event(2)]}]
        calls = []
        def review(events, windows, attempt):
            calls.extend(e['evidence_id'] for e in events)
            return self.response(windows)
        result, report = repair_missing_visuals(chunks, 1000, review)
        self.assertEqual(calls, ['EV2'])
        self.assertEqual(report['review_calls'], 1)
        self.assertEqual(result[0]['events'][0]['visual_sources'], chunks[0]['events'][0]['visual_sources'])

    def test_no_more_than_two_review_calls_for_many_missing_events(self):
        chunks = [{'events': [self.event(i) for i in range(1, 20)]}]
        calls = []
        def review(events, windows, attempt):
            calls.append([e['evidence_id'] for e in events])
            return '{"repairs":[]}'
        result, report = repair_missing_visuals(chunks, 3000, review)
        self.assertEqual(len(calls), 2)
        self.assertEqual([len(group) for group in calls], [6, 6])
        self.assertTrue(set(calls[0]).isdisjoint(calls[1]))
        self.assertEqual(len(report['unlocated_ids']), 19)
        self.assertEqual(result[0]['events'], [])
        self.assertEqual(len(result[0]['story_context']), 19)

    def test_targeted_repair_may_correct_wrong_original_time_within_attached_clip(self):
        chunks = [{'events': [self.event(2)]}]
        def review(events, windows, attempt):
            return json.dumps({'repairs': [{'evidence_id': 'EV2', 'found': True,
                'visual_shots': [{'start_offset': 40, 'end_offset': 44, 'description': 'Dung hanh dong'}]}]})
        result, report = repair_missing_visuals(chunks, 1000, review)
        self.assertEqual(report['unlocated_ids'], [])
        self.assertEqual(result[0]['events'][0]['start_seconds'], 215)
        self.assertEqual(result[0]['events'][0]['end_seconds'], 219)

    def test_multi_minute_summary_does_not_trigger_wrong_first_twenty_second_search(self):
        chunks = [{'events': [self.event(0, span=650)]}]
        def review(*args):
            self.fail('Broad context must not enter short-shot repair')
        result, report = repair_missing_visuals(chunks, 1000, review)
        self.assertEqual(report['review_calls'], 0)
        self.assertEqual(result[0]['events'], [])
        self.assertEqual(result[0]['story_context'][0]['evidence_id'], 'EV0')

    def test_resume_reuses_completed_repair_and_only_sends_unfinished_ids(self):
        chunks = [{'events': [self.event(i) for i in range(1, 9)]}]
        saved, calls = [], []
        def review(events, windows, attempt):
            calls.append(attempt)
            if attempt == 2:
                raise RuntimeError('timeout')
            return self.response(windows)
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            repair_missing_visuals(chunks, 2000, review, save=saved.append)
        self.assertEqual(saved[-1]['completed_calls'], 1)
        resumed = []
        def resume(events, windows, attempt):
            resumed.extend(e['evidence_id'] for e in events)
            return self.response(windows)
        _, report = repair_missing_visuals(chunks, 2000, resume, checkpoint=saved[-1])
        self.assertEqual(resumed, ['EV7', 'EV8'])
        self.assertEqual(report['review_calls'], 2)

    def test_locked_primary_shots_never_invent_images_from_source_gap(self):
        event = self.event(1, True, span=110)
        event['visual_sources'].append({'start_seconds': 200, 'end_seconds': 205,
                                       'verified_visual_shot': True})
        event['cause'] = 'Mot nguyen nhan rieng'
        event['consequence'] = 'Mot ket qua khac'
        prepared, _ = repair_missing_visuals([{'events': [event]}], 1000, lambda *args: None)
        slots = expand_evidence_to_visual_slots(prepared[0]['events'], 2)
        self.assertEqual([s['start_seconds'] for s in slots], [100, 200])

    def test_read_timed_out_retries_identical_request_once_even_with_one_attempt(self):
        calls, sleeps = [], []
        def generate(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError('The read operation timed out')
            return SimpleNamespace(text='ok')
        result = generate_gemini_text_with_recovery(
            SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
            'model', ['same clip'], None, 'review', max_attempts=1, sleep_fn=sleeps.append)
        self.assertEqual(result, 'ok')
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])

    def test_repeated_timeout_stops_after_two_calls(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            raise RuntimeError('The read operation timed out')
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            generate_gemini_text_with_recovery(
                SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
                'model', ['same clip'], None, 'review', max_attempts=5, sleep_fn=lambda _: None)
        self.assertEqual(len(calls), 2)

    def test_planning_does_not_multiply_timeout_retries_across_model_fallback(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs['model'])
            raise RuntimeError('The read operation timed out')
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            generate_story_plan_with_model_fallback(
                SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
                ['evidence'], None, 'plan', stream=False, sleep_fn=lambda _: None,
                model_sequence=('model-a', 'model-b', 'model-c'))
        self.assertEqual(calls, ['model-a', 'model-a'])


if __name__ == '__main__':
    unittest.main()
