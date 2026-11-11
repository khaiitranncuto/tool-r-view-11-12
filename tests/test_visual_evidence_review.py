import copy
import unittest

from visual_evidence_review import review_visual_evidence, visual_review_prompt
from recap_pipeline import event_narration_capacity, event_narration_foci


class VisualEvidenceReviewTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{'events': [{
            'evidence_id': 'EV1', 'start_seconds': 100, 'end_seconds': 110,
            'event': 'Nobita thu don hanh ly.', 'action': 'Nobita thu don hanh ly.',
            'cause': 'Cau muon tu lap.', 'consequence': 'Cau roi nha.',
            'visual_sources': [{'start_seconds': 102, 'end_seconds': 107,
                                'event': 'Thu don hanh ly', 'verified_visual_shot': True}],
        }]}]

    def supported(self, window):
        return {'review_id': window['review_id'], 'supported': True,
                'content_kind': 'story', 'observed_action': 'Nobita thu don hanh ly.',
                'supported_facts': ['Nobita thu don hanh ly.'],
                'shots': [{'start_offset': 2, 'end_offset': 5}]}

    def test_local_offsets_become_source_times_and_only_confirmed_facts_count(self):
        original = copy.deepcopy(self.chunks)
        reviewed = review_visual_evidence(self.chunks, 1000,
            lambda windows: {'reviews': [self.supported(w) for w in windows]})
        event = reviewed[0]['events'][0]
        self.assertEqual(event['start_seconds'], 102)
        self.assertEqual(event['end_seconds'], 105)
        self.assertEqual(event['visual_sources'][0]['visual_match_tier'], 'short_clip_verified')
        self.assertEqual(event_narration_foci(event), ['Nobita thu don hanh ly.'])
        self.assertEqual(event_narration_capacity(event), 1)
        self.assertEqual(self.chunks, original)

    def test_credits_are_rejected_even_if_model_says_supported(self):
        windows_seen, saved = [], []
        def review(windows):
            windows_seen.append(windows)
            row = self.supported(windows[0])
            row['content_kind'] = 'credits'
            return {'reviews': [row]}
        with self.assertRaisesRegex(ValueError, 'VISUAL_EVIDENCE_UNCONFIRMED: EV1:1'):
            review_visual_evidence(self.chunks, 1000, review, save=saved.append)
        self.assertEqual(len(windows_seen), 2)
        self.assertGreater(windows_seen[1][0]['clip_end'] - windows_seen[1][0]['clip_start'],
                           windows_seen[0][0]['clip_end'] - windows_seen[0][0]['clip_start'])
        self.assertLessEqual(windows_seen[1][0]['clip_end'] - windows_seen[1][0]['clip_start'], 60)
        self.assertEqual(saved[-1]['unresolved'], ['EV1:1'])

    def test_incorrect_ranges_can_be_corrected_inside_wider_clip(self):
        calls = []
        def review(windows):
            calls.append(windows)
            if len(calls) == 1:
                return {'reviews': []}
            row = self.supported(windows[0])
            row['shots'] = [{'start_offset': 30, 'end_offset': 34}]
            return {'reviews': [row]}
        event = review_visual_evidence(self.chunks, 1000, review)[0]['events'][0]
        self.assertEqual(event['start_seconds'], 112)
        self.assertEqual(event['end_seconds'], 116)

    def test_missing_invalid_and_hallucinated_offsets_do_not_pass(self):
        for shots in ([], [{'start_offset': -1, 'end_offset': 3}],
                      [{'start_offset': 0, 'end_offset': 1000}],
                      [{'start_offset': float('nan'), 'end_offset': 4}], 'invalid'):
            def review(windows):
                row = self.supported(windows[0])
                row['shots'] = shots
                return {'reviews': [row]}
            with self.subTest(shots=shots), self.assertRaisesRegex(ValueError, 'UNCONFIRMED'):
                review_visual_evidence(self.chunks, 1000, review)

    def test_checkpoint_keeps_successful_shots_across_transport_failure(self):
        events = self.chunks[0]['events']
        for index in range(2, 8):
            event = copy.deepcopy(events[0])
            event['evidence_id'] = f'EV{index}'
            events.append(event)
        saved, calls = [], []
        def review(windows):
            calls.append(windows)
            if len(calls) == 2:
                raise RuntimeError('503 transport')
            return {'reviews': [self.supported(w) for w in windows]}
        with self.assertRaisesRegex(RuntimeError, '503'):
            review_visual_evidence(self.chunks, 1000, review, save=saved.append)
        self.assertEqual(len(saved[-1]['confirmed']), 6)
        resumed = []
        def resume(windows):
            resumed.extend(w['review_id'] for w in windows)
            return {'reviews': [self.supported(w) for w in windows]}
        review_visual_evidence(self.chunks, 1000, resume, checkpoint=saved[-1])
        self.assertEqual(resumed, ['EV7:1'])

    def test_verifier_cannot_invent_facts_to_inflate_capacity(self):
        def review(windows):
            row = self.supported(windows[0])
            row['supported_facts'] = ['Nobita tro thanh anh hung.']
            return {'reviews': [row]}
        event = review_visual_evidence(self.chunks, 1000, review)[0]['events'][0]
        self.assertEqual(event['visual_supported_facts'], [])
        self.assertEqual(event_narration_capacity(event), 1)

    def test_prompt_rejects_movie_memory_and_credits(self):
        prompt = visual_review_prompt([{'review_id': 'EV1:1', 'event': 'Fact',
                                       'action': 'Action', 'facts': []}])
        self.assertIn('khong dung tri nho phim', prompt)
        self.assertIn('credits', prompt)

    def test_replacing_source_file_invalidates_previous_visual_confirmation(self):
        saved = []
        review_visual_evidence(self.chunks, 1000,
            lambda windows: {'reviews': [self.supported(w) for w in windows]},
            source_signature={'size': 100, 'mtime_ns': 1}, save=saved.append)
        calls = []
        def review(windows):
            calls.extend(windows)
            return {'reviews': [self.supported(w) for w in windows]}
        review_visual_evidence(self.chunks, 1000, review, checkpoint=saved[-1],
                              source_signature={'size': 100, 'mtime_ns': 2})
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
