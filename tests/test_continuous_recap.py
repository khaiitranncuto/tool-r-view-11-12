import json
import unittest

from grounded_pipeline import bind_observations, shot_index, non_story
from recap_pipeline import parse_continuous_recap_draft, build_continuous_recap_draft_prompt


class ContinuousRecapTests(unittest.TestCase):
    def test_intro_is_excluded_without_excluding_real_opening_scene(self):
        events = []
        for index, (description, sequence) in enumerate([
            ('Logo hãng phim', 'studio_logo'),
            ('Montage giới thiệu nhân vật bay trên trời', 'opening_montage'),
            ('Nhạc mở đầu với nhóm bạn', 'title_sequence'),
            ('Nhóm bạn bay trên trời', 'opening_montage'),
            ('Cậu bé săn cá rồi trở về làng', 'narrative'),
        ]):
            events.append({'content_kind': 'story', 'event': description,
                           'sequence_type': sequence,
                           'observations': [{'text': description,
                                             'shot_ids': [f'SHOT_{index + 1:06d}']}]})
        result = bind_observations({'events': events}, shot_index(30), 0, 30, 1)
        self.assertEqual(len(result['events']), 1)
        self.assertIn('săn cá', result['events'][0]['event'])
        self.assertFalse(non_story('Cậu bé giới thiệu bạn mình với dân làng'))

    def test_draft_preserves_paragraphs_and_rejects_scene_response(self):
        text = json.dumps({'paragraphs': ['Cậu bé mang cá về.', 'Trong khi đó, Nobita bị mẹ mắng.']})
        self.assertEqual(parse_continuous_recap_draft(text),
                         'Cậu bé mang cá về.\n\nTrong khi đó, Nobita bị mẹ mắng.')
        for invalid in ({'scenes': []}, {'paragraphs': []}, {'paragraphs': ['']},
                        {'paragraphs': [{'tts': 'wrong schema'}]}):
            with self.assertRaisesRegex(ValueError, 'CONTINUOUS_RECAP_DRAFT_INVALID'):
                parse_continuous_recap_draft(json.dumps(invalid))

    def test_draft_gets_previous_and_next_context_without_visual_quota(self):
        prompt = build_continuous_recap_draft_prompt(
            [{'event_id': 'E1', 'narration_focus': 'Cậu bé về làng',
              'observed_facts': [{'text': 'Làng bị phá hủy'}]}],
            {'previous_part_tail': [{'tts': 'Cậu bắt được cá.'}],
             'next_batch_context': [{'event': 'Nobita bị mẹ mắng'}]})
        self.assertIn('Cậu bắt được cá.', prompt)
        self.assertIn('Nobita bị mẹ mắng', prompt)
        self.assertIn('Làng bị phá hủy', prompt)
        self.assertNotIn('visual_id', prompt)


if __name__ == '__main__':
    unittest.main()
