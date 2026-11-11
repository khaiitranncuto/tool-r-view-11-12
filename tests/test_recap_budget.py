import json
from pathlib import Path
import tempfile
import unittest

from recap_budget import narration_budget, measured_narration_timing, normalize_voice_rate
from recap_pipeline import normalize_chunk_evidence, normalize_story_plan_from_evidence, required_tts_scene_count
from grounded_pipeline import (bind_observations, shot_index, lock_plan_visuals,
                               observation_cache_path, observation_prompt)
from story_planning import run_two_stage_planning


class RecapBudgetTests(unittest.TestCase):
    def test_selected_minutes_lock_the_agreed_number_of_sentences(self):
        for minutes, count in [(5, 68), (10, 135), (15, 203), (20, 270), (25, 338), (30, 405)]:
            budget = narration_budget(minutes)
            self.assertEqual(budget['voice_rate'], '+20%')
            self.assertEqual(budget['target_sentence_count'], count)
            self.assertEqual(required_tts_scene_count(budget['target_words']), count)

    def test_rate_changes_budget_and_invalid_rates_are_rejected(self):
        self.assertEqual(narration_budget(20, '+10%')['target_sentence_count'], 248)
        self.assertGreater(narration_budget(20, '+30%')['target_sentence_count'], 270)
        self.assertEqual(normalize_voice_rate('20'), '+20%')
        for rate in ('bad', 'nan', '+20.5%', '-100%'):
            with self.assertRaises(ValueError):
                normalize_voice_rate(rate)
        for minutes in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                narration_budget(minutes)

    def test_actual_tts_time_is_reported_without_claiming_exact_duration(self):
        target = narration_budget(20)
        timing = measured_narration_timing(270, [120.0] * 10, '+20%', target)
        self.assertEqual(timing['target_minutes'], 20)
        self.assertEqual(timing['actual_minutes'], 20)
        self.assertEqual(timing['measured_sentences_per_minute'], 13.5)
        self.assertTrue(target['estimated'])
        short = measured_narration_timing(270, [60.0] * 10, '+20%', target)
        self.assertEqual(short['actual_minutes'], 10)
        self.assertEqual(short['target_minutes'], 20)

    def test_fps_is_part_of_observation_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            source.write_bytes(b'video')
            one = observation_cache_path(directory, source, 0, 13, fps=1)
            two = observation_cache_path(directory, source, 0, 13, fps=2)
            self.assertNotEqual(one, two)
            self.assertEqual(two, observation_cache_path(directory, source, 0, 13, fps=2))

    def test_detailed_observation_prompt_preserves_evidence_rules(self):
        prompt = observation_prompt(1, 0, 480, 64)
        self.assertIn('CHI TIET TUNG SU KIEN', prompt)
        self.assertIn('phan ung cu the', prompt)
        self.assertIn('khong bo chi tiet that chi vi recap du kien ngan', prompt)
        self.assertIn('Chi dung shot_ids THUC SU DOC DUOC', prompt)

    def test_270_sentences_are_locked_from_real_facts_before_writing(self):
        facts = 300
        duration = facts * 6
        raw = {'events': [{'local_event_id': f'L{i}', 'event': f'Sự kiện {i}',
            'content_kind': 'story', 'observations': [
                {'text': [f'Người đàn ông {i} mở cửa căn nhà.',
                          f'Cô gái {i} chạy nhanh xuống bậc thềm.',
                          f'Chiếc xe {i} lao qua cầu rồi bốc cháy.'][j],
                 'shot_ids': [f'SHOT_{i*3+j+1:06d}']} for j in range(3)]}
                         for i in range(facts // 3)]}
        bound = bind_observations(raw, shot_index(duration), 0, duration, 1)
        chunks = [normalize_chunk_evidence(json.dumps(bound), 0, duration, chunk_number=1)]
        words = narration_budget(20)['target_words']
        def normalize(text):
            return normalize_story_plan_from_evidence(text, chunks, duration, words,
                                                     require_editorial_contract=True)
        calls = []
        def generate(prompt, label):
            calls.append(label)
            return json.dumps({'event_importance': {}, 'parts': [
                {'part': p, 'summary': f'Phần {p}'} for p in range(1, 4)]}), 'offline-model'
        text, _ = run_two_stage_planning(chunks, duration, words, 'test', generate,
            json.loads, normalize, lightweight=True, defer_visual_selection=True)
        plan = lock_plan_visuals(normalize(text), chunks)
        slots = [s for part in plan['events_by_part'] for s in part]
        self.assertEqual(len(slots), 270)
        self.assertTrue(all(s['observed_facts'] and s['visual_sources'] for s in slots))
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
