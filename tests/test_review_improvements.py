import copy
import os
import subprocess
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

from edge_tts import SubMaker
from edge_tts.submaker import Subtitle

from story_planning import refine_story_clusters, validate_cluster_detail
from recap_pipeline import expand_evidence_to_visual_slots
from tts_engine import (_exact_cue_partitions, plan_tts_pause_cuts, _compact_group_pauses,
                        parse_sentences_and_visual_beats, subtitle_phrase_cues, get_ffmpeg_exe,
                        process_tts_group, get_audio_duration)


def cue(start, end, content):
    return Subtitle(index=0, start=timedelta(seconds=start), end=timedelta(seconds=end), content=content)


class ReviewImprovementsTests(unittest.TestCase):
    def test_reviewed_visual_slots_never_sample_unverified_gap_between_shots(self):
        event = {'evidence_id': 'EV1', 'event': 'Action then result',
                 'start_seconds': 100, 'end_seconds': 205, 'visual_review_version': 1,
                 'visual_supported_facts': ['First action happens', 'Different result appears'],
                 'visual_sources': [
                     {'start_seconds': 100, 'end_seconds': 105, 'verified_visual_shot': True},
                     {'start_seconds': 200, 'end_seconds': 205, 'verified_visual_shot': True},
                 ]}
        slots = expand_evidence_to_visual_slots([event], 2)
        self.assertEqual([[source['start_seconds'] for source in slot['visual_sources']]
                          for slot in slots], [[100], [200]])
        self.assertEqual([slot['end_seconds'] for slot in slots], [105, 205])

    def test_large_chapter_cluster_is_split_without_omitting_event_ownership(self):
        events = [{'evidence_id': f'EV{i}', 'event': f'Action {i}', 'cause': f'Cause {i}',
                   'consequence': f'Result {i}', 'start_seconds': i * 30,
                   'end_seconds': i * 30 + 20} for i in range(20)]
        ids = [event['evidence_id'] for event in events]
        plan = {'selected_event_clusters': [{'cluster_id': 'C001', 'event_ids': ids,
                 'sentence_count': 66, 'summary': 'An entire chapter', 'required_context_ids': []}]}
        original = copy.deepcopy(plan)
        refined = refine_story_clusters(plan, events)
        clusters = refined['selected_event_clusters']
        self.assertGreater(len(clusters), 3)
        self.assertTrue(all(len(c['event_ids']) <= 4 for c in clusters))
        self.assertEqual([key for c in clusters for key in c['event_ids']], ids)
        self.assertEqual(sum(c['sentence_count'] for c in clusters), 66)
        self.assertEqual(len({c['cluster_id'] for c in clusters}), len(clusters))
        self.assertEqual(plan, original)

    def test_small_cluster_is_preserved(self):
        plan = {'selected_event_clusters': [{'cluster_id': 'C1', 'event_ids': ['EV1'], 'sentence_count': 1}]}
        self.assertEqual(refine_story_clusters(plan, [{'evidence_id': 'EV1', 'event': 'Action'}]), plan)

    def test_detail_cannot_add_a_second_sentence_by_changing_only_padding(self):
        detail = {'cluster_id': 'C1', 'sentences': [
            {'source_event_ids': ['EV1'], 'focus': 'Nobita thu don hanh ly trong phong'},
            {'source_event_ids': ['EV1'], 'focus': 'Nobita thu don hanh ly ngay trong phong'},
        ]}
        with self.assertRaisesRegex(ValueError, 'DETAIL_REPEATED_FOCUS'):
            validate_cluster_detail(detail, {'cluster_id': 'C1', 'event_ids': ['EV1'], 'sentence_count': 2})

    def test_real_render_must_not_move_backwards_source_range_to_another_action(self):
        rows = ['[TIME=01:20-01:25] Action A.', '[TIME=01:15-01:19] Action B.']
        with self.assertRaisesRegex(ValueError, 'SOURCE_RANGE_OVERLAP'):
            parse_sentences_and_visual_beats(rows, strict_source_ranges=True)

    def test_exact_partitions_use_tokens_not_number_of_coarse_cues(self):
        cues = [cue(0, 1, 'Nobita'), cue(1, 2, 'bo nha.'), cue(3, 4, 'Doraemon di theo.')]
        self.assertEqual(_exact_cue_partitions(cues, ['Nobita bo nha.', 'Doraemon di theo.']), [(0, 2), (2, 3)])

    def test_long_pause_is_shortened_only_inside_detected_silence(self):
        cues = [cue(0, 1, 'Cau mot.'), cue(2, 3, 'Cau hai.')]
        cuts = plan_tts_pause_cuts(cues, ['Cau mot.', 'Cau hai.'], [(1.05, 1.95)])
        self.assertEqual(len(cuts), 1)
        self.assertGreaterEqual(cuts[0][0], 1.13)
        self.assertLessEqual(cuts[0][1], 1.87)
        self.assertAlmostEqual(cuts[0][1] - cuts[0][0], .72)
        self.assertEqual(plan_tts_pause_cuts(cues, ['Cau mot.', 'Cau hai.'], []), [])
        self.assertEqual(plan_tts_pause_cuts(cues, ['Wrong text.', 'Cau hai.'], [(1.05, 1.95)]), [])

    def test_pause_compaction_retimes_cues_without_changing_original(self):
        submaker = SubMaker()
        submaker.cues = [cue(0, 1, 'Cau mot.'), cue(2, 3, 'Cau hai.')]
        with mock.patch('tts_engine.subprocess.run', return_value=mock.Mock(
                returncode=0, stderr='silence_start: 1.05\nsilence_end: 1.95')):
            spans, adjusted, duration = _compact_group_pauses('raw.mp3', submaker,
                ['Cau mot.', 'Cau hai.'], 3.18, 'ffmpeg')
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(duration, 2.46)
        self.assertAlmostEqual(adjusted.cues[1].start.total_seconds(), 1.28)
        self.assertEqual(submaker.cues[1].start.total_seconds(), 2)

    def test_request_join_inside_one_row_is_compacted_once(self):
        first = 'Nobita ' * 80 + 'di, '
        second = 'Doraemon di theo.'
        text = first + second
        submaker = SubMaker()
        submaker.cues = [cue(0, 1, first), cue(2, 3, second)]
        with mock.patch('tts_engine._split_long_tts_text', return_value=[first, second]), \
                mock.patch('tts_engine.subprocess.run', return_value=mock.Mock(
                    returncode=0, stderr='silence_start: 1.05\nsilence_end: 1.95')):
            spans, adjusted, duration = _compact_group_pauses(
                'raw.mp3', submaker, [text], 3.18, 'ffmpeg')
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(adjusted.cues[1].start.total_seconds(), 1.26)
        self.assertAlmostEqual(duration, 2.44)

    def test_leading_silence_keeps_guard_before_first_word(self):
        submaker = SubMaker()
        submaker.cues = [cue(1, 2, 'Nobita di.')]
        with mock.patch('tts_engine.subprocess.run', return_value=mock.Mock(
                returncode=0, stderr='silence_start: 0\nsilence_end: 0.95')):
            spans, adjusted, duration = _compact_group_pauses(
                'raw.mp3', submaker, ['Nobita di.'], 2.18, 'ffmpeg')
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0][0], .87)
        self.assertAlmostEqual(adjusted.cues[0].start.total_seconds(), .13)
        self.assertAlmostEqual(duration, 1.31)

    def test_subtitle_phrases_preserve_words_and_measured_cue_times(self):
        words = ['Nobita', 'quyet', 'dinh', 'bo', 'nha', 'di', 'de', 'tim', 'kiem', 'tu', 'do.']
        cues = [cue(i * .2, (i + 1) * .2, word) for i, word in enumerate(words)]
        groups = subtitle_phrase_cues(cues)
        self.assertGreater(len(groups), 1)
        self.assertEqual([word[2] for group in groups for word in group], words)
        self.assertTrue(all(len(group) <= 9 for group in groups))
        self.assertEqual(groups[-1][-1][1], 2.2)

    def test_coarse_sentence_cue_becomes_short_readable_phrases(self):
        text = 'Nobita bi me mang nen quyet dinh bo nha va tim den Doraemon de chuan bi cho hanh trinh moi.'
        groups = subtitle_phrase_cues([cue(1, 6, text)])
        self.assertGreater(len(groups), 2)
        self.assertEqual(' '.join(word[2] for group in groups for word in group), text)
        self.assertEqual(groups[0][0][0], 1)
        self.assertEqual(groups[-1][-1][1], 6)

    def test_real_ffmpeg_pause_edit_preserves_speech_duration_and_cue_clock(self):
        ffmpeg = get_ffmpeg_exe()
        with tempfile.TemporaryDirectory(prefix='review_audio_') as folder:
            raw_path = os.path.join(folder, 'raw.wav')
            generated = subprocess.run([
                ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1:sample_rate=24000',
                '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono:d=1',
                '-f', 'lavfi', '-i', 'sine=frequency=660:duration=1:sample_rate=24000',
                '-filter_complex', '[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]',
                '-map', '[out]', raw_path,
            ], capture_output=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            submaker = SubMaker()
            submaker.cues = [cue(0, 1, 'Cau mot.'), cue(2, 3, 'Cau hai.')]
            async def generate(text, voice, rate, output_path, *args, **kwargs):
                import shutil
                shutil.copyfile(raw_path, output_path)
                return submaker
            import asyncio
            with (mock.patch('tts_engine._load_cached_tts', return_value=None),
                  mock.patch('tts_engine._save_cached_tts'),
                  mock.patch('tts_engine.generate_tts_chunk_with_timing', side_effect=generate)):
                _, results, master = asyncio.run(process_tts_group(
                    0, [(0, 'Cau mot.'), (1, 'Cau hai.')], 'voice', '+0%',
                    folder, asyncio.Semaphore(1), ffmpeg))
            duration = get_audio_duration(master)
            self.assertLess(duration, 2.45)
            self.assertGreater(duration, 2.25)
            self.assertAlmostEqual(sum(row[4] for row in results), duration, places=5)
            self.assertAlmostEqual(results[1][3].cues[0].end.total_seconds()
                                   + results[0][4], 2.28, delta=.12)


if __name__ == '__main__':
    unittest.main()
