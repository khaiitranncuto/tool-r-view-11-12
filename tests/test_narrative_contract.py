import unittest
import json

from narrative_contract import allocate_sentences, sentences, words, validate_chapter, verify_script, constrain_repair_sentences


class NarrativeContractTests(unittest.TestCase):
    def test_quoted_punctuation_is_not_a_boundary_in_narrator_continuation(self):
        from narrative_contract import sentence_texts
        for left, right in [('"', '"'), ('“', '”'), ('«', '»'), ("'", "'")]:
            text = f'Giga Zombie quát {left}Đi đâu!{right} khi Kukuru chạy tới. Cậu tiếp tục leo lên.'
            units = sentence_texts(text)
            self.assertEqual(len(units), 2)
            self.assertEqual(' '.join(units), text)
            self.assertEqual(len(sentences(text)), 2)

    def test_nested_speech_sentences_and_ellipsis_remain_inside_quote(self):
        from narrative_contract import sentence_texts
        text = 'Gian hỏi “Ai đó? Có ai không?” rồi bước tới. Cậu gọi “Chờ...” khi thấy bạn.'
        self.assertEqual(len(sentence_texts(text)), 2)
        self.assertEqual(' '.join(sentence_texts(text)), text)
        self.assertEqual(len(sentence_texts('Cậu nói “Đi thôi!” Shizuka chạy theo.')), 2)
        self.assertEqual(len(sentence_texts('Cậu được 3.5 điểm.')), 1)
        self.assertEqual(len(sentence_texts('Cậu mở cửa. Cậu nói “Chờ')), 2)

    def test_v3_quota_uses_real_footage_instead_of_repeated_facts(self):
        plan = self.plan()
        plan['situation_plan_version'] = 3
        for i, part in enumerate(plan['events_by_part']):
            part[0]['observed_facts'] = [{}] * 10
            part[0]['visual_sources'] = [{'start_seconds': i * 40, 'end_seconds': i * 40 + (6 if i == 0 else 36)}]
        result = allocate_sentences(plan, 12, 240)
        rows = [p[0] for p in result['events_by_part']]
        self.assertEqual(sum(r['sentence_quota'] for r in rows), 12)
        self.assertLessEqual(rows[0]['sentence_quota'], 2)
        self.assertEqual(rows[0]['available_visual_seconds'], 6)

    def test_counts_actual_sentences_when_model_packs_them_in_one_item(self):
        text = 'A. B. C. D. E. F.'
        result = validate_chapter({'scenes': [{'event_id': 'E0111', 'sentences': [text]}]},
                                  [{'event_id': 'E0111', 'sentence_quota': 6}])
        self.assertEqual(result, [{'event_id': 'E0111', 'tts': text}])

    def test_best_candidate_is_selected_per_event_and_excess_is_joined(self):
        from narrative_contract import best_chapter_candidate
        events = [{'event_id': 'E1', 'sentence_quota': 2}, {'event_id': 'E2', 'sentence_quota': 1}]
        candidates = [
            {'scenes': [{'event_id': 'E1', 'sentences': ['A. B. C.']}, {'event_id': 'E2', 'sentences': ['D.']}]},
            {'scenes': [{'event_id': 'E1', 'sentences': ['A. B.']}, {'event_id': 'E2', 'sentences': ['D. E. F.']}]},
        ]
        result = best_chapter_candidate(candidates, events)
        self.assertEqual([r['tts'] for r in result], ['A. B.', 'D.'])
        result = best_chapter_candidate(candidates[:1], events)
        self.assertEqual(len(sentences(result[0]['tts'])), 2)
        self.assertEqual(words(result[0]['tts']), 3)

    def test_best_candidate_shortage_is_kept_for_targeted_repair(self):
        from narrative_contract import best_chapter_candidate
        events = [{'event_id': 'E1', 'sentence_quota': 3}]
        result = best_chapter_candidate([{'scenes': [{'event_id': 'E1', 'sentences': ['A. B.']}]}], events)
        self.assertEqual(result[0]['tts'], 'A. B.')
        with self.assertRaisesRegex(ValueError, 'NO_USABLE_CANDIDATE'):
            best_chapter_candidate([{'scenes': []}], events)

    def plan(self):
        return {'events_by_part': [[{'event_id': 'E1', 'observed_facts': [1, 2]}],
                                  [{'event_id': 'E2', 'observed_facts': [1, 2, 3]}],
                                  [{'event_id': 'E3', 'observed_facts': [1]}]], 'scene_contract': {}}

    def test_allocates_exact_sentence_total_independent_of_paragraph_count(self):
        plan = allocate_sentences(self.plan(), 9, 180)
        rows = [e for part in plan['events_by_part'] for e in part]
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(e['sentence_quota'] for e in rows), 9)
        self.assertGreater(rows[1]['sentence_quota'], rows[2]['sentence_quota'])
        self.assertEqual(sum(e['tts_target_words'] for e in rows), 180)
        with self.assertRaisesRegex(ValueError, 'SHORTAGE'):
            allocate_sentences(self.plan(), 281, 5620)

    def test_metadata_never_counts_as_words_or_sentences(self):
        text = '[TIME=00:01.50-00:06.00] Nobita về nhà. [BEAT=00:08.00-00:12.00] Cậu được 10. Nước cao 3.5 mét!'
        self.assertEqual(len(sentences(text)), 3)
        self.assertEqual(words('[TIME=00:00-00:06] Cậu về. [BEAT=00:06-00:12] Cậu ngủ.'), 4)

    def test_exact_281_is_required_not_95_paragraphs_or_200_sentences(self):
        sentence = ' '.join(['từ'] * 20) + '.'
        self.assertEqual(verify_script(' '.join([sentence] * 281), 281, 5620), (281, 5620))
        for count in (95, 200, 280, 282):
            with self.assertRaisesRegex(ValueError, 'SENTENCE_COUNT_MISMATCH'):
                verify_script(' '.join([sentence] * count), 281, 5620)
        self.assertEqual(verify_script(' '.join(['Cậu đi.'] * 281), 281, 5620), (281, 562))

    def test_chapter_checks_each_sentence_and_word_target(self):
        event = {'event_id': 'E1', 'sentence_quota': 2, 'tts_target_words': 40,
                 'tts_min_words': 30, 'tts_max_words': 50}
        sentence = ' '.join(['từ'] * 20) + '.'
        rows = validate_chapter({'scenes': [{'event_id': 'E1', 'sentences': [sentence, sentence]}]}, [event])
        self.assertEqual(len(sentences(rows[0]['tts'])), 2)
        with self.assertRaisesRegex(ValueError, 'SENTENCE_QUOTA'):
            validate_chapter({'scenes': [{'event_id': 'E1', 'sentences': [sentence]}]}, [event])
        with self.assertRaisesRegex(ValueError, 'SENTENCE_QUOTA'):
            validate_chapter({'scenes': [{'event_id': 'E1', 'sentences': [sentence + sentence, sentence]}]}, [event])

    def test_727_words_for_660_target_is_allowed_when_33_sentences_match(self):
        items = [' '.join(['từ'] * 22) + '.'] * 32 + [' '.join(['từ'] * 23) + '.']
        event = {'event_id': 'E1', 'sentence_quota': 33, 'tts_target_words': 660,
                 'tts_min_words': 495, 'tts_max_words': 825}
        rows = validate_chapter({'scenes': [{'event_id': 'E1', 'sentences': items}]}, [event])
        self.assertEqual(verify_script(rows[0]['tts'], 33, 660), (33, 727))

    def test_repair_preserves_all_words_and_returns_exact_original_quota(self):
        import re
        text = 'Doraemon lấy máy thời gian từ ngăn kéo. Cả nhóm chen chúc vào máy thời gian. Nhóm bạn bay trong không gian thời gian.'
        for quota in (1, 2, 3):
            result = constrain_repair_sentences(text, quota)
            self.assertEqual(len(sentences(result)), quota)
            self.assertEqual(re.findall(r'\w+', result), re.findall(r'\w+', text))
            self.assertEqual(constrain_repair_sentences(result, quota), result)
        with self.assertRaisesRegex(ValueError, 'không thêm câu đệm'):
            constrain_repair_sentences('Nobita về nhà.', 2)

    def test_final_normalization_closes_repair_and_cached_fallback_loophole(self):
        from recap_pipeline import clamp_payload_narrations, coerce_single_part_response
        text = 'Doraemon lấy máy thời gian. Cả nhóm bước vào. Cỗ máy bay đi.'
        payload = {'characters': [], 'parts': [{'scenes': [{'scene_id': 'S0015',
                   'start': '00:00', 'end': '00:06', 'sentence_quota': 1, 'tts': text}]}]}
        normalized = clamp_payload_narrations(payload)
        self.assertEqual(len(sentences(normalized['parts'][0]['scenes'][0]['tts'])), 1)
        event = {'event_id': 'E0015', 'scene_id': 'S0015', 'sentence_quota': 1,
                 'grounded_plan_locked': True, 'situation_narration': True,
                 'visual_sources': [{'start': '00:00', 'end': '00:06'}]}
        raw = json.dumps({'scenes': [{'event_id': 'E0015', 'tts': text}]})
        bound = json.loads(coerce_single_part_response(raw, 1, 10, 20,
                             locked_event_by_id={'E0015': event}))
        self.assertEqual(len(sentences(bound['parts'][0]['scenes'][0]['tts'])), 1)

    def test_paragraph_quota_survives_visual_binding_and_final_timeline(self):
        from recap_pipeline import coerce_single_part_response, merge_single_part_responses, payload_to_timeline
        responses = []
        sentence = ' '.join(['từ'] * 20) + '.'
        for i in range(3):
            event = {'event_id': f'E{i + 1:04d}', 'scene_id': f'S{i + 1:04d}',
                     'sentence_quota': 4, 'situation_narration': True, 'grounded_plan_locked': True,
                     'tts_target_words': 80, 'tts_min_words': 60, 'tts_max_words': 100,
                     'visual_sources': [{'start': '00:00', 'end': '00:06'}]}
            raw = json.dumps({'scenes': [{'event_id': event['event_id'], 'tts': ' '.join([sentence] * 4),
                                         'visual_links': [{'visual_id': 'V001', 'trigger_text': ''}]}]})
            response = coerce_single_part_response(raw, i + 1, 10, 80,
                             locked_event_by_id={event['event_id']: event})
            self.assertEqual(json.loads(response)['parts'][0]['scenes'][0]['sentence_quota'], 4)
            responses.append(response)
        merged = merge_single_part_responses(responses, [0, 10, 20, 30], 240)
        self.assertEqual(verify_script(payload_to_timeline(merged), 12, 240), (12, 240))


if __name__ == '__main__':
    unittest.main()
