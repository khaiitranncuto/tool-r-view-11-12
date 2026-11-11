import unittest

from transcript_pipeline import (
    _deduplicate_overlap,
    attach_transcript_to_evidence,
    format_transcript_window,
    merge_preferred_subtitles,
    parse_subtitle_text,
    promote_uncovered_transcript_events,
    promote_transcript_events_to_target,
)


class TranscriptPipelineTests(unittest.TestCase):
    def test_dialogue_is_attached_to_its_own_fact_not_whole_event(self):
        chunks = [{'events': [{'start_seconds': 0, 'end_seconds': 60,
            'visual_sources': [{'shot_id': 'A', 'start_seconds': 0, 'end_seconds': 6},
                              {'shot_id': 'B', 'start_seconds': 54, 'end_seconds': 60}],
            'grounded_facts': [{'fact_id': 'F1', 'shot_ids': ['A']},
                               {'fact_id': 'F2', 'shot_ids': ['B']}]}]}]
        result, _ = attach_transcript_to_evidence(chunks, [
            {'start_seconds': 1, 'end_seconds': 3, 'text': 'We must leave.'},
            {'start_seconds': 55, 'end_seconds': 57, 'text': 'We are safe now.'}])
        facts = result[0]['events'][0]['grounded_facts']
        self.assertEqual(facts[0]['spoken_text'], 'We must leave.')
        self.assertEqual(facts[1]['spoken_text'], 'We are safe now.')

    def test_srt_is_parsed_and_promotional_rows_are_removed(self):
        payload = parse_subtitle_text("""1
00:00:01,000 --> 00:00:03,000
Kukuru, mau chạy đi!

2
00:00:04,000 --> 00:00:06,000
Hãy subscribe cho kênh của mình nhé.
""", duration_seconds=10)
        self.assertEqual(payload["segment_count"], 1)
        self.assertEqual(payload["removed_promotional"], 1)
        self.assertEqual(payload["segments"][0]["text"], "Kukuru, mau chạy đi!")

    def test_srt_wins_overlap_and_whisper_fills_gap(self):
        merged = merge_preferred_subtitles(
            [{"start_seconds": 1, "end_seconds": 3, "text": "Tên tôi là Kukuru"}],
            [
                {"start_seconds": 1.1, "end_seconds": 3.1, "text": "Tên tôi là Cu ru"},
                {"start_seconds": 7, "end_seconds": 9, "text": "Chúng ta đi thôi"},
            ],
        )
        self.assertEqual([row["text"] for row in merged], [
            "Tên tôi là Kukuru", "Chúng ta đi thôi",
        ])

    def test_transcript_window_keeps_global_timestamps_and_both_ends(self):
        rows = [
            {
                "start_seconds": float(index * 10),
                "end_seconds": float(index * 10 + 2),
                "text": f"dialogue number {index} with enough words",
            }
            for index in range(20)
        ]
        text = format_transcript_window(
            rows, 20, 172, max_characters=1000
        )
        self.assertIn("[20.0-22.0]", text)
        self.assertIn("[170.0-172.0]", text)
        self.assertNotIn("[10.0-12.0]", text)

    def test_overlap_duplicate_is_kept_once(self):
        rows = _deduplicate_overlap([
            {"start_seconds": 10.0, "end_seconds": 12.0, "text": "Chúng ta phải đi ngay."},
            {"start_seconds": 10.2, "end_seconds": 12.2, "text": "Chúng ta phải đi ngay."},
            {"start_seconds": 13.0, "end_seconds": 14.0, "text": "Nhanh lên!"},
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["text"], "Nhanh lên!")

    def test_only_overlapping_dialogue_is_attached(self):
        evidence = [{"events": [
            {"start_seconds": 10.0, "end_seconds": 15.0, "event": "A mở cửa"},
            {"start_seconds": 30.0, "end_seconds": 35.0, "event": "A chạy"},
        ]}]
        result, count = attach_transcript_to_evidence(evidence, [
            {"start_seconds": 11.0, "end_seconds": 12.0, "text": "Mở cửa ra!"},
        ])
        self.assertEqual(count, 1)
        self.assertEqual(result[0]["events"][0]["spoken_text"], "Mở cửa ra!")
        self.assertEqual(result[0]["events"][1]["spoken_text"], "")

    def test_low_confidence_dialogue_is_not_promoted_as_evidence(self):
        evidence = [{"events": [{
            "start_seconds": 10.0, "end_seconds": 15.0, "event": "A mở cửa",
        }]}]
        result, count = attach_transcript_to_evidence(evidence, [{
            "start_seconds": 11.0, "end_seconds": 12.0,
            "text": "câu nhận dạng rất không chắc chắn", "confidence": 0.1,
        }])
        self.assertEqual(count, 0)
        self.assertEqual(result[0]["events"][0]["spoken_text"], "")

    def test_dialogue_in_long_visual_gap_becomes_bridge_event(self):
        evidence = [{
            "chunk_start_seconds": 0.0,
            "chunk_end_seconds": 120.0,
            "events": [
                {"start_seconds": 2.0, "end_seconds": 8.0, "event": "opening"},
                {"start_seconds": 105.0, "end_seconds": 112.0, "event": "result"},
            ],
        }]
        result, added = promote_uncovered_transcript_events(
            evidence,
            [{
                "start_seconds": 55.0,
                "end_seconds": 60.0,
                "text": "The group explains the next important destination clearly",
            }],
            120.0,
            maximum_uncovered_seconds=45.0,
        )
        bridges = [
            event for event in result[0]["events"]
            if event.get("source") == "transcript_bridge"
        ]
        self.assertEqual(added, 1)
        self.assertEqual(bridges[0]["spoken_text"], "The group explains the next important destination clearly")

    def test_short_interjection_does_not_become_bridge_event(self):
        evidence = [{
            "chunk_start_seconds": 0.0,
            "chunk_end_seconds": 120.0,
            "events": [
                {"start_seconds": 2.0, "end_seconds": 8.0, "event": "opening"},
                {"start_seconds": 105.0, "end_seconds": 112.0, "event": "result"},
            ],
        }]
        _, added = promote_uncovered_transcript_events(
            evidence,
            [{"start_seconds": 55.0, "end_seconds": 56.0, "text": "Oh no"}],
            120.0,
        )
        self.assertEqual(added, 0)

    def test_transcript_anchors_fill_target_without_inventing_actions(self):
        evidence = [{
            "start_seconds": 0.0,
            "end_seconds": 120.0,
            "events": [{
                "evidence_id": "EV00001", "start_seconds": 1.0,
                "end_seconds": 5.0, "event": "opening",
            }],
        }]
        transcript = [
            {
                "start_seconds": float(index * 20 + 8),
                "end_seconds": float(index * 20 + 12),
                "text": f"Dialogue evidence number {index} contains enough real words",
            }
            for index in range(6)
        ]
        result, added = promote_transcript_events_to_target(
            evidence, transcript, 120.0, 5,
        )
        anchors = [
            event for event in result[0]["events"]
            if event.get("source") == "transcript_anchor"
        ]
        self.assertEqual(added, 4)
        self.assertEqual(len(anchors), 4)
        self.assertTrue(all(
            event["event"] == event["spoken_text"] for event in anchors
        ))


if __name__ == "__main__":
    unittest.main()
