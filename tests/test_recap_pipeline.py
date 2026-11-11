import json
import unittest

from event_storyboard import validate_render_visual_grounding

from recap_pipeline import (
    _time_seconds,
    assign_story_beats,
    build_full_video_outline_prompt,
    build_evidence_replenishment_prompt,
    build_identity_comparison_prompt,
    build_chunk_evidence_prompt,
    build_single_part_fallback_response,
    build_backend_visual_plan,
    build_part_story_context,
    build_single_part_video_prompt,
    build_story_evidence_digest_prompt,
    build_story_plan_from_digests_prompt,
    build_story_plan_from_evidence_prompt,
    build_story_gap_supplement_prompt,
    build_terminal_anchor_review_prompt,
    build_targeted_visual_shot_repair_prompt,
    build_final_narration_smoothing_prompt,
    apply_final_narration_smoothing,
    clamp_payload_narrations,
    coerce_single_part_response,
    coalesce_dense_story_events,
    compress_story_events_to_scene_budget,
    compact_story_evidence,
    assess_story_evidence_coverage,
    deduplicate_overlapping_chunk_evidence,
    deduplicate_semantic_chunk_evidence,
    stitch_cross_chunk_event_continuations,
    ensure_grounded_terminal_ending,
    expanded_word_target,
    extract_complete_recap_scenes,
    expand_evidence_to_visual_slots,
    inspect_three_part_payload,
    inspect_narration_quality,
    build_narration_repetition_feedback,
    build_targeted_narration_repair_prompt,
    apply_targeted_narration_repairs,
    merge_single_part_responses,
    merge_story_plan_supplement,
    merge_evidence_replenishment,
    normalize_chunk_evidence,
    recover_chunk_evidence,
    normalize_full_video_outline,
    normalize_story_plan_from_evidence,
    normalize_global_evidence_audit,
    parse_json_object,
    requested_word_range,
    required_tts_scene_count,
    select_causal_story_evidence,
    single_part_missing_requirements,
    single_part_has_hard_failures,
    narration_candidate_rank,
    narration_scene_issue_rank,
    style_storytelling_prompt,
    story_outline_for_window,
    split_compact_story_evidence,
    normalize_three_part_payload,
    payload_to_timeline,
    plan_adaptive_story_boundaries,
    plan_evidence_replenishment,
    plan_targeted_evidence_gap_windows,
    proportional_part_targets,
    lock_story_scene_contract,
    three_part_targets,
    validate_single_part_response,
    vietnamese_narration_issues,
    apply_verified_evidence_results,
    apply_terminal_anchor_review,
    apply_targeted_visual_shot_repairs,
    add_objective_evidence_gap_windows,
    fallback_terminal_editorial_anchors,
    drop_unrepairable_visual_events,
    terminal_anchor_review_candidates,
)


def sample_payload():
    return {
        "characters": [{"id": "P1", "name": "Nam", "description": "áo đen"}],
        "parts": [
            {
                "part": 1, "video_start": "00:00", "video_end": "00:30",
                "target_words": 6,
                "scenes": [{
                    "scene_id": "S0001", "start": "00:02", "end": "00:08",
                    "type": "main", "characters": ["P1"],
                    "event": "Nam mở cửa", "tts": "Nam mở cửa và bước vào nhà.",
                    "status": "ok", "errors": [],
                }],
                "reserve_scenes": [],
            },
            {
                "part": 2, "video_start": "00:30", "video_end": "01:00",
                "target_words": 6,
                "scenes": [{
                    "scene_id": "S0002", "start": "00:35", "end": "00:42",
                    "type": "bridge", "characters": ["P1"],
                    "event": "Nam tìm chìa khóa", "tts": "Mọi chuyện.",
                    "status": "error", "errors": ["tts_scene_mismatch"],
                }],
                "reserve_scenes": [{
                    "scene_id": "S0901", "start": "00:45", "end": "00:52",
                    "type": "reserve", "characters": ["P1"],
                    "event": "Nam nhặt chìa khóa", "tts": "",
                    "status": "ok", "errors": [],
                }],
            },
            {
                "part": 3, "video_start": "01:00", "video_end": "01:30",
                "target_words": 6,
                "scenes": [{
                    "scene_id": "S0003", "start": "01:10", "end": "01:18",
                    "type": "main", "characters": ["P1"],
                    "event": "Nam rời đi", "tts": "Nam khóa cửa rồi rời khỏi nhà.",
                    "status": "ok", "errors": [],
                }],
                "reserve_scenes": [],
            },
        ],
    }


class VietnameseNarrationContractTests(unittest.TestCase):
    def test_clear_english_narration_is_rejected(self):
        self.assertIn(
            "english_prose",
            vietnamese_narration_issues(
                "The group is running from the monster after they found the cave."
            ),
        )

    def test_natural_vietnamese_with_foreign_name_is_allowed(self):
        self.assertEqual(
            vietnamese_narration_issues(
                "Nobita cùng Doraemon chạy khỏi hang động khi con quái vật bất ngờ xuất hiện."
            ),
            [],
        )

    def test_writer_broad_range_is_replaced_by_backend_short_shot(self):
        raw = json.dumps({"scenes": [{
            "event_id": "E0001", "tts": "Nobita nhìn thấy cậu bé lạ xuất hiện trước cửa phòng và vội gọi Doraemon đến giúp."
        }]}, ensure_ascii=False)
        locked = {
            "event_id": "E0001", "scene_id": "S0001",
            "start": "00:34", "end": "01:10",
            "visual_sources": [{"start": "00:56", "end": "00:59", "event": "Cậu bé xuất hiện"}],
        }
        result = json.loads(coerce_single_part_response(
            raw, 1, 60, 20, required_event_ids=["E0001"],
            locked_event_by_id={"E0001": locked}, locked_characters=[],
        ))
        scene = result["parts"][0]["scenes"][0]
        self.assertEqual(_time_seconds(scene["start"]), 56)
        self.assertEqual(_time_seconds(scene["end"]), 59)
        clamp_payload_narrations(result)

    def test_writer_beat_outside_locked_visual_is_dropped(self):
        raw = json.dumps({"scenes": [{
            "event_id": "E0001",
            "tts": "Nobita phát hiện người lạ ở cửa phòng, rồi cậu nhanh chóng gọi Doraemon tới giúp.",
            "visual_beats": [{
                "trigger_text": "gọi Doraemon", "start": "00:45", "end": "00:48",
                "confidence": 0.95,
            }],
        }]}, ensure_ascii=False)
        result = json.loads(coerce_single_part_response(
            raw, 1, 60, 20, required_event_ids=["E0001"],
            locked_event_by_id={"E0001": {
                "event_id": "E0001", "scene_id": "S0001",
                "start": "00:10", "end": "00:20",
                "visual_sources": [{"start": "00:10", "end": "00:13"}],
            }}, locked_characters=[],
        ))
        self.assertEqual(result["parts"][0]["scenes"][0]["visual_beats"], [])

    def test_backend_locks_short_shots_and_creates_word_triggered_beat(self):
        plan = build_backend_visual_plan(
            "Nobita bị mẹ mắng nên buồn bã, sau đó cậu vội chạy khỏi nhà để tìm Doraemon.",
            [
                {"start": "00:10", "end": "00:12.8", "event": "mẹ mắng"},
                {"start": "00:18", "end": "00:20.7", "event": "Nobita chạy"},
            ],
            [{"visual_id": "V001", "trigger_text": ""},
             {"visual_id": "V002", "trigger_text": "sau đó cậu"}],
        )
        self.assertEqual(_time_seconds(plan["start"]), 10)
        self.assertLessEqual(_time_seconds(plan["end"]) - _time_seconds(plan["start"]), 3.5)
        self.assertEqual(len(plan["visual_beats"]), 1)
        self.assertTrue(plan["visual_beats"][0]["backend_locked"])

    def test_backend_keeps_flexible_semantic_action_without_fixed_cuts(self):
        plan = build_backend_visual_plan(
            "Nobita chạy qua hang đá rồi phát hiện Doraemon đang chờ ở lối ra.",
            [{"start": "00:10", "end": "00:18", "event": "Nobita chạy qua hang"}],
        )
        self.assertEqual(len(plan["locked_visual_shots"]), 1)
        self.assertEqual(_time_seconds(plan["end"]) - _time_seconds(plan["start"]), 8)
        self.assertEqual(plan["visual_beats"], [])

    def test_backend_rejects_broad_unverified_visual_window(self):
        with self.assertRaisesRegex(ValueError, "VISUAL_GROUNDING_MISSING"):
            build_backend_visual_plan(
                "Một câu recap hợp lệ nhưng chưa có hình được khóa chính xác.",
                [{"start": "00:00", "end": "01:15"}],
            )


class TimestampParsingTests(unittest.TestCase):
    def test_accepts_raw_decimal_seconds_from_gemini(self):
        self.assertEqual(_time_seconds("03.5"), 3.5)
        self.assertEqual(_time_seconds(15), 15.0)
        self.assertEqual(_time_seconds("00:03.50"), 3.5)

    def test_rejects_unsafe_raw_seconds(self):
        for value in ("-1.5", "nan", "inf", "", None, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _time_seconds(value)

    def test_recovers_only_complete_scenes_from_interrupted_stream(self):
        complete = json.dumps({
            "scene_id": "S0001", "start": "00:01", "end": "00:04",
            "story_beat_id": "GB001", "tts": "Một câu hoàn chỉnh để lưu lại.",
        }, ensure_ascii=False)
        partial = complete + '\n{"scene_id":"S0002","tts":"Câu đang'
        scenes = extract_complete_recap_scenes(partial)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["story_beat_id"], "GB001")

class ThreePartRecapTests(unittest.TestCase):
    def test_json_parser_accepts_markdown_and_explanation(self):
        raw = 'Ket qua day:\n```json\n{"parts":[{"part":1}]}\n```\nDa xong.'
        self.assertEqual(parse_json_object(raw)["parts"][0]["part"], 1)

    def test_json_parser_does_not_join_separate_objects(self):
        raw = 'Vi du {"ok":true}. Ket qua {"parts":[{"part":2,"scenes":[]}]}.'
        parsed = parse_json_object(raw)
        self.assertEqual(parsed["parts"][0]["part"], 2)

    def test_json_parser_handles_braces_and_raw_newline_inside_string(self):
        raw = '{"event":"mo {cua}\nroi buoc vao","events":[]}'
        parsed = parse_json_object(raw)
        self.assertIn("{cua}", parsed["event"])

    def test_coerce_single_part_wraps_root_scenes(self):
        raw = json.dumps({
            "characters": [{"id": "P1"}],
            "part": 2,
            "scenes": [{"scene_id": "S1", "tts": "noi dung"}],
        })
        normalized = json.loads(coerce_single_part_response(raw, 2, 90, 100))
        self.assertEqual(len(normalized["parts"]), 1)
        self.assertEqual(normalized["parts"][0]["part"], 2)
        self.assertEqual(normalized["parts"][0]["scenes"][0]["scene_id"], "S1")

    def test_coerce_single_part_accepts_top_level_scene_array(self):
        raw = json.dumps([
            {"scene_id": "S1", "tts": "Câu một", "start": "00:01", "end": "00:04"},
            {"scene_id": "S2", "tts": "Câu hai", "start": "00:05", "end": "00:08"},
        ])
        normalized = json.loads(coerce_single_part_response(
            raw, 3, 30, 40, required_event_ids=["E1", "E2"]
        ))
        scenes = normalized["parts"][0]["scenes"]
        self.assertEqual([scene["scene_id"] for scene in scenes], ["S1", "S2"])
        self.assertEqual(
            [scene["covered_event_ids"] for scene in scenes], [["E1"], ["E2"]]
        )

    def test_coerce_single_part_selects_requested_part(self):
        raw = json.dumps({
            "parts": [
                {"part": 1, "scenes": []},
                {"part": 2, "scenes": [{"scene_id": "S2"}]},
                {"part": 3, "scenes": []},
            ],
        })
        normalized = json.loads(coerce_single_part_response(raw, 2, 90, 100))
        self.assertEqual(len(normalized["parts"]), 1)
        self.assertEqual(normalized["parts"][0]["scenes"][0]["scene_id"], "S2")

    def test_coerce_single_part_restores_fully_missing_event_ids(self):
        raw = json.dumps({
            "parts": [{"part": 1, "scenes": [
                {"scene_id": "S1", "start": "00:01", "end": "00:04"},
                {"scene_id": "S2", "start": "00:05", "end": "00:08"},
            ]}],
        })
        normalized = json.loads(coerce_single_part_response(
            raw, 1, 30, 40, required_event_ids=["E0001", "E0002"]
        ))
        scenes = normalized["parts"][0]["scenes"]
        self.assertEqual(scenes[0]["covered_event_ids"], ["E0001"])
        self.assertEqual(scenes[1]["covered_event_ids"], ["E0002"])

    def test_coerce_does_not_guess_partial_event_mapping(self):
        raw = json.dumps({
            "parts": [{"part": 1, "scenes": [
                {"scene_id": "S1", "covered_event_ids": ["E0001"]},
                {"scene_id": "S2"},
            ]}],
        })
        normalized = json.loads(coerce_single_part_response(
            raw, 1, 30, 40, required_event_ids=["E0001", "E0002"]
        ))
        self.assertNotIn("covered_event_ids", normalized["parts"][0]["scenes"][1])

    def test_coerce_restores_missing_timestamp_from_locked_event(self):
        raw = json.dumps({
            "parts": [{"part": 1, "scenes": [{
                "scene_id": "S1", "start": "00:03", "end": None,
                "covered_event_ids": ["E0001"],
            }]}],
        })
        normalized = json.loads(coerce_single_part_response(
            raw, 1, 30, 20,
            required_event_ids=["E0001"],
            locked_event_by_id={"E0001": {
                "start": "00:02", "end": "00:06",
                "visual_sources": [{"start": "00:02", "end": "00:05"}],
            }},
        ))
        scene = normalized["parts"][0]["scenes"][0]
        self.assertEqual(
            (_time_seconds(scene["start"]), _time_seconds(scene["end"])),
            (2, 5),
        )

    def test_coerce_restores_backend_locked_scene_id_and_word_budget(self):
        raw = json.dumps({
            "parts": [{"part": 1, "scenes": [{
                "scene_id": "WRONG", "start": "00:01", "end": "00:02",
                "tts": "Một câu kể hợp lệ có đủ số từ để mô tả diễn biến quan trọng đang xảy ra trong cảnh phim.",
                "covered_event_ids": ["E0001"],
            }]}],
        }, ensure_ascii=False)
        normalized = json.loads(coerce_single_part_response(
            raw, 1, 30, 20,
            required_event_ids=["E0001"],
            locked_event_by_id={"E0001": {
                "scene_id": "S0042", "start": "00:04", "end": "00:07",
                "tts_target_words": 21,
                "visual_sources": [{"start": "00:04", "end": "00:07"}],
            }},
        ))
        scene = normalized["parts"][0]["scenes"][0]
        self.assertEqual(scene["scene_id"], "S0042")
        self.assertEqual(scene["tts_target_words"], 21)
        self.assertEqual(
            (_time_seconds(scene["start"]), _time_seconds(scene["end"])),
            (4, 7),
        )

    def test_coerce_unwraps_nested_result_payload(self):
        raw = json.dumps({"result": {"characters": [{"id": "P1"}], "parts": [{
            "part": 2,
            "scenes": [{"scene_id": "S1", "covered_event_ids": ["E2"]}],
        }]}})
        normalized = json.loads(coerce_single_part_response(
            raw, 2, 30, 20, required_event_ids=["E2"]
        ))
        self.assertEqual(normalized["characters"], [{"id": "P1"}])
        self.assertEqual(normalized["parts"][0]["part"], 2)
        self.assertEqual(normalized["parts"][0]["scenes"][0]["scene_id"], "S1")

    def test_coerce_uses_root_scenes_when_parts_is_empty(self):
        raw = json.dumps({
            "parts": [],
            "part": 1,
            "scenes": [{"scene_id": "S1", "covered_event_ids": ["E1"]}],
        })
        normalized = json.loads(coerce_single_part_response(
            raw, 1, 30, 20, required_event_ids=["E1"]
        ))
        self.assertEqual(len(normalized["parts"]), 1)
        self.assertEqual(normalized["parts"][0]["scenes"][0]["scene_id"], "S1")

    def test_coerce_selects_candidate_covering_locked_events(self):
        raw = json.dumps({"parts": [
            {"part": "wrong", "scenes": [
                {"scene_id": "BAD", "covered_event_ids": ["OTHER"]},
            ]},
            {"part": 99, "scenes": [
                {"scene_id": "GOOD", "covered_event_ids": ["E2"]},
            ]},
        ]})
        normalized = json.loads(coerce_single_part_response(
            raw, 2, 30, 20, required_event_ids=["E2"]
        ))
        self.assertEqual(normalized["parts"][0]["part"], 2)
        self.assertEqual(normalized["parts"][0]["scenes"][0]["scene_id"], "GOOD")

    def test_coerce_accepts_narration_scenes_alias(self):
        raw = json.dumps({
            "part": 3,
            "narration_scenes": [{"scene_id": "S1"}],
        })
        normalized = json.loads(coerce_single_part_response(raw, 3, 30, 20))
        self.assertEqual(len(normalized["parts"][0]["scenes"]), 1)
        self.assertNotIn("narration_scenes", normalized["parts"][0])

    def test_coerce_replaces_internal_ids_with_character_names(self):
        raw = json.dumps({
            "characters": [
                {"id": "C3", "name": "Nobita"},
                {"id": "C4", "name": "Shizuka"},
            ],
            "parts": [{"part": 1, "scenes": [{
                "scene_id": "S1",
                "event": "C3 gặp C4",
                "tts": "C3 cùng C4 quyết định rời khỏi nhà để bắt đầu chuyến đi.",
            }]}],
        })
        normalized = json.loads(coerce_single_part_response(raw, 1, 30, 20))
        scene = normalized["parts"][0]["scenes"][0]
        self.assertIn("Nobita", scene["tts"])
        self.assertIn("Shizuka", scene["tts"])
        self.assertNotRegex(scene["tts"], r"\bC[34]\b")

    def test_grounded_fallback_is_valid_when_model_schema_keeps_failing(self):
        context = {
            "characters": [{"id": "P1", "name": "Lan"}],
            "required_events": [
                {
                    "event_id": "E0001", "start": "00:01", "end": "00:04",
                    "event": "Lan mở cánh cửa", "visual_detail": "căn phòng tối",
                    "characters": ["P1"], "importance": "main",
                    "story_role": "development",
                },
                {
                    "event_id": "E0002", "start": "00:05", "end": "00:09",
                    "event": "Lan bước vào phòng", "consequence": "ánh đèn bật sáng",
                    "characters": ["P1"], "importance": "main",
                    "story_role": "ending",
                },
            ],
        }
        fallback = build_single_part_fallback_response(context, 1, 30, 40)
        requirements = single_part_missing_requirements(
            fallback, 30, 40,
            required_event_ids=["E0001", "E0002"],
            require_word_floor=True,
            require_tts_word_range=True,
            required_scene_count=2,
        )
        self.assertFalse(requirements["needs_revision"])
        payload = json.loads(fallback)
        self.assertEqual(
            [scene["covered_event_ids"] for scene in payload["parts"][0]["scenes"]],
            [["E0001"], ["E0002"]],
        )
        self.assertTrue(all(scene["tts"].count(".") == 1 for scene in payload["parts"][0]["scenes"]))

    def test_grounded_fallback_replaces_bad_timestamp_with_matching_reserve(self):
        context = {
            "required_events": [
                {
                    "event_id": "E0001", "start": "00:01", "end": "00:04",
                    "event": "Lan mo canh cua", "characters": ["P1"],
                },
                {
                    "event_id": "E0002", "start": "bad", "end": "timestamp",
                    "event": "Lan nhat chiec chia khoa", "characters": ["P1"],
                    "story_beat_id": "B2",
                },
            ],
            "fallback_visual_candidates": [
                {
                    "scene_id": "S0901", "start": "00:08", "end": "00:12",
                    "event": "Lan tim thay va nhat chiec chia khoa",
                    "characters": ["P1"], "story_beat_id": "B2",
                },
            ],
        }
        fallback = build_single_part_fallback_response(context, 1, 20, 40)
        payload = json.loads(fallback)
        repaired = payload["parts"][0]["scenes"][1]
        self.assertEqual(
            (repaired["start"], repaired["end"]),
            ("00:08.00", "00:12.00"),
        )
        self.assertEqual(repaired["timestamp_fallback_from"], "S0901")
        self.assertEqual(repaired["timestamp_fallback_kind"], "reserve_scene")

    def test_grounded_fallback_never_stops_when_all_timestamps_are_bad(self):
        context = {
            "required_events": [
                {"event_id": "E1", "start": "bad", "end": "bad", "event": "Mo dau"},
                {"event_id": "E2", "start": None, "end": None, "event": "Ket thuc"},
            ],
        }
        fallback = build_single_part_fallback_response(context, 1, 20, 40)
        requirements = single_part_missing_requirements(
            fallback, 20, 40, required_event_ids=["E1", "E2"]
        )
        self.assertFalse(requirements["invalid_timestamp_scenes"])
        payload = json.loads(fallback)
        self.assertTrue(all(
            str(scene.get("timestamp_fallback_from", "")).startswith("POSITIONAL_")
            for scene in payload["parts"][0]["scenes"]
        ))

    def test_grounded_fallback_bad_final_scene_still_reaches_required_ending(self):
        context = {
            "required_ending_sec": 19,
            "required_events": [
                {"event_id": "E1", "start": "00:01", "end": "00:04", "event": "Mo dau"},
                {"event_id": "E2", "start": "bad", "end": "bad", "event": "Ket thuc"},
            ],
        }
        fallback = build_single_part_fallback_response(context, 1, 20, 40)
        payload = json.loads(fallback)
        final_scene = payload["parts"][0]["scenes"][-1]
        self.assertEqual(final_scene["end"], "00:19.00")
        self.assertEqual(
            final_scene["timestamp_fallback_from"], "POSITIONAL_ENDING_WINDOW"
        )
        validate_single_part_response(
            fallback, 1, 40, required_event_ids=["E1", "E2"],
            required_ending_sec=19,
        )

    def test_fragmented_metadata_style_tts_is_only_a_soft_warning(self):
        raw = json.dumps({"parts": [{"part": 1, "scenes": [{
            "scene_id": "S1", "start": "00:01", "end": "00:04",
            "tts": (
                "Kukuru dặn dò Nobita. Sự chia ly. Lời hứa. Nobita đồng ý. "
                "Hai người đứng cạnh nhau trước khi rời đi."
            ),
            "covered_event_ids": ["E1"],
        }]}]})
        requirements = single_part_missing_requirements(
            raw, 30, 20, required_event_ids=["E1"], require_tts_word_range=True
        )
        self.assertEqual(requirements["invalid_tts_scenes"], [])
        self.assertTrue(requirements["tts_length_warnings"][0]["fragmented"])
    def test_final_clamp_blocks_english_and_internal_ids(self):
        payload = sample_payload()
        payload["parts"][0]["scenes"][0]["tts"] = (
            "C3 and C4 meet on a bridge and discuss running away from home."
        )
        cleaned = clamp_payload_narrations(payload)
        narration = cleaned["parts"][0]["scenes"][0]["tts"]
        self.assertNotRegex(narration, r"\bC[34]\b")
        self.assertNotIn(" meet ", f" {narration.lower()} ")
        self.assertNotIn("tình thế của các nhân vật", narration)

    def test_quality_inspection_flags_padding_and_nearby_repetition(self):
        payload = {"parts": [{"scenes": [
            {
                "scene_id": "S0001", "story_beat_id": "B001",
                "tts": "Kukuru quyết tâm cao độ chạy về làng sau khi nhìn thấy cột khói đen.",
            },
            {
                "scene_id": "S0002", "story_beat_id": "B001",
                "tts": "Nhìn thấy khói đen, Kukuru quyết tâm chạy thật nhanh trở về ngôi làng.",
            },
        ]}]}
        issues = inspect_narration_quality(payload)
        codes = {issue["code"] for issue in issues}
        self.assertIn("padded_ai_prose", codes)
        self.assertIn("repeated_narration_idea", codes)

    def test_quality_inspection_allows_distinct_locked_foci_in_same_beat(self):
        payload = {"characters": [{"name": "Kukuru"}], "parts": [{"scenes": [
            {
                "scene_id": "S0001", "story_beat_id": "B001",
                "narration_focus": "Kukuru phát hiện cuộc tấn công",
                "tts": "Kukuru chạy về ngôi làng để cảnh báo mọi người trước cuộc tấn công bất ngờ.",
            },
            {
                "scene_id": "S0002", "story_beat_id": "B001",
                "narration_focus": "Mọi người dựng hàng rào phòng thủ",
                "tts": "Kukuru chạy quanh ngôi làng rồi giúp mọi người dựng hàng rào chống cuộc tấn công.",
            },
        ]}]}
        repeated = [
            issue for issue in inspect_narration_quality(payload)
            if issue["code"] == "repeated_narration_idea"
        ]
        self.assertEqual(repeated, [])

    def test_repetition_feedback_identifies_pair_and_required_new_fact(self):
        payload = {"parts": [{"scenes": [
            {"scene_id": "S0001", "tts": "Kukuru nhìn thấy ngôi làng đang bốc cháy dữ dội."},
            {
                "scene_id": "S0002",
                "narration_focus": "Kukuru gọi mọi người chạy khỏi cổng sau",
                "tts": "Kukuru nhìn thấy ngôi làng bốc cháy nên lập tức quay lại.",
            },
        ]}]}
        feedback = build_narration_repetition_feedback(payload, [{
            "code": "repeated_narration_idea",
            "scene_id": "S0002", "related_scene_id": "S0001",
        }])
        self.assertIn("S0002 lặp S0001", feedback)
        self.assertIn("Kukuru gọi mọi người chạy khỏi cổng sau", feedback)

    def test_candidate_rank_prefers_less_repetition_before_soft_distance(self):
        base = {
            "coverage_missing": False,
            "invalid_timestamp_scenes": [],
            "scene_count_gap": 0,
        }
        close_but_repeated = dict(base, distance=1)
        farther_but_unique = dict(base, distance=500)
        self.assertLess(
            narration_candidate_rank(farther_but_unique, 0),
            narration_candidate_rank(close_but_repeated, 2),
        )

    def test_targeted_repair_requests_only_bad_and_missing_events(self):
        payload = {
            "parts": [{"part": 1, "scenes": [
                {"scene_id": "S0001", "covered_event_ids": ["E1"],
                 "tts": "Cau nay da dat yeu cau."},
                {"scene_id": "S0002", "covered_event_ids": ["E2"],
                 "tts": "Too short."},
            ]}],
        }
        locked = {
            "E1": {"scene_id": "S0001", "event": "mot", "tts_min_words": 15,
                   "tts_max_words": 25},
            "E2": {"scene_id": "S0002", "event": "hai", "tts_min_words": 15,
                   "tts_max_words": 25},
            "E3": {"scene_id": "S0003", "event": "ba", "tts_min_words": 15,
                   "tts_max_words": 25},
        }
        requirements = {
            "invalid_tts_scenes": [{"scene_id": "S0002", "word_count": 2}],
            "invalid_character_mentions": [],
            "missing_event_ids": ["E3"],
        }
        prompt = build_targeted_narration_repair_prompt(
            payload, requirements, locked
        )
        repair_rows = json.loads(
            prompt.split("CAC CAU CAN VIET LAI:\n", 1)[1]
            .split("\n\nChi tra JSON", 1)[0]
        )
        self.assertEqual(
            [row["event_id"] for row in repair_rows], ["E2", "E3"]
        )

    def test_targeted_repair_preserves_good_tts_and_restores_missing_scene(self):
        payload = {
            "parts": [{"part": 1, "scenes": [
                {"scene_id": "S0001", "covered_event_ids": ["E1"],
                 "tts": "giu nguyen cau tot"},
                {"scene_id": "S0002", "covered_event_ids": ["E2"],
                 "tts": "cau loi"},
            ]}],
        }
        locked = {
            "E1": {"scene_id": "S0001"},
            "E2": {"scene_id": "S0002"},
            "E3": {"scene_id": "S0003"},
        }
        repaired = apply_targeted_narration_repairs(
            payload,
            json.dumps([
                {"scene_id": "S0002", "event_id": "E2", "tts": "cau hai moi"},
                {"scene_id": "S0003", "event_id": "E3", "tts": "cau ba moi"},
            ]),
            ["E2", "E3"], locked,
        )
        scenes = repaired["parts"][0]["scenes"]
        self.assertEqual([scene["scene_id"] for scene in scenes], [
            "S0001", "S0002", "S0003",
        ])
        self.assertEqual(scenes[0]["tts"], "giu nguyen cau tot")
        self.assertEqual(scenes[1]["tts"], "cau hai moi")

    def test_scene_rank_keeps_safe_old_sentence_over_worse_retry(self):
        safe = {"invalid_tts_scenes": [], "invalid_character_mentions": []}
        wrong_character = {
            "invalid_tts_scenes": [],
            "invalid_character_mentions": [{"scene_id": "S0002"}],
        }
        self.assertLess(
            narration_scene_issue_rank(safe, "S0002"),
            narration_scene_issue_rank(wrong_character, "S0002"),
        )

    def test_final_clamp_uses_canonical_name_and_removes_padding(self):
        payload = sample_payload()
        payload["characters"] = [{
            "id": "P1", "name": "Kukuru", "aliases": ["Kuku", "Kukul"],
        }]
        payload["parts"][0]["scenes"][0]["tts"] = (
            "Kukul quyết tâm cao độ đưa mọi người đến nơi an toàn tuyệt đối."
        )
        cleaned = clamp_payload_narrations(payload)
        narration = cleaned["parts"][0]["scenes"][0]["tts"]
        self.assertIn("Kukuru", narration)
        self.assertNotIn("Kukul", narration)
        self.assertNotIn("quyết tâm cao độ", narration)
        self.assertNotIn("an toàn tuyệt đối", narration)

    def test_chunk_prompt_receives_locked_character_registry(self):
        prompt = build_chunk_evidence_prompt(
            1, 2, 60, 0, 60, 8,
            character_registry=[{"id": "P01", "name": "Kukuru"}],
        )
        self.assertIn("SO NHAN VAT DA KHOA TU CAC DOAN TRUOC", prompt)
        self.assertIn("Kukuru", prompt)
        self.assertIn('"name":"Kukuru"', prompt)

    def test_chunk_evidence_is_converted_to_global_timestamps(self):
        raw = json.dumps({
            "characters": [{"id": "C1", "name": "Lan"}],
            "events": [{
                "start_seconds": 12.0,
                "end_seconds": 16.0,
                "event": "Lan mo cua",
            }],
        })
        evidence = normalize_chunk_evidence(raw, 600.0, 120.0)
        self.assertEqual(evidence["events"][0]["start_seconds"], 612.0)
        self.assertEqual(evidence["events"][0]["end_seconds"], 616.0)

    def test_chunk_evidence_keeps_flexible_visual_beat_with_event_id(self):
        raw = json.dumps({
            "characters": [],
            "events": [{
                "start": 12.0,
                "end": 25.0,
                "event": "Nobita chạy qua hang đá",
                "visual_shots": [{
                    "start_seconds": 14.0,
                    "end_seconds": 22.0,
                    "description": "Nobita chạy qua nhiều lối rẽ trong hang",
                }],
            }],
        }, ensure_ascii=False)
        evidence = normalize_chunk_evidence(
            raw, 600.0, 120.0, chunk_number=2,
        )
        event = evidence["events"][0]
        self.assertEqual(event["evidence_id"], "EV002_0001")
        self.assertEqual(len(event["visual_sources"]), 1)
        self.assertEqual(event["visual_sources"][0]["start_seconds"], 614.0)
        self.assertFalse(event["visual_grounding_missing"])

    def test_chunk_recovery_repairs_missing_timestamps(self):
        raw = json.dumps({
            "events": [
                {"event": "Nobita tìm thấy con cá"},
                {"event": "Nobita lao xuống suối bắt cá"},
            ],
        })
        evidence = recover_chunk_evidence([raw], 600, 120, chunk_number=4)
        self.assertEqual(len(evidence["events"]), 2)
        self.assertGreaterEqual(evidence["events"][0]["start_seconds"], 600)
        self.assertLessEqual(evidence["events"][-1]["end_seconds"], 720)
        self.assertTrue(evidence["events"][0]["recovered_from_retry"])

    def test_chunk_recovery_closes_truncated_json(self):
        raw = '{"characters":[],"events":[{"event":"Nobita bắt cá","start_seconds":2,"end_seconds":8}'
        evidence = recover_chunk_evidence([raw], 0, 30, chunk_number=1)
        self.assertEqual(len(evidence["events"]), 1)
        self.assertEqual(evidence["events"][0]["event"], "Nobita bắt cá")
        self.assertFalse(evidence.get("fallback_evidence", False))

    def test_chunk_recovery_converts_accidental_global_timestamps(self):
        raw = json.dumps({
            "events": [{
                "event": "Nobita chạy khỏi bờ suối",
                "start_seconds": 610,
                "end_seconds": 620,
            }],
        })
        evidence = recover_chunk_evidence([raw], 600, 120, chunk_number=4)
        self.assertEqual(evidence["events"][0]["start_seconds"], 610)
        self.assertEqual(evidence["events"][0]["end_seconds"], 620)

    def test_chunk_recovery_uses_safe_fallback_when_all_retries_are_empty(self):
        evidence = recover_chunk_evidence(
            ["", "không có JSON", "{}"], 300, 60, chunk_number=3,
        )
        self.assertTrue(evidence["fallback_evidence"])
        self.assertEqual(len(evidence["events"]), 1)
        self.assertTrue(evidence["events"][0]["needs_visual_review"])
        self.assertEqual(evidence["events"][0]["start_seconds"], 300)
        self.assertEqual(evidence["events"][0]["end_seconds"], 360)

    def test_chunk_evidence_qualifies_local_event_cluster(self):
        raw = json.dumps({
            "characters": [],
            "events": [{
                "start_seconds": 12.0, "end_seconds": 16.0,
                "event": "The hero transforms", "event_cluster": "CLIMAX_01",
            }],
        })
        evidence = normalize_chunk_evidence(
            raw, 600.0, 120.0, chunk_number=2
        )
        self.assertEqual(
            evidence["events"][0]["event_cluster"], "CH002_CLIMAX_01"
        )

    def test_chunk_evidence_replaces_internal_character_ids_in_prose(self):
        raw = json.dumps({
            "characters": [
                {"id": "C3", "name": "Nobita"},
                {"id": "C4", "name": "Shizuka"},
            ],
            "events": [{
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "event": "C3 gặp C4 trên cầu",
                "visual_detail": "C3 nói chuyện với C4",
                "characters": ["C3", "C4"],
            }],
        })
        evidence = normalize_chunk_evidence(raw, 0, 10)
        event = evidence["events"][0]
        self.assertEqual(event["event"], "Nobita gặp Shizuka trên cầu")
        self.assertNotRegex(event["visual_detail"], r"\bC[34]\b")
        self.assertEqual(event["characters"], ["Nobita", "Shizuka"])

    def test_chunk_evidence_preserves_chunk_qualified_unknown_identity(self):
        raw = json.dumps({
            "characters": [{
                "id": "C1", "name": "", "description": "cậu bé cầm giáo",
                "visual_signature": "tóc đen, áo da",
                "identity_status": "unknown", "identity_confidence": 0.2,
            }],
            "events": [{
                "start_seconds": 2, "end_seconds": 5,
                "event": "C1 chạy khỏi đám cháy", "characters": ["C1"],
                "temporal_relation": "flashback",
            }],
        })
        evidence = normalize_chunk_evidence(raw, 600, 20, chunk_number=2)
        character = evidence["characters"][0]
        event = evidence["events"][0]
        self.assertEqual(character["id"], "CH002_C1")
        self.assertEqual(character["identity_status"], "unknown")
        self.assertEqual(event["characters"], ["CH002_C1"])
        self.assertEqual(event["evidence_id"], "EV002_0001")
        self.assertEqual(event["start_seconds"], 602)
        self.assertEqual(event["temporal_relation"], "flashback")

    def test_global_audit_rejects_unknown_ids_and_bounds_review_windows(self):
        evidence = [{
            "characters": [{"id": "CH001_C1"}],
            "events": [{"evidence_id": "EV001_0001"}],
        }]
        raw = json.dumps({
            "canonical_characters": [{"id": "P1", "name": "Kukuru"}],
            "identity_links": [
                {"local_id": "CH001_C1", "canonical_id": "P1",
                 "confidence": 0.95, "basis": "explicit_name"},
                {"local_id": "CH999_C1", "canonical_id": "P1",
                 "confidence": 1, "basis": "explicit_name"},
            ],
            "contradictions": [{
                "type": "identity", "evidence_ids": ["EV001_0001", "BAD"]
            }],
            "review_windows": [{
                "start_seconds": 10, "end_seconds": 300,
                "evidence_ids": ["EV001_0001"], "questions": ["Ai đây?"]
            }],
        })
        audit = normalize_global_evidence_audit(raw, evidence, 400)
        self.assertEqual(len(audit["identity_links"]), 1)
        self.assertEqual(audit["contradictions"][0]["evidence_ids"], ["EV001_0001"])
        window = audit["review_windows"][0]
        self.assertLessEqual(window["end_seconds"] - window["start_seconds"], 120)

    def test_confirmed_chunk_name_cannot_be_overwritten_by_wrong_villain(self):
        evidence = [{
            "characters": [{
                "id": "CH001_C3", "name": "Doraemon",
                "proposed_name": "Doraemon", "identity_status": "confirmed",
                "identity_confidence": 1.0, "description": "mèo máy màu xanh",
            }],
            "events": [{
                "evidence_id": "EV001_0001", "event": "Doraemon lấy bảo bối",
                "characters": ["CH001_C3"],
            }],
        }]
        wrong_audit = json.dumps({
            "canonical_characters": [{"id": "P1", "name": "Giga Zombie"}],
            "identity_links": [{
                "local_id": "CH001_C3", "canonical_id": "P1",
                "confidence": 0.99, "basis": "multiple_evidence",
            }],
        }, ensure_ascii=False)
        audit = normalize_global_evidence_audit(wrong_audit, evidence, 100)
        link = next(
            item for item in audit["identity_links"]
            if item["local_id"] == "CH001_C3"
        )
        canonical = {
            item["id"]: item for item in audit["canonical_characters"]
        }
        self.assertEqual(canonical[link["canonical_id"]]["name"], "Doraemon")
        self.assertEqual(link["basis"], "explicit_name")

        wrong_video_verification = json.dumps({
            "identity_verifications": [{
                "local_id": "CH001_C3", "canonical_id": "P1",
                "confirmed": True, "confidence": 0.99,
                "basis": "distinctive_visual",
            }],
        }, ensure_ascii=False)
        resolved = apply_verified_evidence_results(
            evidence, audit, [wrong_video_verification]
        )
        character = resolved[0]["characters"][0]
        self.assertEqual(character["name"], "Doraemon")
        self.assertNotEqual(character["canonical_id"], "P1")

    def test_objective_opening_gap_forces_video_review(self):
        evidence = [{
            "start_seconds": 0, "end_seconds": 600,
            "events": [{
                "evidence_id": "EV001_0001", "start_seconds": 176,
                "end_seconds": 180, "event": "Mẹ mắng cậu bé",
            }],
        }]
        audit = add_objective_evidence_gap_windows(
            {"review_windows": [], "identity_comparisons": []}, evidence, 600
        )
        window = audit["review_windows"][0]
        self.assertEqual(window["start_seconds"], 0)
        self.assertLessEqual(window["end_seconds"], 120)
        self.assertTrue(window["objective_gap"])

    def test_verified_opening_gap_can_add_only_in_window_event(self):
        evidence = [{
            "start_seconds": 0, "end_seconds": 600,
            "characters": [], "events": [{
                "evidence_id": "EV001_0001", "start_seconds": 176,
                "end_seconds": 180, "event": "Mẹ mắng cậu bé", "characters": [],
            }],
        }]
        verification = json.dumps({
            "_review_window": {
                "start_seconds": 0, "end_seconds": 120,
                "allow_missing_events": True,
            },
            "missing_events": [
                {"start_seconds": 20, "end_seconds": 30,
                 "event": "Kukuru đi săn cá", "characters": ["P1"],
                 "narrative_function": "setup"},
                {"start_seconds": 150, "end_seconds": 160,
                 "event": "Ngoài cửa sổ kiểm chứng", "characters": []},
            ],
        })
        resolved = apply_verified_evidence_results(
            evidence, {"canonical_characters": [], "identity_links": []},
            [verification],
        )
        added = [
            event for event in resolved[0]["events"]
            if event.get("source") == "verified_missing_gap"
        ]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["event"], "Kukuru đi săn cá")
        self.assertEqual(added[0]["evidence_id"], "EVR0001")

    def test_audit_normalizes_cross_time_identity_comparison(self):
        evidence = [{
            "characters": [{"id": "CH001_C1"}, {"id": "CH005_C2"}],
            "events": [{"evidence_id": "EV001_0001"}],
        }]
        raw = json.dumps({
            "canonical_characters": [{"id": "P1", "name": "Kukuru"}],
            "identity_comparisons": [{
                "first_start_seconds": 10, "first_end_seconds": 100,
                "second_start_seconds": 2400, "second_end_seconds": 2500,
                "local_ids": ["CH001_C1", "CH005_C2"],
                "question": "Có cùng là Kukuru không?",
            }],
        })
        audit = normalize_global_evidence_audit(raw, evidence, 3000)
        comparison = audit["identity_comparisons"][0]
        self.assertEqual(comparison["first_end_seconds"], 70)
        self.assertEqual(comparison["second_end_seconds"], 2460)
        prompt = build_identity_comparison_prompt(comparison, audit, evidence)
        self.assertIn("HAI khoang", prompt)

    def test_only_strong_or_video_verified_identity_is_applied(self):
        evidence = [{
            "characters": [
                {"id": "CH001_C1", "identity_status": "unknown",
                 "proposed_name": "Nobita", "visual_signature": "cậu bé cầm giáo"},
                {"id": "CH001_C2", "identity_status": "unknown"},
            ],
            "events": [{
                "evidence_id": "EV001_0001",
                "event": "cậu bé chạy", "characters": ["CH001_C1"],
            }],
        }]
        audit = {
            "canonical_characters": [
                {"id": "P1", "name": "Kukuru"},
                {"id": "P2", "name": "Nobita"},
            ],
            "identity_links": [
                {"local_id": "CH001_C1", "canonical_id": "P1",
                 "confidence": 0.65, "basis": "distinctive_visual"},
                {"local_id": "CH001_C2", "canonical_id": "P2",
                 "confidence": 0.95, "basis": "explicit_name"},
            ],
        }
        verification = json.dumps({
            "identity_verifications": [{
                "local_id": "CH001_C1", "canonical_id": "P1",
                "confirmed": True, "confidence": 0.91,
                "basis": "direct_transition",
            }],
            "event_repairs": [{
                "evidence_id": "EV001_0001",
                "event": "Kukuru chạy khỏi đám cháy",
                "temporal_relation": "flashback",
            }],
        })
        resolved = apply_verified_evidence_results(
            evidence, audit, [verification]
        )
        characters = resolved[0]["characters"]
        self.assertEqual(characters[0]["canonical_id"], "P1")
        self.assertEqual(characters[1]["canonical_id"], "P2")
        self.assertEqual(resolved[0]["events"][0]["characters"], ["P1"])
        self.assertIn("Kukuru", resolved[0]["events"][0]["event"])
        self.assertNotIn("Nobita", resolved[0]["events"][0]["event"])
        self.assertEqual(resolved[0]["events"][0]["temporal_relation"], "flashback")

    def test_unverified_guessed_name_becomes_neutral_description(self):
        evidence = [{
            "characters": [{
                "id": "CH001_C1", "proposed_name": "Nobita",
                "description": "cậu bé cầm giáo", "identity_status": "probable",
            }],
            "events": [{
                "evidence_id": "EV001_0001", "event": "Nobita chạy vào rừng",
                "characters": ["CH001_C1"],
            }],
        }]
        resolved = apply_verified_evidence_results(
            evidence, {"canonical_characters": [], "identity_links": []}, []
        )
        event = resolved[0]["events"][0]
        self.assertEqual(event["characters"], ["cậu bé cầm giáo"])
        self.assertEqual(event["event"], "cậu bé cầm giáo chạy vào rừng")
        self.assertEqual(event["local_character_ids"], ["CH001_C1"])

    def test_chunk_prompt_does_not_request_tts(self):
        prompt = build_chunk_evidence_prompt(1, 3, 600, 0, 600, 20)
        self.assertIn("CHUA viet loi TTS", prompt)
        self.assertIn("Muc tieu khoang 20 bien co", prompt)
        self.assertIn("Doan yen binh duoc tra it hon", prompt)
        self.assertIn('"cluster":"C01"', prompt)
        self.assertIn("khong phai ly do cat ngang cao trao", prompt)
        self.assertIn("khong ep 2-3 giay", prompt)
        self.assertNotIn('"visual_shots":[]', prompt)

    def test_chunk_prompt_marks_primary_window_and_open_continuations(self):
        prompt = build_chunk_evidence_prompt(
            1, 3, 1080, 0, 1080, 54, primary_end_sec=900,
            previous_end_state={"summary": "Nobita đang chạy trốn"},
        )
        self.assertIn("giay cuc bo 900.00", prompt)
        self.assertIn("continues_to_next", prompt)
        self.assertIn("continues_from_previous", prompt)
        self.assertIn("KHONG PHAI gioi han output", prompt)
        self.assertIn("TRANG THAI CUOI CLIP TRUOC", prompt)

    def test_adaptive_boundaries_do_not_cut_inside_structural_beats(self):
        outline = {"story_beats": [
            {"beat_id": "GB001", "start_seconds": 0, "end_seconds": 120,
             "narrative_function": "setup", "importance": "main"},
            {"beat_id": "GB002", "start_seconds": 500, "end_seconds": 700,
             "narrative_function": "climax", "story_role": "climax",
             "importance": "critical", "depends_on": ["GB001"]},
            {"beat_id": "GB003", "start_seconds": 1050, "end_seconds": 1250,
             "narrative_function": "reversal", "importance": "critical"},
            {"beat_id": "GB004", "start_seconds": 1650, "end_seconds": 1780,
             "narrative_function": "ending", "importance": "main"},
        ]}
        boundaries = plan_adaptive_story_boundaries(outline, 1800)
        self.assertEqual(len(boundaries), 4)
        self.assertFalse(any(500 < cut < 700 for cut in boundaries[1:-1]))
        self.assertFalse(any(1050 < cut < 1250 for cut in boundaries[1:-1]))
        self.assertNotEqual(boundaries, [0, 600, 1200, 1800])

    def test_adaptive_boundaries_fall_back_to_equal_without_outline(self):
        self.assertEqual(
            plan_adaptive_story_boundaries({}, 1800),
            [0.0, 600.0, 1200.0, 1800.0],
        )

    def test_full_video_pass_requests_duration_aware_chapter_map(self):
        prompt = build_full_video_outline_prompt(6200, 3100)
        self.assertIn("xem TOÀN BỘ video", prompt)
        self.assertIn("BẢN ĐỒ CHƯƠNG", prompt)
        self.assertIn("Lượt xem clip sau sẽ tách chi tiết", prompt)
        self.assertIn("detail_level", prompt)
        self.assertIn("is_final_action", prompt)
        self.assertIn("bridge_from_previous", prompt)
        self.assertIn("Phần kết chỉ giữ MỘT beat ngắn", prompt)
        self.assertIn("visual_regions", prompt)
        self.assertIn("2-3 giây", prompt)
        self.assertIn("CHƯA viết TTS", prompt)
        self.assertNotIn("supporting_scene_ids", prompt)

    def test_outline_keeps_only_grounded_bridge_evidence(self):
        raw = json.dumps({"story_beats": [{
            "beat_id": "A", "start_seconds": 0, "end_seconds": 8,
            "summary": "Cậu bé phát hiện ngôi làng bị phá hủy",
            "detail_level": "normal", "narrative_function": "setup",
            "visual_regions": [{"start_seconds": 0, "end_seconds": 5}],
        }, {
            "beat_id": "B", "start_seconds": 12, "end_seconds": 20,
            "summary": "Cậu bé tiến vào khu rừng", "detail_level": "normal",
            "is_final_action": True, "narrative_function": "climax",
            "bridge_from_previous": {
                "start_seconds": 8, "end_seconds": 12,
                "relation": "location_change",
                "facts": ["Cậu bé rời khỏi ngôi làng"],
                "confidence": 0.9,
            },
            "visual_regions": [{"start_seconds": 12, "end_seconds": 18}],
        }, {
            "beat_id": "C", "start_seconds": 22, "end_seconds": 28,
            "summary": "Câu chuyện kết thúc", "detail_level": "brief",
            "story_role": "ending", "narrative_function": "ending",
            "bridge_from_previous": {
                "start_seconds": 100, "end_seconds": 101,
                "relation": "cause", "facts": ["Bằng chứng ngoài video"],
                "confidence": 0.99,
            },
            "visual_regions": [{"start_seconds": 22, "end_seconds": 26}],
        }]}, ensure_ascii=False)
        outline = normalize_full_video_outline(raw, 30)
        self.assertEqual(
            outline["story_beats"][1]["bridge_from_previous"]["bridge_id"],
            "BR002",
        )
        self.assertIsNone(outline["story_beats"][2]["bridge_from_previous"])

    def test_backend_infers_grounded_final_resolution_as_ending(self):
        raw = json.dumps({"story_beats": [{
            "start_seconds": 0, "end_seconds": 40,
            "summary": "Nhân vật bắt đầu hành trình",
            "narrative_function": "setup", "detail_level": "normal",
        }, {
            "start_seconds": 92, "end_seconds": 99,
            "summary": "Mọi người trở về nhà",
            "outcome": "Mọi người đã an toàn",
            "state_after": "Cuộc sống trở lại bình thường",
            "narrative_function": "resolution", "story_role": "resolution",
            "detail_level": "normal",
        }]}, ensure_ascii=False)
        outline = normalize_full_video_outline(raw, 100)
        ending = outline["story_beats"][-1]
        self.assertEqual(ending["story_role"], "ending")
        self.assertEqual(ending["narrative_function"], "ending")
        self.assertEqual(ending["detail_level"], "brief")
        self.assertTrue(ending["ending_inferred_by_backend"])
        self.assertEqual(outline["ending_inference"]["status"], "backend_inferred")

    def test_backend_does_not_call_unfinished_climax_an_ending(self):
        raw = json.dumps({"story_beats": [{
            "start_seconds": 90, "end_seconds": 99,
            "summary": "Hai nhân vật vẫn đang giao chiến",
            "outcome": "Đòn tấn công vừa bị chặn",
            "state_after": "Trận chiến tiếp tục",
            "narrative_function": "climax", "story_role": "climax",
            "detail_level": "detailed", "is_final_action": True,
        }]}, ensure_ascii=False)
        outline = normalize_full_video_outline(raw, 100)
        self.assertEqual(outline["ending_inference"]["status"], "missing")
        self.assertNotEqual(outline["story_beats"][-1]["story_role"], "ending")

    def test_full_video_outline_guides_chunk_without_becoming_evidence(self):
        raw = json.dumps({
            "story_contract": {"premise": "A journey"},
            "story_beats": [
                {"start_seconds": 0, "end_seconds": 40,
                 "summary": "Opening", "narrative_function": "setup"},
                {"start_seconds": 100, "end_seconds": 150,
                 "summary": "Attack", "narrative_function": "climax",
                 "importance": "critical", "story_role": "climax"},
                {"start_seconds": 240, "end_seconds": 290,
                 "summary": "Ending", "narrative_function": "ending",
                 "story_role": "ending"},
            ],
        })
        outline = normalize_full_video_outline(raw, 300)
        self.assertEqual(len(outline["story_beats"]), 3)
        window = story_outline_for_window(outline, 80, 180)
        self.assertTrue(any(beat["summary"] == "Attack" for beat in window))
        prompt = build_chunk_evidence_prompt(
            2, 3, 100, 80, 180, 12, story_outline=window
        )
        self.assertIn("DINH HUONG TU LUOT XEM TOAN PHIM", prompt)
        self.assertIn("Attack", prompt)
        self.assertIn("video trong clip nay moi la bang chung", prompt)

    def test_full_video_outline_caps_output_at_one_hundred_fifty_beats(self):
        raw = json.dumps({"story_beats": [
            {
                "start_seconds": index * 10,
                "end_seconds": index * 10 + 5,
                "summary": f"Beat {index}",
                "narrative_function": (
                    "ending" if index == 199 else "development"
                ),
                "importance": "critical" if index == 199 else "main",
            }
            for index in range(200)
        ]})
        outline = normalize_full_video_outline(raw, 2100)
        self.assertEqual(len(outline["story_beats"]), 150)
        self.assertTrue(any(
            beat["narrative_function"] == "ending"
            for beat in outline["story_beats"]
        ))

    def test_story_plan_from_evidence_uses_text_only_locked_data(self):
        prompt = build_story_plan_from_evidence_prompt(
            [{"characters": [], "events": [{
                "start_seconds": 1.0, "end_seconds": 4.0,
                "event": "mot su kien",
            }]}],
            1800,
            220,
        )
        self.assertIn("KHONG co video", prompt)
        self.assertIn("bang chung", prompt.lower())
        self.assertIn("mot su kien", prompt)

    def test_story_outline_prompt_strips_audit_detail_and_derived_slots(self):
        evidence = [{
            "chunk_number": 1,
            "chunk_start_seconds": 0,
            "chunk_end_seconds": 600,
            "characters": [{"visual_signature": "SECRET_VISUAL_DETAIL"}],
            "events": [
                {
                    "evidence_id": "EV001", "start_seconds": 1,
                    "end_seconds": 4, "event": "The real turning point",
                    "visual_detail": "SECRET_VISUAL_DETAIL",
                    "narrative_function": "turning_point",
                },
                {
                    "evidence_id": "EV002", "start_seconds": 4,
                    "end_seconds": 6, "event": "Mechanical split",
                    "derived_visual_slot": True,
                },
                {
                    "evidence_id": "EV003", "start_seconds": 6,
                    "end_seconds": 8, "event": "Repeated walking shot",
                    "importance": "bridge",
                    "narrative_function": "development",
                },
            ],
        }]
        compact = compact_story_evidence(evidence)
        self.assertEqual(
            [row["evidence_id"] for row in compact[0]["events"]], ["EV001"]
        )
        self.assertNotIn("visual_detail", compact[0]["events"][0])
        prompt = build_story_plan_from_evidence_prompt(evidence, 600, 150)
        self.assertNotIn("SECRET_VISUAL_DETAIL", prompt)
        self.assertNotIn("Mechanical split", prompt)
        self.assertNotIn("Repeated walking shot", prompt)
        self.assertIn("The real turning point", prompt)

    def test_compact_story_evidence_splits_without_loss_or_duplication(self):
        compact = [{
            "chunk_number": 1,
            "events": [
                {"evidence_id": f"EV{index:03d}", "event": f"event {index}"}
                for index in range(125)
            ],
        }]
        groups = split_compact_story_evidence(compact, maximum_events=60)
        counts = [sum(len(chunk["events"]) for chunk in group) for group in groups]
        self.assertEqual(counts, [60, 60, 5])
        ids = [
            event["evidence_id"]
            for group in groups for chunk in group for event in chunk["events"]
        ]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 125)

    def test_timeout_digest_prompts_keep_causal_schema(self):
        digest_prompt = build_story_evidence_digest_prompt(
            [{"events": [{"evidence_id": "EV001", "event": "turn"}]}],
            1, 2,
        )
        self.assertIn("source_evidence_ids", digest_prompt)
        plan_prompt = build_story_plan_from_digests_prompt(
            [{"story_beats": [{"summary": "turn"}]}], 600, 150
        )
        self.assertIn("boundary_1_seconds", plan_prompt)
        self.assertIn("ending", plan_prompt.lower())

    def test_compact_story_plan_reuses_locked_events(self):
        evidence = [{"events": [
            {
                "start_seconds": float(index * 10 + 1),
                "end_seconds": float(index * 10 + 4),
                "event": f"event {index}",
                "importance": "main",
                "story_role": "ending" if index == 11 else "development",
            }
            for index in range(12)
        ]}]
        plan = json.dumps({
            "boundary_1_seconds": 40,
            "boundary_2_seconds": 80,
            "part_weights": [1, 1, 1],
            "parts": [
                {"part": 1, "summary": "a", "end_state": "a", "handoff_to_next": "b"},
                {"part": 2, "summary": "b", "end_state": "b", "handoff_to_next": "c"},
                {"part": 3, "summary": "c", "end_state": "c", "handoff_to_next": ""},
            ],
        })
        result = normalize_story_plan_from_evidence(plan, evidence, 120, 120)
        selected = [event for part in result["events_by_part"] for event in part]
        self.assertGreaterEqual(len(selected), required_tts_scene_count(120))
        self.assertLessEqual(len(selected) * 15, 120)
        self.assertFalse(any(event.get("derived_visual_slot") for event in selected))
        self.assertTrue(all(event["event"].startswith("event ") for event in selected))
        self.assertTrue(any(event["story_role"] == "ending" for event in result["events_by_part"][2]))

    def test_editorial_contract_requires_six_grounded_anchor_groups(self):
        evidence = [{"events": [{
            "evidence_id": f"EV{index:03d}",
            "start_seconds": float(index * 10),
            "end_seconds": float(index * 10 + 3),
            "event": f"event {index}",
            "importance": "main",
            "story_role": "ending" if index == 12 else "development",
        } for index in range(1, 13)]}]
        base = {
            "boundary_1_seconds": 40,
            "boundary_2_seconds": 80,
            "part_weights": [1, 1, 1],
            "parts": [{"part": 1}, {"part": 2}, {"part": 3}],
        }
        with self.assertRaisesRegex(ValueError, "EDITORIAL_ANCHORS_MISSING"):
            normalize_story_plan_from_evidence(
                json.dumps(base), evidence, 130, 120,
                require_editorial_contract=True,
            )
        base.update({
            "editorial_contract_version": 1,
            "editorial_anchors": {
                "opening": ["EV001"],
                "inciting_incident": ["EV002"],
                "protagonist_goal": ["EV003"],
                "climax": ["EV009"],
                "resolution": ["EV010"],
                "ending": ["EV012"],
            },
        })
        result = normalize_story_plan_from_evidence(
            json.dumps(base), evidence, 130, 120,
            require_editorial_contract=True,
        )
        self.assertEqual(result["editorial_contract_version"], 1)
        self.assertEqual(result["editorial_anchors"]["ending"], ["EV012"])

    def test_missing_ending_label_uses_final_resolution_without_failure(self):
        evidence = [{"events": [
            {
                "evidence_id": f"EV{index:03d}",
                "start_seconds": float(index * 10 + 1),
                "end_seconds": float(index * 10 + 4),
                "event": f"event {index}",
                "importance": "main",
                "story_role": "resolution" if index == 11 else "development",
                "narrative_function": (
                    "resolution" if index == 11 else "development"
                ),
            }
            for index in range(12)
        ]}]
        plan = json.dumps({
            "boundary_1_seconds": 40,
            "boundary_2_seconds": 80,
            "part_weights": [1, 1, 1],
            "parts": [{"part": 1}, {"part": 2}, {"part": 3}],
        })
        result = normalize_story_plan_from_evidence(plan, evidence, 120, 120)
        endings = [
            event for event in result["events_by_part"][2]
            if event.get("story_role") == "ending"
        ]
        self.assertEqual(len(endings), 1)
        self.assertTrue(endings[0].get("ending_inferred_by_backend"))
        self.assertTrue(result["scene_contract"]["ending_inferred"])

    def test_compact_story_plan_preserves_opening_causal_chain(self):
        functions = [
            "setup", "inciting_incident", "turning_point", "development",
            "development", "development", "climax", "resolution", "ending",
        ]
        seconds = [5, 10, 15, 25, 45, 65, 85, 105, 115]
        events = []
        for index, (second, function) in enumerate(zip(seconds, functions), start=1):
            events.append({
                "evidence_id": f"EV001_{index:04d}",
                "start_seconds": second, "end_seconds": second + 2,
                "event": f"event {index}", "importance": "main",
                "narrative_function": function,
                "depends_on": [f"EV001_{index - 1:04d}"] if 1 < index <= 3 else [],
                "story_role": "ending" if function == "ending" else (
                    "resolution" if function == "resolution" else
                    "climax" if function == "climax" else "development"
                ),
            })
        evidence = [{"events": events}]
        plan = json.dumps({
            "boundary_1_seconds": 40, "boundary_2_seconds": 80,
            "part_weights": [1, 1, 1],
            "parts": [
                {"part": 1, "summary": "mở", "end_state": "a"},
                {"part": 2, "summary": "giữa", "end_state": "b"},
                {"part": 3, "summary": "cuối", "end_state": "c"},
            ],
        })
        result = normalize_story_plan_from_evidence(plan, evidence, 120, 120)
        selected = [
            event for part in result["events_by_part"] for event in part
        ]
        selected_source_ids = {
            evidence_id
            for event in selected
            for evidence_id in (
                event.get("merged_evidence_ids") or [event.get("evidence_id")]
            )
        }
        self.assertTrue({
            "EV001_0001", "EV001_0002", "EV001_0003"
        }.issubset(selected_source_ids))

    def test_global_cluster_selection_restores_bounded_action_neighborhood(self):
        kinds = [
            "reaction", "discovery", "transformation", "attack", "reveal",
            "defense", "consequence", "reaction", "reaction",
        ]
        events = []
        for index, kind in enumerate(kinds, start=1):
            events.append({
                "evidence_id": f"EV{index:03d}",
                "start_seconds": index * 20.0,
                "end_seconds": index * 20.0 + 8.0,
                "event": f"event {index}",
                "event_kind": kind,
                "importance": "main",
                "narrative_function": "development",
                "repeat_group": "same-reaction" if index in {8, 9} else "",
            })
        payload = {"selected_event_clusters": [{
            "cluster_id": "C001",
            "importance": "critical",
            "event_ids": ["EV005"],
            "required_context_ids": [],
        }]}
        selected = select_causal_story_evidence(
            events, payload, 120, boundaries=[0, 70, 140, 220]
        )
        selected_ids = {event["evidence_id"] for event in selected}
        self.assertTrue({
            "EV002", "EV003", "EV004", "EV005", "EV006", "EV007",
        }.issubset(selected_ids))
        self.assertLessEqual(len({"EV008", "EV009"} & selected_ids), 1)

    def test_global_selection_keeps_whole_locked_local_climax_cluster(self):
        events = [
            {
                "evidence_id": "EV001", "start_seconds": 100,
                "end_seconds": 106, "event": "The group is discovered",
                "event_kind": "discovery", "event_cluster": "CH001_FINAL",
                "narrative_function": "development", "importance": "main",
            },
            {
                "evidence_id": "EV002", "start_seconds": 107,
                "end_seconds": 114, "event": "The hero transforms",
                "event_kind": "transformation", "event_cluster": "CH001_FINAL",
                "narrative_function": "climax", "story_role": "climax",
                "importance": "main",
            },
            {
                "evidence_id": "EV003", "start_seconds": 115,
                "end_seconds": 124, "event": "The enemy counterattacks",
                "event_kind": "attack", "event_cluster": "CH001_FINAL",
                "narrative_function": "development", "importance": "main",
            },
            {
                "evidence_id": "EV004", "start_seconds": 125,
                "end_seconds": 132, "event": "The battle is resolved",
                "event_kind": "consequence", "event_cluster": "CH001_FINAL",
                "narrative_function": "resolution", "story_role": "resolution",
                "importance": "main",
            },
        ]
        selected = select_causal_story_evidence(
            events,
            {"selected_event_clusters": [{
                "cluster_id": "C001", "importance": "critical",
                "event_ids": ["EV002"], "required_context_ids": [],
            }]},
            40,
            boundaries=[0, 90, 120, 180],
        )
        self.assertEqual(
            {event["evidence_id"] for event in selected},
            {"EV001", "EV002", "EV003", "EV004"},
        )

    def test_global_selection_scales_real_scene_count_with_requested_minutes(self):
        events = [{
            "evidence_id": f"EV{index:03d}",
            "start_seconds": float(index * 5),
            "end_seconds": float(index * 5 + 3),
            "event": f"New story change {index}",
            "event_kind": "action",
            "importance": "main",
            "narrative_function": "development",
            "story_role": "ending" if index == 120 else "development",
        } for index in range(1, 121)]
        selected = select_causal_story_evidence(
            events, {"selected_event_clusters": []}, 2200,
            boundaries=[0, 200, 400, 700],
        )
        self.assertEqual(len(selected), required_tts_scene_count(2200))

    def test_story_plan_prompt_requests_global_event_clusters(self):
        evidence = [{"events": [{
            "evidence_id": "EV001", "start_seconds": 1, "end_seconds": 3,
            "event": "A threat appears", "importance": "main",
        }]}]
        prompt = build_story_plan_from_evidence_prompt(evidence, 600, 150)
        self.assertIn("selected_event_clusters", prompt)
        self.assertIn("required_context_ids", prompt)
        self.assertIn("editorial_anchors", prompt)
        self.assertIn("protagonist_goal", prompt)
        self.assertIn("khong chia deu", prompt.lower())

    def test_each_ui_style_has_a_distinct_storytelling_instruction(self):
        styles = ["dramatic", "humorous", "shorts", "news", "tutorial"]
        prompts = [style_storytelling_prompt(style) for style in styles]
        self.assertEqual(len(set(prompts)), len(styles))
        self.assertIn("nguyên nhân-kết quả", prompts[0])
        self.assertIn("dí dỏm", prompts[1])
        self.assertIn("Kể nhanh", prompts[2])
        self.assertIn("bản tin", prompts[3])
        self.assertIn("giải thích sâu", prompts[4])

    def test_targets_preserve_total(self):
        targets = three_part_targets(10)
        self.assertEqual(targets, [4, 3, 3])
        self.assertEqual(sum(targets), 10)

    def test_proportional_targets_follow_story_importance(self):
        targets = proportional_part_targets([1, 3, 6], 100)
        self.assertEqual(sum(targets), 100)
        self.assertLess(targets[0], targets[1])
        self.assertLess(targets[1], targets[2])

    def test_story_gap_retry_preserves_old_anchors_and_adds_new_ones(self):
        old = {
            "boundary_1_seconds": 30, "boundary_2_seconds": 60,
            "parts": [
                {"part": 1, "summary": "mở", "required_events": [{
                    "event_id": "E1", "start_seconds": 5, "end_seconds": 8,
                    "event": "mốc cũ",
                }]},
                {"part": 2, "required_events": []},
                {"part": 3, "required_events": []},
            ],
        }
        supplement = {
            "boundary_1_seconds": 35, "boundary_2_seconds": 65,
            "parts": [
                {"part": 1, "required_events": [
                    {"event_id": "E1", "start_seconds": 5, "end_seconds": 8,
                     "event": "không được ghi đè"},
                    {"event_id": "E4", "start_seconds": 20, "end_seconds": 24,
                     "event": "mốc bổ sung"},
                ]},
                {"part": 2, "required_events": []},
                {"part": 3, "required_events": []},
            ],
        }
        merged = json.loads(merge_story_plan_supplement(
            json.dumps(old, ensure_ascii=False),
            json.dumps(supplement, ensure_ascii=False),
        ))
        events = merged["parts"][0]["required_events"]
        self.assertEqual(merged["boundary_1_seconds"], 30)
        self.assertEqual([row["event_id"] for row in events], ["E1", "E4"])
        self.assertEqual(events[0]["start_seconds"], 5)
        prompt = build_story_gap_supplement_prompt(
            json.dumps(old, ensure_ascii=False), "EVENT_STORY_GAP",
        )
        self.assertIn("không phải tạo lại từ đầu", prompt)
        self.assertIn("Không xóa, đổi ID, đổi timestamp", prompt)

    def test_story_gap_retry_reassigns_reused_id_for_a_new_anchor(self):
        old = {
            "boundary_1_seconds": 30, "boundary_2_seconds": 60,
            "parts": [
                {"part": 1, "required_events": [{
                    "event_id": "E0001", "start_seconds": 5,
                    "end_seconds": 8, "event": "mốc đã giữ",
                }]},
                {"part": 2, "required_events": []},
                {"part": 3, "required_events": []},
            ],
        }
        supplement = {
            "boundary_1_seconds": 30, "boundary_2_seconds": 60,
            "parts": [
                {"part": 1, "required_events": [{
                    "event_id": "E0001", "start_seconds": 20,
                    "end_seconds": 24, "event": "mốc mới nhưng Gemini tái dùng ID",
                }]},
                {"part": 2, "required_events": []},
                {"part": 3, "required_events": []},
            ],
        }
        merged = json.loads(merge_story_plan_supplement(
            json.dumps(old, ensure_ascii=False),
            json.dumps(supplement, ensure_ascii=False),
        ))
        events = merged["parts"][0]["required_events"]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_id"], "E0001")
        self.assertEqual(events[1]["event_id"], "E0002")

    def test_required_events_and_word_floor_trigger_revision(self):
        raw = {
            "parts": [{
                "part": 1,
                "video_end": "01:40",
                "scenes": [{
                    "start": "00:01", "end": "01:35",
                    "tts": "mot hai ba", "covered_event_ids": ["E0001"],
                }],
            }],
        }
        requirements = single_part_missing_requirements(
            json.dumps(raw),
            100,
            100,
            required_event_ids=["E0001", "E0002"],
            require_word_floor=True,
            required_ending_sec=95,
        )

        self.assertEqual(requirements["missing_event_ids"], ["E0002"])
        self.assertEqual(requirements["word_deficit"], 94)
        self.assertTrue(requirements["needs_revision"])
        self.assertEqual(requirements["action"], "events")

    def test_each_required_event_must_map_to_exactly_one_scene(self):
        raw = {
            "parts": [{
                "part": 1,
                "video_end": "00:30",
                "scenes": [
                    {
                        "scene_id": "S1", "start": "00:01", "end": "00:10",
                        "tts": " ".join(["tu"] * 20),
                        "covered_event_ids": ["E1", "E2"],
                    },
                    {
                        "scene_id": "S2", "start": "00:11", "end": "00:20",
                        "tts": " ".join(["tu"] * 20),
                        "covered_event_ids": ["E1"],
                    },
                ],
            }],
        }
        requirements = single_part_missing_requirements(
            json.dumps(raw), 30, 40,
            required_event_ids=["E1", "E2"], required_scene_count=2,
            require_tts_word_range=True,
        )

        self.assertEqual(requirements["missing_event_ids"], [])
        self.assertTrue(requirements["coverage_missing"])
        self.assertEqual(requirements["action"], "events")
        self.assertEqual(
            [item["issue"] for item in requirements["invalid_event_mappings"]],
            ["multiple_ids"],
        )
        with self.assertRaisesRegex(ValueError, "ánh xạ event/TTS không hợp lệ"):
            validate_single_part_response(
                json.dumps(raw), 1, 40,
                required_event_ids=["E1", "E2"], required_scene_count=2,
            )

    def test_event_map_ending_does_not_have_to_reach_end_of_clip(self):
        raw = {
            "parts": [{
                "part": 3,
                "video_end": "10:00",
                "scenes": [{
                    "scene_id": "S0001",
                    "start": "07:00",
                    "end": "07:20",
                    "tts": "Ket thuc that cua cau chuyen da duoc ke day du trong canh nay.",
                    "covered_event_ids": ["E_END"],
                }],
            }],
        }
        requirements = single_part_missing_requirements(
            json.dumps(raw),
            600,
            15,
            required_event_ids=["E_END"],
            required_ending_sec=590,
        )

        self.assertFalse(requirements["ending_missing"])
        self.assertFalse(requirements["coverage_missing"])
        validate_single_part_response(
            json.dumps(raw),
            3,
            15,
            required_event_ids=["E_END"],
            required_ending_sec=590,
        )

    def test_tts_outside_15_to_25_words_does_not_trigger_revision(self):
        raw = {
            "parts": [{
                "part": 1,
                "video_end": "00:30",
                "scenes": [
                    {
                        "scene_id": "S0001", "start": "00:01", "end": "00:10",
                        "tts": "Nobita vội chạy về nhà sau khi phát hiện một dấu vết rất lạ.",
                    },
                    {
                        "scene_id": "S0002", "start": "00:12", "end": "00:28",
                        "tts": (
                            "Doraemon lập tức sử dụng bảo bối để lần theo dấu vết, "
                            "nhưng cả nhóm nhanh chóng nhận ra kẻ lạ đã cố tình dẫn "
                            "họ sang một con đường hoàn toàn khác."
                        ),
                    },
                ],
            }],
        }
        requirements = single_part_missing_requirements(
            json.dumps(raw), 30, 40, require_tts_word_range=True
        )

        self.assertEqual(requirements["action"], "ok")
        self.assertEqual(requirements["invalid_tts_scenes"], [])
        self.assertEqual(len(requirements["tts_length_warnings"]), 2)
        validate_single_part_response(
            json.dumps(raw), 1, 40,
            require_coverage=False,
            require_tts_word_range=True,
        )

    def test_soft_tts_issue_does_not_count_as_hard_part_failure(self):
        requirements = {
            "coverage_missing": False,
            "invalid_timestamp_scenes": [],
            "scene_count_gap": 0,
            "invalid_tts_scenes": [{
                "scene_id": "S0055", "word_count": 21,
                "fragmented": True,
            }],
        }
        self.assertFalse(single_part_has_hard_failures(requirements))
        requirements["scene_count_gap"] = 1
        self.assertTrue(single_part_has_hard_failures(requirements))

    def test_required_scene_count_triggers_revision(self):
        raw = {
            "parts": [{
                "part": 1,
                "video_end": "00:30",
                "scenes": [{
                    "scene_id": "S0001", "start": "00:01", "end": "00:20",
                    "tts": " ".join(["tu"] * 20),
                    "covered_event_ids": ["E0001"],
                }],
            }],
        }
        requirements = single_part_missing_requirements(
            json.dumps(raw), 30, 60,
            required_event_ids=["E0001"], required_scene_count=3,
        )
        self.assertEqual(requirements["action"], "scene_count")
        self.assertEqual(requirements["scene_count_gap"], 2)
        with self.assertRaisesRegex(ValueError, "cần đúng 3 scene/TTS"):
            validate_single_part_response(
                json.dumps(raw), 1, 60,
                require_coverage=False,
                required_scene_count=3,
            )

    def test_tts_between_15_and_25_words_is_accepted_with_beat(self):
        words = ["Nam", "bước", "vào", "phòng", "rồi", "phát", "hiện", "người", "bạn",
                 "đang", "bị", "trói", "bên", "chiếc", "ghế", "gỗ"]
        narration = " ".join(words)
        raw = {
            "parts": [{
                "part": 1,
                "video_end": "00:20",
                "scenes": [{
                    "scene_id": "S0001", "start": "00:01", "end": "00:20",
                    "tts": narration,
                    "visual_beats": [{
                        "trigger_text": "phát hiện người bạn",
                        "start": "00:10", "end": "00:13",
                        "match_type": "literal", "confidence": 0.9,
                    }],
                }],
            }],
        }
        requirements = single_part_missing_requirements(
            json.dumps(raw), 20, 16, require_tts_word_range=True
        )
        self.assertEqual(requirements["invalid_tts_scenes"], [])

    def test_requested_word_range_stays_close_to_requested_duration(self):
        self.assertEqual(requested_word_range(300), (300, 291, 309))
        self.assertEqual(requested_word_range(600), (600, 582, 618))

    def test_expanded_word_target_no_longer_adds_hidden_words(self):
        self.assertEqual(expanded_word_target(90), 90)
        self.assertEqual(expanded_word_target(150), 150)
        self.assertEqual(expanded_word_target(300), 300)
        self.assertEqual(expanded_word_target(600), 600)

    def test_required_tts_scene_count_uses_twenty_word_average(self):
        self.assertEqual(required_tts_scene_count(90), 5)
        self.assertEqual(required_tts_scene_count(350), 18)

    def test_backend_locks_global_scene_ids_and_exact_word_budget(self):
        parts = [
            [{"event_id": "E1", "story_role": "development"}],
            [{"event_id": "E2", "story_role": "climax"}],
            [{"event_id": "E3", "story_role": "ending"}],
        ]
        contract = lock_story_scene_contract(parts, 60)
        flat = [event for part in parts for event in part]
        self.assertEqual(contract["scene_count"], 3)
        self.assertEqual([event["scene_id"] for event in flat], [
            "S0001", "S0002", "S0003",
        ])
        self.assertEqual(sum(event["tts_target_words"] for event in flat), 60)
        self.assertLessEqual(flat[-1]["tts_target_words"], 20)
        self.assertEqual(sum(contract["part_word_targets"]), 60)

    def test_backend_rejects_impossible_word_scene_contract(self):
        with self.assertRaisesRegex(ValueError, "WORD_SCENE_CONTRACT_INFEASIBLE"):
            lock_story_scene_contract([
                [{"event_id": "E1", "story_role": "ending"}], [], [],
            ], 30)

    def test_batch_validation_rejects_missing_locked_story_anchor(self):
        payload = {"parts": [{"part": 1, "scenes": [{
            "scene_id": "S1", "start": "00:01", "end": "00:04",
            "tts": "Một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba mười bốn mười lăm",
            "covered_event_ids": ["P1B1E0001"],
            "covered_story_anchor_ids": [],
        }]}]}
        requirements = single_part_missing_requirements(
            json.dumps(payload, ensure_ascii=False), 60, 15,
            require_tts_word_range=True, required_scene_count=1,
            required_story_anchor_ids=["E0001"],
        )
        self.assertEqual(requirements["missing_story_anchor_ids"], ["E0001"])
        self.assertTrue(requirements["coverage_missing"])

    def test_batch_validation_requires_explicit_transition_for_hard_cut(self):
        payload = {"parts": [{"part": 1, "scenes": [{
            "scene_id": "S1", "start": "00:01", "end": "00:04",
            "tts": "Ở phía xa kẻ địch cũng bắt đầu chuẩn bị cuộc truy đuổi nhằm chặn nhóm bạn trước cửa hang",
            "relation_to_previous": "parallel", "transition_text": "",
            "state_before": "nhóm đang chạy trốn", "state_after": "kẻ địch bắt đầu truy đuổi",
        }]}]}
        requirements = single_part_missing_requirements(
            json.dumps(payload, ensure_ascii=False), 60, 18,
            require_narrative_links=True, required_ending_sec=0,
        )
        self.assertEqual(requirements["action"], "continuity")
        self.assertTrue(requirements["invalid_narrative_links"])

        payload["parts"][0]["scenes"][0]["tts"] = (
            "Trong khi đó, kẻ địch bắt đầu chuẩn bị cuộc truy đuổi nhằm chặn nhóm bạn trước cửa hang"
        )
        payload["parts"][0]["scenes"][0]["transition_text"] = "Trong khi đó"
        requirements = single_part_missing_requirements(
            json.dumps(payload, ensure_ascii=False), 60, 16,
            require_narrative_links=True, required_ending_sec=0,
        )
        self.assertFalse(requirements["invalid_narrative_links"])

    def test_batch_validation_detects_scenes_clustered_at_clip_start(self):
        scenes = []
        for index, (start, end) in enumerate(
            (("00:10", "00:13"), ("00:30", "00:33"),
             ("00:50", "00:53"), ("01:10", "01:13")),
            start=1,
        ):
            scenes.append({
                "scene_id": f"S{index}", "start": start,
                "end": end, "tts": "từ " * 15,
            })
        payload = {"parts": [{"part": 1, "scenes": scenes}]}
        requirements = single_part_missing_requirements(
            json.dumps(payload), 900, 60, require_timeline_spread=True,
            required_ending_sec=0,
        )
        self.assertEqual(requirements["action"], "timeline_spread")
        self.assertIsNotNone(requirements["timeline_clustering"])

    @unittest.skip("legacy verbose writer prompt replaced by compact schema")
    def test_single_part_prompt_requires_event_map_and_word_floor(self):
        prompt = build_single_part_video_prompt(2, 30, 30, 60, 90)
        self.assertIn("phần 2/3", prompt)
        self.assertIn("target_words=90", prompt)
        self.assertIn("ít nhất 88 từ", prompt)
        self.assertIn("KHÔNG có video", prompt)
        self.assertIn("covered_event_ids", prompt)
        self.assertIn("đúng một TTS từ 15-25 từ", prompt)
        self.assertIn("Không gộp nhiều event", prompt)
        self.assertIn("đúng 4 scene/TTS", prompt)
        self.assertIn("kể một ý trọn vẹn và đọc tự nhiên", prompt)
        self.assertIn("thời gian CỤC BỘ", prompt)
        self.assertIn("thêm visual_beat", prompt)
        self.assertIn("không trả lời bằng tiếng Anh", prompt)

    @unittest.skip("backend now owns visual timing")
    def test_single_part_prompt_treats_three_second_shot_as_soft_target(self):
        prompt = build_single_part_video_prompt(1, 60, 0, 60, 60)
        self.assertIn("khoảng 3 giây", prompt)
        self.assertIn("không bắt buộc đúng 3 giây", prompt)

    @unittest.skip("legacy wording replaced; floor is tested through contract helpers")
    def test_single_part_prompt_enforces_97_percent_floor(self):
        prompt = build_single_part_video_prompt(1, 600, 0, 600, 300)
        self.assertIn("target_words=300", prompt)
        self.assertIn("ít nhất 291 từ", prompt)
        self.assertIn("hướng tới khoảng 300 từ", prompt)

    @unittest.skip("visual beats are backend-owned in compact writer schema")
    def test_single_part_prompt_uses_contextual_nonempty_beat_example(self):
        prompt = build_single_part_video_prompt(1, 600, 0, 600, 300)
        self.assertIn("Một câu có thể có 0, 1 hoặc nhiều beat", prompt)
        self.assertIn('"trigger_text":"người bạn hoảng hốt"', prompt)
        self.assertNotIn('"story_role":"development","visual_beats":[]', prompt)

    def test_merge_single_parts_offsets_timestamps_and_scene_ids(self):
        responses = []
        for part_number in range(1, 4):
            raw = {
                "characters": [{"id": "P1", "name": "Nam", "description": "ao den"}],
                "parts": [{
                    "part": part_number,
                    "video_start": "00:00",
                    "video_end": "00:30",
                    "scenes": [{
                        "scene_id": "S0001", "start": "00:02", "end": "00:08",
                        "type": "main", "characters": ["P1"], "event": "x",
                        "tts": "Nam tiep tuc hanh dong trong cau chuyen dang dien ra truoc mat.",
                        "story_role": "ending" if part_number == 3 else "development",
                        "visual_beats": [], "status": "ok", "errors": [],
                    }],
                    "reserve_scenes": [],
                }],
            }
            responses.append(json.dumps(raw))

        payload = merge_single_part_responses(responses, [0, 30, 60, 90], 18)

        self.assertEqual(
            [part["scenes"][0]["start"] for part in payload["parts"]],
            ["00:02.00", "00:32.00", "01:02.00"],
        )
        self.assertEqual(
            [part["scenes"][0]["scene_id"] for part in payload["parts"]],
            ["S0001", "S0002", "S0003"],
        )

    def test_merge_repairs_equal_global_timestamps_and_renumbers_ids(self):
        responses = []
        for part_number in range(1, 4):
            scenes = [{
                "scene_id": f"S{part_number:04d}",
                "start": "00:02", "end": "00:08",
                "tts": "Mot cau recap co du noi dung de kiem tra thu tu canh.",
                "story_role": "ending" if part_number == 3 else "development",
            }]
            if part_number == 2:
                scenes.insert(0, {
                    "scene_id": "S0099", "start": "00:02", "end": "00:10",
                    "tts": "Cau thu hai cung timestamp van phai duoc backend xu ly.",
                    "story_role": "development",
                })
            responses.append(json.dumps({
                "characters": [],
                "parts": [{
                    "part": part_number, "video_start": "00:00",
                    "video_end": "00:30", "scenes": scenes,
                    "reserve_scenes": [],
                }],
            }))
        payload = merge_single_part_responses(
            responses, [0, 30, 60, 90], 72,
        )
        rows = [scene for part in payload["parts"] for scene in part["scenes"]]
        self.assertEqual(
            [scene["scene_id"] for scene in rows],
            ["S0001", "S0002", "S0003", "S0004"],
        )
        starts = [_time_seconds(scene["start"]) for scene in rows]
        self.assertEqual(starts, sorted(set(starts)))

    def test_single_part_validation_does_not_require_scene_capacity(self):
        raw = {
            "parts": [{
                "part": 1, "video_end": "00:10",
                "scenes": [{
                    "start": "00:01", "end": "00:10",
                    "tts": "mot hai ba bon nam sau bay tam",
                }],
                "reserve_scenes": [],
            }],
        }
        validated = validate_single_part_response(json.dumps(raw), 1, 6)
        self.assertEqual(len(validated["parts"][0]["scenes"]), 1)

    def test_single_part_validation_accepts_too_few_actual_words(self):
        raw = {
            "parts": [{
                "part": 1, "video_end": "00:10",
                "scenes": [
                    {"start": "00:01", "end": "00:05", "tts": "qua ngan"},
                    {"start": "00:06", "end": "00:10", "tts": "van qua ngan"},
                ],
                "reserve_scenes": [],
            }],
        }
        validated = validate_single_part_response(json.dumps(raw), 1, 26)
        self.assertEqual(len(validated["parts"][0]["scenes"]), 2)

    def test_single_part_reports_missing_timestamp_before_strict_validation(self):
        raw = {"parts": [{
            "part": 1, "video_end": "00:10",
            "scenes": [{"scene_id": "S1", "start": "00:01", "end": None, "tts": "noi dung"}],
        }]}
        requirements = single_part_missing_requirements(json.dumps(raw), 10, 20)
        self.assertEqual(requirements["action"], "timestamps")
        self.assertEqual(
            requirements["invalid_timestamp_scenes"][0]["reasons"],
            ["missing_end"],
        )
        with self.assertRaisesRegex(ValueError, "S1.*missing_end"):
            validate_single_part_response(json.dumps(raw), 1, 20)

    def test_single_part_validation_accepts_words_above_target(self):
        sentence = "mot hai ba bon nam sau bay tam chin muoi mot muoi hai"
        raw = {
            "parts": [{
                "part": 1,
                "video_end": "00:10",
                "scenes": [{"start": "00:01", "end": "00:10", "tts": sentence}],
                "reserve_scenes": [],
            }],
        }
        validated = validate_single_part_response(
            json.dumps(raw), 1, 10, require_coverage=False
        )
        self.assertEqual(len(validated["parts"][0]["scenes"]), 1)

    def test_single_part_detects_missing_final_coverage_without_scene_quota(self):
        sentence = "mot hai ba bon nam sau bay tam chin muoi muoi mot muoi hai muoi ba"
        raw = {
            "parts": [{
                "part": 1, "video_start": "00:00", "video_end": "01:40",
                "scenes": [
                    {"start": "00:01", "end": "00:05", "tts": sentence},
                    {"start": "00:10", "end": "00:14", "tts": sentence},
                    {"start": "00:20", "end": "00:24", "tts": sentence},
                ],
                "reserve_scenes": [],
            }],
        }
        requirements = single_part_missing_requirements(json.dumps(raw), 100, 45)
        self.assertTrue(requirements["coverage_missing"])
        self.assertTrue(requirements["ending_missing"])

    def test_single_part_coverage_distance_rewards_a_later_retry(self):
        def response_ending_at(end):
            return json.dumps({
                "parts": [{
                    "part": 3,
                    "video_start": "00:00",
                    "video_end": "01:40",
                    "scenes": [{
                        "start": "00:01",
                        "end": end,
                        "tts": "Ke lai dien bien cua doan phim.",
                    }],
                    "reserve_scenes": [],
                }],
            })

        original = single_part_missing_requirements(
            response_ending_at("00:30"), 100, 45
        )
        improved_retry = single_part_missing_requirements(
            response_ending_at("01:20"), 100, 45
        )

        self.assertTrue(original["needs_revision"])
        self.assertTrue(improved_retry["needs_revision"])
        self.assertLess(improved_retry["distance"], original["distance"])
        self.assertEqual(original["distance"], 60.0)
        self.assertEqual(improved_retry["distance"], 10.0)

    def test_single_part_allows_unimportant_early_and_middle_regions_to_be_skipped(self):
        raw = {
            "parts": [{
                "part": 1, "video_start": "00:00", "video_end": "01:40",
                "scenes": [{
                    "start": "01:32", "end": "01:40",
                    "tts": "Kết cục thật của phần clip được kể tại đây.",
                }],
                "reserve_scenes": [],
            }],
        }
        requirements = single_part_missing_requirements(json.dumps(raw), 100, 45)
        self.assertFalse(requirements["coverage_missing"])
        self.assertFalse(requirements["needs_revision"])

    def test_single_part_word_difference_does_not_request_revision(self):
        raw = {
            "parts": [{
                "part": 1, "video_start": "00:00", "video_end": "01:30",
                "scenes": [{
                    "start": "00:01", "end": "01:30",
                    "tts": "mot hai ba bon nam sau bay tam chin muoi",
                }],
                "reserve_scenes": [],
            }],
        }
        requirements = single_part_missing_requirements(json.dumps(raw), 90, 90)
        self.assertEqual(requirements["action"], "ok")
        self.assertFalse(requirements["needs_revision"])
        self.assertEqual(requirements["requested_words"], 90)

    def test_invalid_continuation_may_keep_word_complete_base_result(self):
        sentence = "mot hai ba bon nam sau bay tam chin muoi muoi mot muoi hai muoi ba"
        raw = {
            "parts": [{
                "part": 1, "video_start": "00:00", "video_end": "01:40",
                "scenes": [
                    {"start": "00:01", "end": "00:05", "tts": sentence},
                    {"start": "00:10", "end": "00:14", "tts": sentence},
                    {"start": "00:20", "end": "00:24", "tts": sentence},
                ],
                "reserve_scenes": [],
            }],
        }
        with self.assertRaisesRegex(ValueError, "chưa kể đến diễn biến cuối clip"):
            validate_single_part_response(json.dumps(raw), 1, 39)
        validated = validate_single_part_response(
            json.dumps(raw), 1, 39, require_coverage=False
        )
        self.assertEqual(validated["parts"][0]["part"], 1)

    def test_visual_beat_is_normalized_and_encoded_at_trigger_phrase(self):
        raw = sample_payload()
        raw["parts"][0]["scenes"][0].update({
            "tts": "Nam mo cua, nguoi dan ong lap tuc rut sung.",
            "visual_beats": [{
                "trigger_text": "nguoi dan ong",
                "start": "00:10",
                "end": "00:13",
                "match_type": "literal",
                "confidence": 0.91,
            }],
        })
        payload = normalize_three_part_payload(json.dumps(raw), 90, 18)
        timeline = payload_to_timeline(payload)
        self.assertIn(
            "Nam mo cua, [BEAT=00:10.00-00:13.00] nguoi dan ong",
            timeline,
        )

    def test_invalid_or_ambiguous_trigger_is_not_emitted(self):
        raw = sample_payload()
        raw["parts"][0]["scenes"][0].update({
            "tts": "Nam thay Nam roi bo di.",
            "visual_beats": [{
                "trigger_text": "Nam",
                "start": "00:10",
                "end": "00:13",
            }],
        })
        payload = normalize_three_part_payload(json.dumps(raw), 90, 18)
        self.assertNotIn("[BEAT=", payload_to_timeline(payload))

    def test_low_confidence_visual_beat_is_not_emitted(self):
        raw = sample_payload()
        raw["parts"][0]["scenes"][0].update({
            "tts": "Nam mo cua, nguoi dan ong lap tuc rut sung.",
            "visual_beats": [{
                "trigger_text": "nguoi dan ong",
                "start": "00:10",
                "end": "00:13",
                "confidence": 0.55,
            }],
        })
        payload = normalize_three_part_payload(json.dumps(raw), 90, 18)
        self.assertNotIn("[BEAT=", payload_to_timeline(payload))

    def test_visual_beat_near_sentence_opening_is_not_emitted(self):
        raw = sample_payload()
        raw["parts"][0]["scenes"][0].update({
            "tts": "Doraemon su dung vong xuyen thau de thoat ra ngoai.",
            "visual_beats": [{
                "trigger_text": "vong xuyen thau",
                "start": "00:10",
                "end": "00:13",
                "confidence": 0.95,
            }],
        })
        payload = normalize_three_part_payload(json.dumps(raw), 90, 18)
        self.assertNotIn("[BEAT=", payload_to_timeline(payload))

    def test_visual_beat_uses_three_seconds_as_soft_target(self):
        raw = sample_payload()
        raw["parts"][0]["scenes"][0].update({
            "tts": "Nam mo cua, nguoi dan ong lap tuc rut sung.",
            "visual_beats": [{
                "trigger_text": "nguoi dan ong",
                "start": "00:10",
                "end": "00:16",
                "confidence": 0.95,
            }],
        })
        payload = normalize_three_part_payload(json.dumps(raw), 90, 18)
        self.assertIn(
            "[BEAT=00:10.00-00:16.00] nguoi dan ong",
            payload_to_timeline(payload),
        )

    def test_short_visual_beat_is_preserved_when_grounded(self):
        raw = sample_payload()
        raw["parts"][0]["scenes"][0].update({
            "tts": "Nam mo cua, nguoi dan ong lap tuc rut sung.",
            "visual_beats": [{
                "trigger_text": "nguoi dan ong",
                "start": "00:10",
                "end": "00:11.5",
                "confidence": 0.95,
            }],
        })
        payload = normalize_three_part_payload(json.dumps(raw), 90, 18)
        self.assertIn("[BEAT=00:10.00-00:11.50]", payload_to_timeline(payload))

    def test_short_scene_narration_does_not_create_a_scene_quota_issue(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        issues = inspect_three_part_payload(payload)
        short_ids = {
            issue["target_id"] for issue in issues
            if issue["code"] == "tts_too_short"
        }
        self.assertEqual(short_ids, set())

    def test_part_above_word_target_is_not_sent_to_repair(self):
        raw = sample_payload()
        raw["parts"][0]["scenes"][0]["tts"] = " ".join(["dai"] * 40)
        payload = normalize_three_part_payload(json.dumps(raw), 90, 18)
        issues = inspect_three_part_payload(payload)
        excessive = [issue for issue in issues if issue["code"] == "part_too_long"]
        self.assertEqual(excessive, [])

    def test_part_word_count_never_creates_repair_issue(self):
        for word_count in (1, 500):
            payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
            payload["parts"][0]["target_words"] = 100
            payload["parts"][0]["scenes"][0]["tts"] = " ".join(
                ["tu"] * word_count
            )
            budget_issues = [
                issue for issue in inspect_three_part_payload(payload)
                if issue["target_id"] == "PART_1"
                and issue["code"] in {"part_too_short", "part_too_long"}
            ]
            self.assertEqual(budget_issues, [])

    def test_normalize_requires_three_contiguous_parts(self):
        raw = sample_payload()
        normalized = normalize_three_part_payload(json.dumps(raw), 90, 18)
        self.assertEqual([item["part"] for item in normalized["parts"]], [1, 2, 3])
        broken = sample_payload()
        broken["parts"][1]["video_start"] = "00:40"
        with self.assertRaisesRegex(ValueError, "nối tiếp"):
            normalize_three_part_payload(json.dumps(broken), 90, 18)

        unordered = sample_payload()
        unordered["parts"][0], unordered["parts"][1] = (
            unordered["parts"][1], unordered["parts"][0]
        )
        with self.assertRaisesRegex(ValueError, "thứ tự 1, 2, 3"):
            normalize_three_part_payload(json.dumps(unordered), 90, 18)

    def test_normalize_accepts_unequal_story_boundaries_and_word_weights(self):
        raw = sample_payload()
        raw["parts"][0].update({"video_end": "00:10", "target_words": 2})
        raw["parts"][1].update({"video_start": "00:10", "video_end": "01:10", "target_words": 8})
        raw["parts"][2].update({"video_start": "01:10", "video_end": "01:30", "target_words": 8})

        normalized = normalize_three_part_payload(json.dumps(raw), 90, 18)

        self.assertEqual(normalized["parts"][0]["video_end"], "00:10.00")
        self.assertEqual(normalized["parts"][1]["video_end"], "01:10.00")
        targets = [part["target_words"] for part in normalized["parts"]]
        self.assertEqual(sum(targets), 18)
        self.assertGreater(targets[1], targets[0])

    def test_scene_may_cross_part_boundary_when_it_overlaps_the_part(self):
        raw = sample_payload()
        raw["parts"][1]["scenes"][0]["start"] = "00:28"
        raw["parts"][1]["scenes"][0]["end"] = "00:35"

        normalized = normalize_three_part_payload(json.dumps(raw), 90, 18)

        scene = normalized["parts"][1]["scenes"][0]
        self.assertEqual(scene["start"], "00:28.00")
        self.assertEqual(scene["end"], "00:35.00")

    def test_scene_may_be_fully_outside_its_assigned_part(self):
        raw = sample_payload()
        raw["parts"][1]["scenes"][0]["start"] = "00:20"
        raw["parts"][1]["scenes"][0]["end"] = "00:25"

        normalized = normalize_three_part_payload(json.dumps(raw), 90, 18)

        scene = normalized["parts"][1]["scenes"][0]
        self.assertEqual(scene["start"], "00:20.00")
        self.assertEqual(scene["end"], "00:25.00")

    def test_normalize_never_rejects_scene_capacity(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 90)
        self.assertEqual(len(payload["parts"][0]["scenes"]), 1)

    def test_reserve_scene_is_rehomed_by_timestamp(self):
        raw = sample_payload()
        misplaced = raw["parts"][1]["reserve_scenes"].pop()
        raw["parts"][0]["reserve_scenes"].append(misplaced)
        normalized = normalize_three_part_payload(json.dumps(raw), 90, 18)
        self.assertEqual(normalized["parts"][0]["reserve_scenes"], [])
        self.assertEqual(
            normalized["parts"][1]["reserve_scenes"][0]["scene_id"],
            "S0901",
        )

    def test_inspection_never_reports_too_few_scenes(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        for part in payload["parts"]:
            part["target_words"] = 30
        issues = inspect_three_part_payload(payload)
        self.assertFalse(any(issue["code"] == "part_too_few_scenes" for issue in issues))

    def test_inspection_never_reports_too_many_scenes(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        part = payload["parts"][0]
        part["target_words"] = 18
        template = part["scenes"][0]
        for index in range(2, 5):
            extra = dict(template)
            extra.update({
                "scene_id": f"S10{index:02d}",
                "start": f"00:{8 + index * 2:02d}",
                "end": f"00:{9 + index * 2:02d}",
                "type": "bridge",
            })
            part["scenes"].append(extra)
        issues = inspect_three_part_payload(payload)
        self.assertFalse(any(issue["code"] == "part_too_many_scenes" for issue in issues))

    def test_material_uncovered_tail_does_not_imply_missing_ending(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        payload["parts"][0]["video_end"] = "06:40.00"
        payload["parts"][1]["video_start"] = "06:40.00"
        payload["parts"][1]["video_end"] = "13:20.00"
        payload["parts"][2]["video_start"] = "13:20.00"
        payload["parts"][2]["video_end"] = "20:00.00"
        payload["parts"][2]["scenes"][0]["start"] = "14:00.00"
        payload["parts"][2]["scenes"][0]["end"] = "14:10.00"
        issues = inspect_three_part_payload(payload)
        ending = [item for item in issues if item["code"] == "story_ending_missing"]
        self.assertEqual(ending, [])

    def test_normal_credit_tail_does_not_trigger_missing_ending(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        issues = inspect_three_part_payload(payload)
        self.assertFalse(any(item["code"] == "story_ending_missing" for item in issues))

    def test_long_story_does_not_require_an_ending_scene_quota(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        payload["parts"][2]["video_end"] = "20:00.00"
        payload["parts"][2]["scenes"] = []
        for index, role in enumerate(["climax", "climax", "climax", "climax", "ending"]):
            payload["parts"][2]["scenes"].append({
                "scene_id": f"S20{index:02d}",
                "start": f"19:{index * 5:02d}.00",
                "end": f"19:{index * 5 + 4:02d}.00",
                "type": "main",
                "characters": ["P1"],
                "event": "event",
                "tts": "Mot cau ke co du noi dung cu the cho canh phim nay.",
                "story_role": role,
                "visual_beats": [],
                "status": "ok",
                "errors": [],
            })
        issues = inspect_three_part_payload(payload)
        self.assertFalse(any(item["code"] == "story_role_ending_missing" for item in issues))

    def test_final_word_clamp_updates_payload_and_timeline_together(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        payload["parts"][0]["scenes"][0]["tts"] = " ".join(
            f"tu{index}" for index in range(30)
        )
        clamped = clamp_payload_narrations(payload)
        narration = clamped["parts"][0]["scenes"][0]["tts"]
        self.assertEqual(len(narration.split()), 30)
        self.assertIn(narration, payload_to_timeline(clamped))

    def test_inspection_detects_repeated_vocabulary_in_adjacent_scenes(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        payload["parts"][0]["scenes"][0]["tts"] = (
            "Nam mở cánh cửa rồi bước vào căn phòng tối để tìm chiếc chìa khóa thất lạc."
        )
        payload["parts"][1]["scenes"][0]["tts"] = (
            "Nam mở cánh cửa rồi bước vào căn phòng tối để tìm chiếc hộp đang phát sáng."
        )
        issues = inspect_three_part_payload(payload)
        self.assertTrue(any(
            item["target_id"] == "S0002"
            and item["code"] in {"repeated_opening", "repeated_vocabulary"}
            for item in issues
        ))

    def test_inspection_flags_story_ending_before_final_scenes(self):
        payload = normalize_three_part_payload(json.dumps(sample_payload()), 90, 18)
        payload["parts"][0]["scenes"][0]["tts"] = (
            "Kẻ địch hoàn toàn bị đánh bại, để lại sự bình yên cho vùng đất cổ xưa này."
        )
        extra = dict(payload["parts"][2]["scenes"][0])
        extra.update({
            "scene_id": "S0004", "start": "01:20.00", "end": "01:25.00",
            "tts": "Không lâu sau, cả nhóm tiếp tục đối mặt với một hiểm họa mới ở phía trước.",
        })
        payload["parts"][2]["scenes"].append(extra)
        issues = inspect_three_part_payload(payload)
        self.assertTrue(any(
            item["target_id"] == "S0001" and item["code"] == "premature_ending"
            for item in issues
        ))

    def test_evidence_replenishment_prefers_largest_uncovered_gap(self):
        chunks = [
            {
                "start_seconds": 0, "end_seconds": 100,
                "events": [
                    {"start_seconds": 0, "end_seconds": 10},
                    {"start_seconds": 90, "end_seconds": 100},
                ],
            },
            {
                "start_seconds": 100, "end_seconds": 200,
                "events": [
                    {"start_seconds": value, "end_seconds": value + 8}
                    for value in range(100, 200, 10)
                ],
            },
        ]
        plan = plan_evidence_replenishment(chunks, 1, max_chunks=1)
        self.assertEqual(plan, [{"chunk_index": 0, "requested_count": 1}])

    def test_targeted_gap_replenishment_uses_short_windows_and_exact_deficit(self):
        chunks = [{
            "start_seconds": 0, "end_seconds": 600,
            "events": [
                {"start_seconds": 10, "end_seconds": 20},
                {"start_seconds": 280, "end_seconds": 290},
                {"start_seconds": 570, "end_seconds": 580},
            ],
        }]
        windows = plan_targeted_evidence_gap_windows(
            chunks, 7, 600, max_windows=4, window_seconds=90
        )
        self.assertTrue(1 <= len(windows) <= 4)
        self.assertEqual(sum(row["requested_count"] for row in windows), 7)
        self.assertTrue(all(
            29.9 <= row["end_seconds"] - row["start_seconds"] <= 90.1
            for row in windows
        ))
        self.assertTrue(all(row["chunk_index"] == 0 for row in windows))

    def test_targeted_gap_replenishment_keeps_reserve_windows(self):
        chunks = [{
            "start_seconds": 0, "end_seconds": 1000,
            "events": [
                {"start_seconds": 20, "end_seconds": 30},
                {"start_seconds": 240, "end_seconds": 250},
                {"start_seconds": 480, "end_seconds": 490},
                {"start_seconds": 720, "end_seconds": 730},
                {"start_seconds": 960, "end_seconds": 970},
            ],
        }]
        windows = plan_targeted_evidence_gap_windows(
            chunks, 2, 1000, max_windows=4, window_seconds=60
        )
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(row["requested_count"] == 1 for row in windows))

    def test_evidence_replenishment_merges_only_valid_new_video_events(self):
        chunk = {
            "chunk_number": 1,
            "start_seconds": 100.0,
            "end_seconds": 110.0,
            "characters": [{"id": "CH001_C1", "source_id": "C1"}],
            "events": [{
                "evidence_id": "EV001_0001",
                "start_seconds": 101.0,
                "end_seconds": 103.0,
                "event": "existing event",
            }],
        }
        supplement = json.dumps({"events": [
            {
                "start_seconds": 1.0, "end_seconds": 3.0,
                "event": "same moment with different wording",
                "characters": ["C1"],
            },
            {
                "start_seconds": 4.0, "end_seconds": 6.0,
                "event": "a real additional event",
                "characters": ["C1"],
            },
            {
                "start_seconds": 11.0, "end_seconds": 12.0,
                "event": "outside clip",
                "characters": ["C1"],
            },
        ]})
        merged, added, diagnostics = merge_evidence_replenishment(
            chunk, supplement, return_diagnostics=True
        )
        self.assertEqual(added, 1)
        self.assertEqual(len(merged["events"]), 2)
        self.assertEqual(merged["events"][1]["start_seconds"], 104.0)
        self.assertEqual(merged["events"][1]["characters"], ["CH001_C1"])
        self.assertEqual(merged["events"][1]["source"], "video_replenishment")
        self.assertEqual(diagnostics["duplicate_time"], 1)
        self.assertEqual(diagnostics["invalid_timestamp"], 1)

    def test_evidence_replenishment_can_skip_chunks_that_returned_no_new_events(self):
        chunks = [
            {"start_seconds": 0, "end_seconds": 100, "events": []},
            {"start_seconds": 100, "end_seconds": 200, "events": []},
        ]
        plan = plan_evidence_replenishment(
            chunks, 1, max_chunks=1, excluded_indices={1}
        )
        self.assertEqual(plan, [{"chunk_index": 0, "requested_count": 1}])

    def test_evidence_replenishment_repairs_only_missing_visual_event_id(self):
        chunk = {
            "chunk_number": 1,
            "start_seconds": 100.0,
            "end_seconds": 120.0,
            "characters": [],
            "events": [{
                "evidence_id": "EV001_0001",
                "start_seconds": 104.0,
                "end_seconds": 112.0,
                "event": "Nobita chạy khỏi hang",
                "visual_grounding_missing": True,
                "visual_sources": [],
            }],
        }
        supplement = json.dumps({"visual_repairs": [{
            "evidence_id": "EV001_0001",
            "visual_shots": [{
                "start_seconds": 5.0,
                "end_seconds": 7.8,
                "description": "Nobita chạy khỏi cửa hang",
            }],
        }], "events": []}, ensure_ascii=False)
        merged, added, diagnostics = merge_evidence_replenishment(
            chunk, supplement, return_diagnostics=True
        )
        event = merged["events"][0]
        self.assertEqual(added, 1)
        self.assertEqual(diagnostics["visual_repaired"], 1)
        self.assertFalse(event["visual_grounding_missing"])
        self.assertEqual(event["visual_sources"][0]["start_seconds"], 105.0)

    def test_targeted_visual_repair_converts_clip_offsets_for_only_requested_id(self):
        chunks = [{"events": [
            {
                "evidence_id": "EV1", "start_seconds": 100,
                "end_seconds": 110, "event": "Nobita chạy",
                "visual_grounding_missing": True, "visual_sources": [],
            },
            {
                "evidence_id": "EV2", "start_seconds": 120,
                "end_seconds": 123, "event": "Doraemon xuất hiện",
                "visual_grounding_missing": False,
                "visual_sources": [{"start_seconds": 120, "end_seconds": 123}],
            },
        ]}]
        response = json.dumps({"repairs": [{
            "evidence_id": "EV1", "found": True,
            "visual_shots": [{
                "start_offset": 3.0, "end_offset": 5.8,
                "description": "Nobita chạy khỏi nhà",
            }, {
                "start_offset": 7.0, "end_offset": 9.8,
                "description": "Nobita chạy qua con đường",
            }],
        }, {
            "evidence_id": "EV2", "found": True,
            "visual_shots": [{"start_offset": 1, "end_offset": 3}],
        }]}, ensure_ascii=False)
        repaired, repaired_ids = apply_targeted_visual_shot_repairs(
            chunks, response, [{
                "evidence_id": "EV1", "start_seconds": 95, "end_seconds": 115,
            }]
        )
        self.assertEqual(repaired_ids, ["EV1"])
        self.assertEqual(
            repaired[0]["events"][0]["visual_sources"][0]["start_seconds"],
            98.0,
        )
        self.assertFalse(repaired[0]["events"][0]["visual_grounding_missing"])

    def test_targeted_visual_repair_accepts_flexible_semantic_beat(self):
        chunks = [{"events": [{
            "evidence_id": "EV1", "start_seconds": 10, "end_seconds": 20,
            "visual_grounding_missing": True, "visual_sources": [],
        }]}]
        response = json.dumps({"repairs": [{
            "evidence_id": "EV1", "found": True,
            "visual_shots": [{"start_offset": 1, "end_offset": 8}],
        }]})
        repaired, repaired_ids = apply_targeted_visual_shot_repairs(
            chunks, response,
            [{"evidence_id": "EV1", "start_seconds": 10, "end_seconds": 20}],
        )
        self.assertEqual(repaired_ids, ["EV1"])
        self.assertFalse(repaired[0]["events"][0]["visual_grounding_missing"])

    def test_drop_unrepairable_visual_events_removes_only_failed_ids(self):
        chunks = [{"events": [
            {"evidence_id": "EV1", "depends_on": []},
            {"evidence_id": "EV2", "depends_on": ["EV1"]},
            {"evidence_id": "EV3", "depends_on": ["EV1", "EV2"]},
        ]}]
        cleaned, removed = drop_unrepairable_visual_events(chunks, ["EV2"])
        self.assertEqual(removed, 1)
        self.assertEqual(
            [row["evidence_id"] for row in cleaned[0]["events"]],
            ["EV1", "EV3"],
        )
        self.assertEqual(cleaned[0]["events"][1]["depends_on"], ["EV1"])

    def test_evidence_replenishment_prompt_forbids_padding(self):
        prompt = build_evidence_replenishment_prompt({
            "start_seconds": 0, "end_seconds": 20,
            "characters": [], "events": [],
        }, 3)
        self.assertIn("khong tao event dem", prompt)
        self.assertIn("tra it hon so luong yeu cau", prompt)

    def test_assign_story_beats_keeps_scenes_but_groups_story_ideas(self):
        events = [[
            {"event_id": "E0001", "start_seconds": 0, "end_seconds": 5,
             "event": "Mở đầu", "narrative_function": "setup", "importance": "main"},
            {"event_id": "E0002", "start_seconds": 30, "end_seconds": 35,
             "event": "Hình minh họa", "narrative_function": "development"},
            {"event_id": "E0003", "start_seconds": 70, "end_seconds": 75,
             "event": "Biến cố", "narrative_function": "turning_point", "importance": "main"},
            {"event_id": "E0004", "start_seconds": 100, "end_seconds": 105,
             "event": "Hành động tiếp nối", "narrative_function": "development"},
            {"event_id": "E0005", "start_seconds": 350, "end_seconds": 355,
             "event": "Diễn biến mới", "narrative_function": "development"},
        ]]
        grouped, beats = assign_story_beats(events)
        self.assertEqual(len(grouped[0]), 5)
        self.assertEqual(len(beats[0]), 3)
        self.assertEqual(grouped[0][0]["story_beat_id"], grouped[0][1]["story_beat_id"])
        self.assertNotEqual(grouped[0][1]["story_beat_id"], grouped[0][2]["story_beat_id"])
        self.assertFalse(grouped[0][1]["visual_support_only"])

    def test_assign_story_beats_protects_climax_causal_spine(self):
        events = [[
            {"event_id": "E0001", "start_seconds": 0, "end_seconds": 5,
             "event": "Enemy attacks", "story_role": "climax",
             "narrative_function": "development"},
            {"event_id": "E0002", "start_seconds": 10, "end_seconds": 15,
             "event": "Hero finds a counter", "story_role": "climax",
             "narrative_function": "reversal"},
            {"event_id": "E0003", "start_seconds": 20, "end_seconds": 25,
             "event": "Enemy falls", "story_role": "climax",
             "narrative_function": "development"},
        ]]
        grouped, beats = assign_story_beats(events)
        self.assertEqual(
            beats[0][0]["climax_spine_ids"],
            ["E0001", "E0002", "E0003"],
        )
        self.assertEqual(
            [event["climax_spine_role"] for event in grouped[0]],
            ["pressure", "turning_point", "outcome"],
        )
        self.assertTrue(all(
            not event["visual_support_only"] for event in grouped[0]
        ))

    @unittest.skip("legacy verbose writer wording removed")
    def test_single_part_prompt_treats_scene_count_as_distinct_facts(self):
        prompt = build_single_part_video_prompt(
            1, 300, 0, 300, 44,
            story_context={
                "required_events": [
                    {"event_id": "E0001", "story_beat_id": "B001"},
                    {"event_id": "E0002", "story_beat_id": "B001",
                     "visual_support_only": True},
                ],
                "story_beats": [{"story_beat_id": "B001", "summary": "Biến cố chính"}],
            },
        )
        self.assertIn("Số scene là số FACT thật", prompt)
        self.assertIn("không tách một fact thành nhiều câu", prompt)
        self.assertIn("Mỗi required_event bắt buộc tạo đúng", prompt)

    def test_compact_writer_schema_restores_backend_contract(self):
        raw = json.dumps({"scenes": [{
            "event_id": "E0001",
            "tts": "Day la cau recap duoc viet tu su kien da khoa cua backend.",
        }]})
        result = json.loads(coerce_single_part_response(
            raw, 1, 60, 22,
            required_event_ids=["E0001"],
            locked_event_by_id={"E0001": {
                "scene_id": "S0001", "start": "00:01", "end": "00:05",
                "tts_target_words": 22,
                "visual_sources": [{"start": "00:01", "end": "00:04"}],
            }},
            locked_characters=[],
        ))
        scene = result["parts"][0]["scenes"][0]
        self.assertEqual(scene["covered_event_ids"], ["E0001"])
        self.assertEqual(scene["scene_id"], "S0001")
        self.assertEqual(
            (_time_seconds(scene["start"]), _time_seconds(scene["end"])),
            (1, 4),
        )

    def test_coerce_restores_locked_story_beat_metadata(self):
        raw = json.dumps({"scenes": [{
            "scene_id": "S0001", "start": "00:01", "end": "00:04",
            "event": "Sự kiện", "tts": "Một câu kể hợp lệ cho sự kiện đang diễn ra trong câu chuyện.",
            "covered_event_ids": ["E0001"],
        }]}, ensure_ascii=False)
        normalized = json.loads(coerce_single_part_response(
            raw, 1, 10, 20,
            required_event_ids=["E0001"],
            story_beat_by_event_id={"E0001": "B007"},
            locked_event_by_id={"E0001": {
                "start": "00:01", "end": "00:04",
                "narrative_function": "reversal",
                "story_beat_summary": "Counterattack",
                "story_beat_cause": "Enemy attacks",
                "story_beat_outcome": "Hero gains ground",
                "story_beat_position": 2, "story_beat_size": 3,
                "climax_spine_role": "turning_point",
                "visual_support_only": False,
                "visual_sources": [{"start": "00:01", "end": "00:04"}],
            }},
        ))
        scene = normalized["parts"][0]["scenes"][0]
        self.assertEqual(scene["story_beat_id"], "B007")
        self.assertEqual(scene["climax_spine_role"], "turning_point")
        self.assertEqual(scene["story_beat_outcome"], "Hero gains ground")

    def test_coerce_uses_backend_character_registry_not_writer_registry(self):
        raw = json.dumps({
            "characters": [{"id": "P1", "name": "Tên sai"}],
            "scenes": [{
                "scene_id": "S0001", "start": "00:01", "end": "00:08",
                "tts": "Kukuru nhanh chóng quay về báo tin cho mọi người trong bộ tộc.",
                "covered_event_ids": ["E0001"],
            }],
        }, ensure_ascii=False)
        normalized = json.loads(coerce_single_part_response(
            raw, 1, 10, 20,
            required_event_ids=["E0001"],
            locked_characters=[{"id": "P1", "name": "Kukuru"}],
        ))
        self.assertEqual(normalized["characters"], [{"id": "P1", "name": "Kukuru"}])

    def test_dense_story_events_become_one_narration_with_multiple_visuals(self):
        events, collapsed = coalesce_dense_story_events([[
            {
                "evidence_id": "EV1", "start_seconds": 10, "end_seconds": 11,
                "event": "Quái vật xuất hiện", "narration_focus": "mối đe dọa xuất hiện",
            },
            {
                "evidence_id": "EV2", "start_seconds": 12, "end_seconds": 14,
                "event": "Nobita bỏ chạy", "narration_focus": "Nobita tìm đường thoát",
            },
            {
                "evidence_id": "EV3", "start_seconds": 20, "end_seconds": 23,
                "event": "Nobita gặp Kukuru", "narration_focus": "Kukuru xuất hiện",
            },
        ], [], []])
        self.assertEqual(collapsed, 1)
        self.assertEqual(len(events[0]), 2)
        self.assertEqual(len(events[0][0]["visual_sources"]), 2)
        self.assertEqual(events[0][0]["merged_evidence_ids"], ["EV1", "EV2"])

    def test_character_name_outside_locked_scene_registry_requests_retry(self):
        payload = json.dumps({
            "characters": [
                {"id": "P1", "name": "Kukuru"},
                {"id": "P2", "name": "Gigazombie"},
            ],
            "parts": [{"part": 1, "video_end": "00:20", "scenes": [{
                "scene_id": "S0001", "start": "00:01", "end": "00:20",
                "tts": "Gigazombie quay về bộ tộc để báo cho mọi người biết mối nguy hiểm đang tới.",
                "allowed_character_names": ["Kukuru"],
            }]}],
        }, ensure_ascii=False)
        requirements = single_part_missing_requirements(
            payload, 20, 20, require_tts_word_range=False
        )
        self.assertEqual(requirements["action"], "characters")
        self.assertEqual(
            requirements["invalid_character_mentions"][0]["unexpected_names"],
            ["Gigazombie"],
        )

    def test_event_text_adds_named_targets_to_locked_character_ids(self):
        plan = {
            "boundaries": [0, 30, 60, 90],
            "summaries": ["", "", ""],
            "characters": [
                {"id": "P2", "name": "Doraemon"},
                {"id": "P4", "name": "Gian"},
                {"id": "P5", "name": "Suneo"},
            ],
            "events_by_part": [[{
                "event_id": "E0001", "scene_id": "S0001",
                "story_beat_id": "B001", "start_seconds": 1,
                "end_seconds": 10, "characters": ["P2"],
                "event": "Doraemon dùng bảo bối cứu Gian và Suneo",
            }], [], []],
            "story_beats_by_part": [[], [], []],
        }
        event = build_part_story_context(plan, 1)["required_events"][0]
        self.assertEqual(event["characters"], ["P2", "P4", "P5"])
        self.assertEqual(
            event["allowed_character_names"],
            ["Doraemon", "Gian", "Suneo"],
        )

    def test_visual_slot_expansion_refuses_weak_story_coverage(self):
        events = [{
            "evidence_id": f"EV{index:04d}",
            "start_seconds": float(index * 20),
            "end_seconds": float(index * 20 + 10),
            "event": f"Bằng chứng {index}",
        } for index in range(10)]
        with self.assertRaisesRegex(ValueError, "EVENT_CAPACITY_INSUFFICIENT"):
            expand_evidence_to_visual_slots(events, 142)

    def test_rich_event_is_split_into_distinct_global_sentence_slots(self):
        slots = expand_evidence_to_visual_slots([{
            "evidence_id": "EV0001", "start_seconds": 0, "end_seconds": 18,
            "event": "Doraemon lấy bảo bối",
            "cause": "Cả nhóm gặp nguy hiểm",
            "consequence": "Mọi người thoát nạn",
            "causal_anchor": True, "selection_importance": "critical",
            "visual_sources": [{
                "start_seconds": 0, "end_seconds": 3,
                "event": "Doraemon lấy bảo bối",
                "verified_visual_shot": True,
            }],
        }], 3)
        self.assertEqual(len(slots), 3)
        self.assertEqual(len({slot["narration_focus"] for slot in slots}), 3)
        self.assertTrue(all(
            slot["source_evidence_id"] == "EV0001" for slot in slots
        ))
        self.assertEqual(
            [slot["narration_slot_index"] for slot in slots], [1, 2, 3]
        )
        self.assertLessEqual(
            slots[0]["end_seconds"], slots[1]["start_seconds"]
        )
        self.assertTrue(all(1 <= len(slot["visual_sources"]) <= 3 for slot in slots))
        self.assertTrue(any(
            source.get("visual_match_tier") == "same_event_nearest"
            for slot in slots for source in slot["visual_sources"]
        ))
        self.assertTrue(all(
            0 < source["end_seconds"] - source["start_seconds"] <= 15.0
            for slot in slots for source in slot["visual_sources"]
        ))

    def test_semantic_beat_crossing_sentence_bands_is_partitioned(self):
        event = {
            "evidence_id": "EV_BAND",
            "start_seconds": 0,
            "end_seconds": 18,
            "event": "Nobita chạy qua hang",
            "cause": "Kẻ thù đuổi theo",
            "consequence": "Nobita tìm thấy lối thoát",
            "visual_sources": [{
                "start_seconds": 2,
                "end_seconds": 16,
                "event": "Nobita chạy liên tục qua hang",
                "verified_visual_shot": True,
            }],
        }
        slots = expand_evidence_to_visual_slots([event], 3)
        ranges = [
            (source["start_seconds"], source["end_seconds"])
            for slot in slots for source in slot["visual_sources"]
        ]
        self.assertEqual(ranges, [(2, 6), (6, 12), (12, 16)])
        self.assertTrue(all(
            left[1] <= right[0]
            for left, right in zip(ranges, ranges[1:])
        ))

    def test_missing_optional_visual_shots_never_deletes_timestamped_event(self):
        event = {
            "evidence_id": "EV0001",
            "start_seconds": 10,
            "end_seconds": 30,
            "event": "Nobita chạy qua hang đá",
            "cause": "Cậu bị kẻ thù truy đuổi",
            "change": "Cậu tìm thấy một lối rẽ",
            "consequence": "Cậu tạm thời thoát khỏi nguy hiểm",
            "visual_grounding_missing": True,
            "visual_sources": [],
        }
        slots = expand_evidence_to_visual_slots([event], 3)
        self.assertEqual(len(slots), 3)
        self.assertTrue(all(slot["visual_sources"] for slot in slots))
        self.assertTrue(all(
            source.get("visual_match_tier") in {
                "same_event_window", "same_event_nearest",
            }
            for slot in slots for source in slot["visual_sources"]
        ))

    def test_strict_visual_slots_reject_unverified_event_window(self):
        event = {
            "evidence_id": "EV_STRICT",
            "start_seconds": 10,
            "end_seconds": 30,
            "event": "Nobita chạy qua hang đá",
            "visual_sources": [],
        }
        with self.assertRaisesRegex(ValueError, "VISUAL_GROUNDING_MISSING"):
            expand_evidence_to_visual_slots(
                [event], 1, require_verified_visuals=True,
            )

    def test_clamp_keeps_later_noncontiguous_verified_visual_beat(self):
        payload = {
            "characters": [],
            "parts": [{
                "part": 1,
                "video_start": "00:00.00",
                "video_end": "01:00.00",
                "scenes": [{
                    "scene_id": "S0001",
                    "start": "00:00.00",
                    "end": "00:03.50",
                    "tts": "Nobita chạy khỏi nhà, sau đó cậu gặp Doraemon ở công viên.",
                    "visual_beats": [{
                        "trigger_text": "sau đó cậu",
                        "start": "00:25.00",
                        "end": "00:28.00",
                        "confidence": 1.0,
                        "match_type": "literal",
                    }],
                }],
                "reserve_scenes": [],
            }],
        }
        result = clamp_payload_narrations(payload)
        beats = result["parts"][0]["scenes"][0]["visual_beats"]
        self.assertEqual(len(beats), 1)
        self.assertEqual(beats[0]["start"], "00:25.00")

    def test_clamp_drops_beat_owned_by_next_sentence_time(self):
        payload = {
            "characters": [],
            "parts": [{
                "part": 1,
                "video_start": "00:00.00",
                "video_end": "10:00.00",
                "scenes": [{
                    "scene_id": "S0001",
                    "start": "04:56.00",
                    "end": "04:58.00",
                    "tts": "Nobita chịu trận khi mẹ tiếp tục mắng cậu rất nặng lời.",
                    "visual_beats": [{
                        "trigger_text": "mẹ tiếp tục",
                        "start": "04:59.00",
                        "end": "05:00.00",
                    }],
                    "locked_visual_shots": [
                        {"start": "04:56.00", "end": "04:58.00"},
                        {"start": "04:59.00", "end": "05:00.00"},
                    ],
                }, {
                    "scene_id": "S0002",
                    "start": "04:58.00",
                    "end": "05:01.00",
                    "tts": "Nobita buồn bã vì những lời trách mắng vừa nghe từ mẹ.",
                    "visual_beats": [],
                }],
                "reserve_scenes": [],
            }],
        }
        cleaned = clamp_payload_narrations(payload)
        self.assertEqual(cleaned["parts"][0]["scenes"][0]["visual_beats"], [])
        self.assertEqual(
            len(cleaned["parts"][0]["scenes"][0]["locked_visual_shots"]), 1
        )
        timeline = payload_to_timeline(cleaned)
        self.assertTrue(validate_render_visual_grounding(timeline, 600))

    def test_dense_minor_events_are_all_covered_by_fewer_sentence_slots(self):
        events = [{
            "evidence_id": f"EV{index:04d}",
            "start_seconds": float(index * 4),
            "end_seconds": float(index * 4 + 3),
            "event": f"Sự kiện nhỏ {index}",
        } for index in range(4)]
        slots = expand_evidence_to_visual_slots(events, 2)
        self.assertEqual(len(slots), 2)
        covered = {
            row["evidence_id"]
            for slot in slots for row in slot.get("covered_events") or []
        }
        self.assertEqual(covered, {f"EV{index:04d}" for index in range(4)})

    def test_global_plan_locks_target_sentence_count_before_writing(self):
        chunks = [{"events": [{
            "evidence_id": f"EV{index:04d}",
            "start_seconds": float((index - 1) * 40),
            "end_seconds": float(index * 40),
            "event": f"Nhân vật thực hiện hành động {index}",
            "cause": f"Nguyên nhân riêng {index}",
            "consequence": f"Kết quả riêng {index}",
            "importance": "main",
            "story_role": "ending" if index == 3 else "development",
            "narrative_function": "ending" if index == 3 else "development",
        } for index in range(1, 4)]}]
        plan_text = json.dumps({
            "boundary_1_seconds": 40,
            "boundary_2_seconds": 80,
            "part_weights": [1, 1, 1],
            "selected_event_clusters": [],
            "parts": [
                {"part": 1, "summary": "Mở đầu"},
                {"part": 2, "summary": "Phát triển"},
                {"part": 3, "summary": "Kết thúc"},
            ],
        }, ensure_ascii=False)
        plan = normalize_story_plan_from_evidence(
            plan_text, chunks, 120, 120
        )
        contract = plan["scene_contract"]
        self.assertEqual(contract["requested_scene_count"], 6)
        self.assertEqual(contract["scene_count"], 6)
        self.assertTrue(contract["sentence_tolerance_met"])
        self.assertEqual(contract["covered_source_event_count"], 3)
        self.assertGreaterEqual(
            plan["selection_stats"]["split_source_event_count"], 1
        )

    def test_global_plan_never_shrinks_sentence_target_to_sparse_capacity(self):
        chunks = [{"events": [{
            "evidence_id": f"EV{index:04d}",
            "start_seconds": float((index - 1) * 40),
            "end_seconds": float((index - 1) * 40 + 3),
            "event": f"Sự kiện đã xác minh {index}",
            "importance": "main",
            "story_role": "ending" if index == 3 else "development",
            "narrative_function": "ending" if index == 3 else "development",
            "visual_sources": [{
                "start_seconds": float((index - 1) * 40),
                "end_seconds": float((index - 1) * 40 + 3),
            }],
        } for index in range(1, 4)]}]
        plan_text = json.dumps({
            "boundary_1_seconds": 40,
            "boundary_2_seconds": 80,
            "parts": [{"part": 1}, {"part": 2}, {"part": 3}],
        })
        with self.assertRaisesRegex(ValueError, "EVENT_CAPACITY_INSUFFICIENT"):
            normalize_story_plan_from_evidence(plan_text, chunks, 120, 120)

    def test_terminal_review_changes_only_resolution_and_ending_ids(self):
        chunks = [{"events": [
            {"evidence_id": "EV1", "start_seconds": 10, "end_seconds": 13},
            {"evidence_id": "EV2", "start_seconds": 90, "end_seconds": 93},
            {"evidence_id": "EV3", "start_seconds": 100, "end_seconds": 103},
            {"evidence_id": "EV4", "start_seconds": 106, "end_seconds": 109},
        ]}]
        original = {
            "editorial_anchors": {
                "opening": ["EV1"], "climax": ["EV2"],
                "resolution": ["EV4"], "ending": ["EV3"],
            },
            "parts": [{"part": 1}, {"part": 2}, {"part": 3}],
        }
        reviewed = apply_terminal_anchor_review(
            json.dumps(original),
            json.dumps({
                "recognized": True,
                "resolution_ids": ["EV3"],
                "ending_ids": ["EV4"],
            }),
            chunks,
        )
        patched = json.loads(reviewed)
        self.assertEqual(patched["editorial_anchors"]["opening"], ["EV1"])
        self.assertEqual(patched["editorial_anchors"]["climax"], ["EV2"])
        self.assertEqual(patched["editorial_anchors"]["resolution"], ["EV3"])
        self.assertEqual(patched["editorial_anchors"]["ending"], ["EV4"])

    def test_terminal_fallback_uses_same_last_verified_complete_id(self):
        chunks = [{"events": [
            {
                "evidence_id": "EV2", "start_seconds": 90,
                "end_seconds": 96, "completion_status": "open",
            },
            {
                "evidence_id": "EV3", "start_seconds": 100,
                "end_seconds": 104, "completion_status": "complete",
                "consequence": "Mọi người thoát khỏi nguy hiểm",
            },
        ]}]
        plan = json.dumps({
            "editorial_anchors": {
                "resolution": ["EV3"], "ending": ["EV2"],
            }
        })
        repaired = json.loads(fallback_terminal_editorial_anchors(
            plan, chunks
        ))
        self.assertEqual(repaired["editorial_anchors"]["resolution"], ["EV3"])
        self.assertEqual(repaired["editorial_anchors"]["ending"], ["EV3"])
        self.assertEqual(
            repaired["terminal_anchor_status"], "last_verified_complete"
        )

    def test_terminal_fallback_never_jumps_back_over_later_open_event(self):
        chunks = [{"events": [
            {
                "evidence_id": "EV3", "start_seconds": 100,
                "end_seconds": 104, "completion_status": "complete",
                "consequence": "Một mục tiêu phụ hoàn thành",
            },
            {
                "evidence_id": "EV4", "start_seconds": 110,
                "end_seconds": 116, "completion_status": "open",
                "event": "Cảnh cuối vẫn còn dang dở",
            },
        ]}]
        repaired = json.loads(fallback_terminal_editorial_anchors(
            json.dumps({"editorial_anchors": {}}), chunks
        ))
        self.assertEqual(repaired["editorial_anchors"]["ending"], ["EV4"])
        self.assertEqual(repaired["terminal_anchor_status"], "last_verified_only")

    def test_terminal_review_candidates_include_neighbor_ids_only(self):
        chunks = [{"events": [
            {"evidence_id": f"EV{i}", "start_seconds": i * 10, "end_seconds": i * 10 + 3}
            for i in range(1, 8)
        ]}]
        plan = json.dumps({"editorial_anchors": {
            "resolution": ["EV4"], "ending": ["EV6"],
        }})
        candidates = terminal_anchor_review_candidates(plan, chunks, 1)
        self.assertEqual(
            [row["evidence_id"] for row in candidates],
            ["EV3", "EV4", "EV5", "EV6", "EV7"],
        )

    def test_overlap_dedupe_keeps_one_grounded_event(self):
        chunks = [
            {"events": [{
                "evidence_id": "EV001_0001", "start_seconds": 90,
                "end_seconds": 98, "event": "Robot phá cửa và lao vào phòng",
                "visual_detail": "Cánh cửa vỡ tung",
            }]},
            {"events": [{
                "evidence_id": "EV002_0001", "start_seconds": 90.5,
                "end_seconds": 98.5, "event": "Robot phá cửa rồi lao vào phòng",
                "visual_detail": "Cánh cửa vỡ tung",
            }, {
                "evidence_id": "EV002_0002", "start_seconds": 110,
                "end_seconds": 114, "event": "Cả nhóm bỏ chạy",
            }]},
        ]
        cleaned, removed = deduplicate_overlapping_chunk_evidence(chunks)
        self.assertEqual(removed, 1)
        self.assertEqual(len(cleaned[1]["events"]), 1)

    def test_semantic_dedupe_collapses_paraphrased_fact_and_keeps_shots(self):
        chunks = [{"events": [{
            "evidence_id": "EV001", "start_seconds": 100, "end_seconds": 106,
            "event": "Đại Ưu Sư ra lệnh thuộc hạ bắt Nobita",
            "subject": "Đại Ưu Sư", "action": "ra lệnh bắt",
            "object": "Nobita", "result": "thuộc hạ lên đường",
            "fact_key": "villain_orders_capture_nobita",
            "event_kind": "decision", "characters": ["P1"],
        }, {
            "evidence_id": "EV002", "start_seconds": 180, "end_seconds": 188,
            "event": "Kẻ phản diện sai tay chân truy lùng Nobita",
            "subject": "Đại Ưu Sư", "action": "ra lệnh bắt",
            "object": "Nobita", "result": "thuộc hạ lên đường",
            "fact_key": "villain_orders_capture_nobita",
            "event_kind": "decision", "characters": ["P1"],
        }]}]
        cleaned, removed = deduplicate_semantic_chunk_evidence(chunks)
        self.assertEqual(removed, 1)
        self.assertEqual(len(cleaned[0]["events"]), 1)
        self.assertEqual(len(cleaned[0]["events"][0]["visual_sources"]), 2)

    def test_semantic_dedupe_preserves_distinct_causal_steps(self):
        chunks = [{"events": [{
            "evidence_id": "EV001", "start_seconds": 10, "end_seconds": 16,
            "event": "Doraemon gắn chong chóng cho Kukuru",
            "fact_key": "doraemon_equips_kukuru", "event_kind": "preparation",
            "characters": ["P1", "P2"],
        }, {
            "evidence_id": "EV002", "start_seconds": 20, "end_seconds": 27,
            "event": "Kukuru bay lên thành công",
            "fact_key": "kukuru_flies_successfully", "event_kind": "consequence",
            "characters": ["P2"],
        }]}]
        cleaned, removed = deduplicate_semantic_chunk_evidence(chunks)
        self.assertEqual(removed, 0)
        self.assertEqual(len(cleaned[0]["events"]), 2)

    def test_semantic_dedupe_does_not_merge_distant_repeated_fact(self):
        chunks = [{"events": [{
            "evidence_id": "EV001", "start_seconds": 10, "end_seconds": 16,
            "event": "Nobita bị mẹ mắng", "fact_key": "mother_scolds_nobita",
            "event_kind": "reaction", "characters": ["P1"],
        }, {
            "evidence_id": "EV002", "start_seconds": 400, "end_seconds": 406,
            "event": "Nobita lại bị mẹ mắng", "fact_key": "mother_scolds_nobita",
            "event_kind": "reaction", "characters": ["P1"],
        }]}]
        cleaned, removed = deduplicate_semantic_chunk_evidence(chunks)
        self.assertEqual(removed, 0)
        self.assertEqual(len(cleaned[0]["events"]), 2)

    def test_boundary_stitch_links_but_does_not_delete_event_fragments(self):
        chunks = [{
            "primary_end_seconds": 900,
            "events": [{
                "evidence_id": "EV001", "start_seconds": 880, "end_seconds": 930,
                "characters": ["P1"], "continues_to_next": True,
                "completion_status": "open",
            }],
        }, {
            "events": [{
                "evidence_id": "EV002", "start_seconds": 900, "end_seconds": 960,
                "characters": ["P1"], "continues_from_previous": True,
                "completion_status": "complete",
            }],
        }]
        stitched, count, unresolved = stitch_cross_chunk_event_continuations(chunks)
        self.assertEqual(count, 1)
        self.assertEqual(unresolved, [])
        self.assertEqual(sum(len(chunk["events"]) for chunk in stitched), 2)
        self.assertIn("EV001", stitched[1]["events"][0]["depends_on"])
        self.assertEqual(
            stitched[0]["events"][0]["parent_cluster_id"],
            stitched[1]["events"][0]["parent_cluster_id"],
        )

    def test_scene_budget_compression_preserves_every_source_event(self):
        parts = [[{
            "evidence_id": f"EV{i:03d}", "start_seconds": i * 10,
            "end_seconds": i * 10 + 4, "event": f"Sự kiện {i}",
            "narrative_function": "development", "story_role": "development",
        } for i in range(1, 11)], [], []]
        compressed, collapsed = compress_story_events_to_scene_budget(parts, 90)
        self.assertEqual(sum(len(part) for part in compressed), 6)
        self.assertEqual(collapsed, 4)
        covered = {
            row["evidence_id"]
            for event in compressed[0]
            for row in event.get("covered_events", [])
        }
        self.assertEqual(covered, {f"EV{i:03d}" for i in range(1, 11)})

    def test_scene_contract_extends_visual_range_for_tts_without_overlap(self):
        parts = [[
            {"event_id": "E1", "start_seconds": 10, "end_seconds": 11,
             "part_boundary_end_seconds": 30},
            {"event_id": "E2", "start_seconds": 20, "end_seconds": 21,
             "part_boundary_end_seconds": 30},
        ], [], []]
        contract = lock_story_scene_contract(parts, 30)
        self.assertEqual(contract["target_words"], 30)
        self.assertGreaterEqual(parts[0][0]["end_seconds"] - 10, 5)
        self.assertLessEqual(parts[0][0]["end_seconds"], 20)
        self.assertFalse(parts[0][0]["visual_coverage_risk"])

    def test_terminal_ending_is_not_inferred_from_active_attack(self):
        chunks = [{"events": [{
            "evidence_id": "EV001", "start_seconds": 90, "end_seconds": 99,
            "event": "Hai bên vẫn giao chiến", "event_kind": "attack",
            "story_role": "climax", "consequence": "Đòn đánh làm đối thủ ngã",
        }]}]
        promoted, changed = ensure_grounded_terminal_ending(chunks, 100)
        self.assertFalse(changed)
        report = assess_story_evidence_coverage(promoted, 20, 100)
        self.assertFalse(report["has_terminal_ending"])

    def test_unknown_english_signature_becomes_vietnamese_alias(self):
        raw = json.dumps({
            "characters": [{
                "id": "C1", "name": "",
                "description": "Blue, round, wears a red collar with a bell.",
                "visual_signature": "Blue, round, wears a red collar with a bell.",
                "identity_status": "unknown", "identity_confidence": 0,
            }],
            "events": [{
                "local_event_id": "L001", "start_seconds": 1, "end_seconds": 3,
                "event": "C1 chạy vào hang", "characters": ["C1"],
            }],
        }, ensure_ascii=False)
        normalized = normalize_chunk_evidence(raw, 0, 10, chunk_number=1)
        self.assertEqual(
            normalized["characters"][0]["descriptive_alias"],
            "mèo máy màu xanh",
        )
        self.assertEqual(normalized["events"][0]["event"], "mèo máy màu xanh chạy vào hang")

    def test_final_smoothing_preserves_locked_scene_contract(self):
        payload = {"parts": [{"part": 1, "scenes": [
            {
                "scene_id": "S0001", "event": "Nhóm phát hiện dấu vết",
                "narration_focus": "phát hiện", "tts": "câu cũ",
                "allowed_character_names": [],
            },
            {
                "scene_id": "S0002", "event": "Nhóm lần theo dấu vết",
                "narration_focus": "lần theo", "tts": "câu cũ",
                "allowed_character_names": [],
            },
        ]}]}
        prompt = build_final_narration_smoothing_prompt(payload)
        self.assertIn("scene_id", prompt)
        response = json.dumps({"scenes": [
            {
                "scene_id": "S0001",
                "tts": "Cả nhóm bất ngờ phát hiện dấu vết lạ dẫn sâu vào khu rừng tối phía trước.",
            },
            {
                "scene_id": "S0002",
                "tts": "Từ manh mối ấy, họ thận trọng lần theo con đường để tìm nơi kẻ lạ ẩn náu.",
            },
        ]}, ensure_ascii=False)
        result = apply_final_narration_smoothing(payload, response)
        self.assertEqual(result["parts"][0]["scenes"][0]["scene_id"], "S0001")
        self.assertIn("Từ manh mối ấy", result["parts"][0]["scenes"][1]["tts"])

    def test_final_smoothing_rejects_changed_scene_order(self):
        payload = {"parts": [{"scenes": [
            {"scene_id": "S0001", "tts": "cũ"},
            {"scene_id": "S0002", "tts": "cũ"},
        ]}]}
        response = json.dumps({"scenes": [
            {"scene_id": "S0002", "tts": "Một câu kể có đủ mười lăm từ để kiểm tra hợp đồng cảnh đã khóa chặt."},
            {"scene_id": "S0001", "tts": "Một câu kể khác cũng đủ mười lăm từ nhưng thứ tự cảnh đã bị đổi."},
        ]}, ensure_ascii=False)
        with self.assertRaisesRegex(ValueError, "SCENE_CONTRACT_CHANGED"):
            apply_final_narration_smoothing(payload, response)

if __name__ == "__main__":
    unittest.main()
