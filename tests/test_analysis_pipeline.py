import copy
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import (
    BATCH_REQUEST_COOLDOWN_SECONDS,
    DETAIL_VIDEO_ANALYSIS_FPS,
    EVIDENCE_CONTRACT_VERSION,
    FULL_VIDEO_ANALYSIS_FPS,
    GEMINI_CAPACITY_RESUME_DELAY_SECONDS,
    GEMINI_GENERATION_MAX_ATTEMPTS,
    GeminiMinuteQuotaPacer,
    GEMINI_REQUEST_TIMEOUT_SECONDS,
    GEMINI_EVENT_LOOKAHEAD_SECONDS,
    GEMINI_PRIMARY_PART_COUNT,
    MAX_TARGETED_VISUAL_REVIEWS,
    STORY_PLAN_MAX_ATTEMPTS,
    STORY_PLAN_FALLBACK_MODEL,
    STORY_PLAN_PRIMARY_MODEL,
    STORY_PLAN_MODEL_SEQUENCE,
    STORY_PLAN_REQUEST_ATTEMPTS,
    TASKS,
    _format_review_time,
    analysis_proxy_fps,
    analyze_video_bg,
    app as flask_app,
    ascii_upload_filename,
    build_model_candidates,
    describe_gemini_models,
    build_full_video_gemini_part,
    calculate_script_budget,
    recap_depth_instruction,
    create_task,
    generate_gemini_text_with_recovery,
    generate_story_plan_with_model_fallback,
    generate_recap_with_stream_checkpoints,
    gemini_retry_delay_seconds,
    estimate_gemini_video_input_tokens,
    is_daily_project_gemini_quota_error,
    is_gemini_capacity_503_error,
    is_gemini_transport_error,
    is_gemini_auth_error,
    is_per_minute_gemini_quota_error,
    is_transient_gemini_error,
    measure_script_narration,
    normalize_existing_video_filenames,
    normalize_simple_gemini_timeline,
    resolve_selected_gemini_model,
    resumable_chapter_evidence,
    resumable_part_responses,
    resumable_story_plan,
    thinking_config_for_effort,
    run_with_gemini_capacity_resume,
    script_with_error,
    stage_gemini_upload,
    split_video_into_three_parts,
    split_video_into_batches,
    split_video_into_overlapping_batches,
    upload_gemini_file_with_recovery,
    validate_gemini_api_key,
    wait_for_gemini_file,
)


class AnalysisPipelineTests(unittest.TestCase):
    def test_invalid_api_input_returns_json_without_starting_work(self):
        client = flask_app.test_client()
        for body in ([], 'text', {'custom_prompt': None}, {'bgm_volume': 'NaN'},
                     {'zoom_factor': []}, {'speed_factor': True}, {'font_size': '2.5'},
                     {'speed_factor': 0}, {'text': []}):
            with self.subTest(body=body):
                response = client.post('/api/analyze-video', json=body)
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.get_json()['success'])
        malformed = client.post('/api/analyze-video', data='{broken', content_type='application/json')
        self.assertEqual(malformed.status_code, 400)
        self.assertFalse(malformed.get_json()['success'])

    def test_terminal_state_is_persisted_even_immediately_after_progress(self):
        import app as application
        task_id = 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee'
        TASKS[task_id] = {'persistent': True, 'status': 'running', 'logs': []}
        try:
            with patch('app.save_task_state') as save:
                application._persist_task_if_needed(task_id)
                self.assertEqual(save.call_count, 1)
                TASKS[task_id]['status'] = 'failed'
                application._persist_task_if_needed(task_id)
                self.assertEqual(save.call_count, 2)
                self.assertEqual(save.call_args.args[1]['status'], 'failed')
        finally:
            TASKS.pop(task_id, None)
            application._task_log_checkpoint_at.pop(task_id, None)

    def test_status_hides_large_internal_caches_from_live_and_disk_tasks(self):
        task_id = 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee'
        state = {'task_kind': 'analysis', 'status': 'success', 'result': 'recap',
                 'source_writing_batches': {'large': 'data'}, 'source_writing_candidates': {},
                 'story_plan_checkpoint': {'large': 'plan'}}
        try:
            for live in (False, True):
                if live:
                    TASKS[task_id] = state
                with self.subTest(live=live), patch('app.load_task_state', return_value=state):
                    response = flask_app.test_client().get('/api/task-status/' + task_id)
                    data = response.get_json()
                    self.assertEqual(data['result'], 'recap')
                    self.assertNotIn('source_writing_batches', data)
                    self.assertNotIn('source_writing_candidates', data)
                    self.assertNotIn('story_plan_checkpoint', data)
                TASKS.pop(task_id, None)
        finally:
            TASKS.pop(task_id, None)

    def test_dead_analysis_process_is_reported_as_interrupted_with_checkpoint_kept(self):
        task_id = 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee'
        state = {'task_kind': 'analysis', 'status': 'running', 'analysis_pid': 123,
                 'story_plan_checkpoint': {'plan': 'saved'}}
        try:
            with patch('app.load_task_state', return_value=state), \
                 patch('render_guard.process_is_alive', return_value=False), patch('app.save_task_state') as save:
                data = flask_app.test_client().get('/api/task-status/' + task_id).get_json()
                self.assertEqual(data['status'], 'failed')
                self.assertEqual(save.call_args.args[1]['story_plan_checkpoint'], {'plan': 'saved'})
        finally:
            TASKS.pop(task_id, None)

    def test_disconnect_detection_uses_exception_types_and_message(self):
        import httpx
        for error in (httpx.RemoteProtocolError('unexpected end'),
                      RuntimeError('peer closed connection without sending complete message body (incomplete chunked read)'),
                      httpx.ReadError('socket closed')):
            with self.subTest(error=error):
                self.assertTrue(is_gemini_transport_error(error))
                self.assertTrue(is_transient_gemini_error(error))
                wrapped = RuntimeError('SDK request failed')
                wrapped.__cause__ = error
                self.assertTrue(is_transient_gemini_error(wrapped))
        self.assertFalse(is_transient_gemini_error(RuntimeError('403 permission denied: connection reset')))
        self.assertFalse(is_transient_gemini_error(RuntimeError('429 quota: peer closed connection')))

    def test_interrupted_stream_restarts_only_request_and_discards_partial_text(self):
        import httpx
        calls = []
        class Models:
            def generate_content_stream(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    yield SimpleNamespace(text='incomplete JSON', usage_metadata=None)
                    raise httpx.RemoteProtocolError('peer closed connection without sending complete message body (incomplete chunked read)')
                yield SimpleNamespace(text='complete JSON', usage_metadata=None)
        pacer = GeminiMinuteQuotaPacer(safe_limit=100)
        result = generate_gemini_text_with_recovery(SimpleNamespace(models=Models()),
            'same-model', ['same-clip', 'same-prompt'], None, 'test', stream=True,
            max_attempts=1, sleep_fn=lambda _: None, quota_pacer=pacer, estimated_input_tokens=10)
        self.assertEqual(result, 'complete JSON')
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertFalse(pacer._inflight)

    def test_transport_retry_is_bounded_and_can_be_owned_by_caller(self):
        import httpx
        for retry, expected_calls in ((True, 2), (False, 1)):
            calls = []
            class Models:
                def generate_content(self, **kwargs):
                    calls.append(kwargs)
                    raise httpx.RemoteProtocolError('peer closed connection')
            with self.subTest(retry=retry), self.assertRaises(httpx.RemoteProtocolError):
                generate_gemini_text_with_recovery(SimpleNamespace(models=Models()), 'model',
                    ['prompt'], None, 'test', max_attempts=1, sleep_fn=lambda _: None,
                    retry_transport_errors=retry)
            self.assertEqual(len(calls), expected_calls)

    def test_index_renders_every_cached_gemini_model(self):
        rows = [
            {"id": "gemini-model-alpha", "selectable": True},
            {"id": "gemini-model-image", "selectable": False},
            {"id": "gemini-model-omega", "selectable": True},
        ]
        with patch("app._load_gemini_model_cache", return_value=rows):
            response = flask_app.test_client().get("/")
        page = response.get_data(as_text=True)
        self.assertIn("gemini-model-alpha", page)
        self.assertIn("gemini-model-image", page)
        self.assertIn("gemini-model-omega", page)
        self.assertNotIn("Gemini 3.5 Flash Lite (Nhẹ, nhanh)", page)

    def test_recap_stream_503_keeps_complete_scene_and_continues(self):
        task_id = "checkpoint-test"
        TASKS[task_id] = {"persistent": False, "logs": []}
        first_scene = json.dumps({
            "scene_id": "S0001", "start": "00:01", "end": "00:04",
            "story_beat_id": "GB001", "tts": "Câu đầu tiên đã hoàn chỉnh và được giữ lại.",
        }, ensure_ascii=False)
        second_scene = json.dumps({
            "scene_id": "S0002", "start": "00:05", "end": "00:08",
            "story_beat_id": "GB001", "tts": "Câu tiếp theo được viết sau khi kết nối phục hồi.",
        }, ensure_ascii=False)
        prompts = []

        class Models:
            def generate_content_stream(self, **kwargs):
                prompts.append(kwargs["contents"][0])
                if len(prompts) == 1:
                    yield SimpleNamespace(text=first_scene)
                    raise RuntimeError("503 UNAVAILABLE high demand")
                yield SimpleNamespace(text=second_scene)

        sleeps = []
        try:
            text, checkpoints = generate_recap_with_stream_checkpoints(
                SimpleNamespace(models=Models()),
                "gemini-test", "base prompt", None, task_id, 100,
                sleep_fn=sleeps.append,
            )
        finally:
            TASKS.pop(task_id, None)
        self.assertEqual(text, second_scene)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(sum(sleeps), 180.0)
        self.assertIn("TIEP TUC SAU STREAM BI GIAN DOAN", prompts[1])
        self.assertIn("Câu đầu tiên", prompts[1])

    def test_partial_script_keeps_content_and_appends_error_as_last_line(self):
        result = script_with_error(
            "[TIME=00:01-00:04] Nội dung đã viết.",
            "Gemini 503\nUNAVAILABLE",
        )
        self.assertTrue(result.startswith("[TIME=00:01-00:04]"))
        self.assertEqual(result.splitlines()[-1], "[LỖI: Gemini 503 UNAVAILABLE]")

    def test_empty_partial_script_still_displays_the_error(self):
        self.assertEqual(
            script_with_error("", "Timestamp sai"),
            "[LỖI: Timestamp sai]",
        )

    def test_analysis_polling_prefers_live_thread_over_stale_persisted_checkpoint(self):
        task_id = "99999999-9999-9999-9999-999999999999"
        TASKS[task_id] = {
            "status": "running", "progress": 42,
            "message": "Gemini dang phan tich", "task_kind": "analysis",
        }
        stale = {
            "status": "failed", "progress": 0,
            "message": "Worker render da dung dot ngot.",
            "task_kind": "analysis",
        }
        try:
            with patch("app.load_task_state", return_value=stale):
                response = flask_app.test_client().get(f"/api/task-status/{task_id}")
            data = response.get_json()
            self.assertEqual(data["status"], "running")
            self.assertEqual(data["progress"], 42)
        finally:
            TASKS.pop(task_id, None)

    def test_error_display_does_not_repeat_error_prefix(self):
        self.assertEqual(
            script_with_error("", "Lỗi: Gemini không trả về JSON object hợp lệ."),
            "[LỖI: Gemini không trả về JSON object hợp lệ.]",
        )

    def test_full_video_pass_uses_tenth_fps_and_low_media_resolution(self):
        from google.genai import types as genai_types

        uploaded = SimpleNamespace(
            uri="https://example.invalid/full-video.mp4",
            mime_type="video/mp4",
        )
        content = build_full_video_gemini_part(genai_types, uploaded)

        self.assertEqual(FULL_VIDEO_ANALYSIS_FPS, 0.1)
        self.assertEqual(content.video_metadata.fps, 0.1)
        self.assertEqual(
            content.media_resolution.level.value,
            "MEDIA_RESOLUTION_LOW",
        )

    def test_primary_visual_analysis_is_three_bounded_low_cost_passes(self):
        self.assertEqual(GEMINI_PRIMARY_PART_COUNT, 3)
        self.assertEqual(DETAIL_VIDEO_ANALYSIS_FPS, 2.0)
        self.assertEqual(GEMINI_EVENT_LOOKAHEAD_SECONDS, 90)
        self.assertEqual(MAX_TARGETED_VISUAL_REVIEWS, 2)
        self.assertGreater(estimate_gemini_video_input_tokens(480, 2),
                           estimate_gemini_video_input_tokens(480, 1))

    def test_non_gap_story_plan_errors_use_two_attempt_limit(self):
        self.assertEqual(STORY_PLAN_MAX_ATTEMPTS, 2)

    def test_story_plan_transport_does_not_multiply_outer_retries(self):
        self.assertEqual(STORY_PLAN_REQUEST_ATTEMPTS, 1)

    def test_budget_has_a_tight_duration_range(self):
        minutes, target, minimum, maximum = calculate_script_budget(30)
        self.assertEqual(minutes, 6)
        self.assertEqual(target, 1620)
        self.assertEqual(minimum, 1572)
        self.assertEqual(maximum, 1668)

    def test_gemini_request_timeout_is_bounded_to_six_minutes(self):
        self.assertEqual(GEMINI_REQUEST_TIMEOUT_SECONDS, 6 * 60)

    def test_auto_budget_is_exactly_one_fifth_without_twenty_minute_cap(self):
        minutes, target, _, _ = calculate_script_budget(104)
        self.assertEqual(minutes, 20.8)
        self.assertEqual(target, 5620)

    def test_recap_depth_is_relative_to_one_fifth(self):
        self.assertIn("CO_DONG", recap_depth_instruction(100, 10))
        self.assertIn("CAN_BANG", recap_depth_instruction(100, 20))
        self.assertIn("MO_RONG", recap_depth_instruction(100, 30))

    def test_manual_short_budget_keeps_decimal_minutes(self):
        minutes, target, _, _ = calculate_script_budget(30, review_minutes=0.5)
        self.assertEqual(minutes, 0.5)
        self.assertEqual(target, 140)

    def test_proxy_fps_decreases_for_long_video(self):
        self.assertEqual(analysis_proxy_fps(10 * 60), 12)
        self.assertEqual(analysis_proxy_fps(30 * 60), 8)
        self.assertEqual(analysis_proxy_fps(60 * 60), 6)
        self.assertEqual(analysis_proxy_fps(61 * 60), 3)

    def test_timeline_normalizes_and_clamps_narration(self):
        words = " ".join(f"tu{index}" for index in range(30))
        normalized = normalize_simple_gemini_timeline(
            f"```\n[TIME=00:01-00:06] {words}\n```", 20
        )
        count, events = measure_script_narration(normalized)
        self.assertEqual(events, 1)
        self.assertEqual(count, 30)
        self.assertEqual(_format_review_time(65.5), "01:05.5")

    def test_existing_video_is_renamed_to_ascii_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            unicode_name = "cảnh quay.mp4"
            with open(os.path.join(temp_dir, unicode_name), "wb") as output:
                output.write(b"unicode")
            with open(os.path.join(temp_dir, "canh_quay.mp4"), "wb") as output:
                output.write(b"existing")
            videos = normalize_existing_video_filenames(temp_dir)
            self.assertEqual(videos, ["canh_quay.mp4", "canh_quay_1.mp4"])
            with open(os.path.join(temp_dir, "canh_quay.mp4"), "rb") as source:
                self.assertEqual(source.read(), b"existing")

    def test_uploaded_filename_is_safe_ascii(self):
        self.assertEqual(
            ascii_upload_filename("Phim tiếng Việt - tập 1.MP4"),
            "Phim_tieng_Viet_tap_1.mp4",
        )
        self.assertEqual(ascii_upload_filename("../../cảnh quay.mov"), "canh_quay.mov")

    def test_video_is_split_into_three_equal_local_clips(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            with open(command[-1], "wb") as output:
                output.write(b"clip")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.subprocess.run", side_effect=fake_run):
                boundaries, outputs = split_video_into_three_parts(
                    "source.mp4", temp_dir, 90, "ffmpeg"
                )

        self.assertEqual(boundaries, [0.0, 30.0, 60.0, 90.0])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            [call[call.index("-ss") + 1] for call in calls],
            ["0.000", "30.000", "60.000"],
        )
        self.assertEqual(
            [call[call.index("-t") + 1] for call in calls],
            ["30.000"] * 3,
        )

    def test_story_part_is_split_into_output_safe_batches(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            with open(command[-1], "wb") as output:
                output.write(b"clip")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.subprocess.run", side_effect=fake_run):
                boundaries, outputs = split_video_into_batches(
                    "part.mp4", temp_dir, 90, 3, "ffmpeg", "part_1_batch"
                )

        self.assertEqual(boundaries, [0.0, 30.0, 60.0, 90.0])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            [call[call.index("-ss") + 1] for call in calls],
            ["0.000", "30.000", "60.000"],
        )

    def test_video_split_accepts_valid_story_aware_boundaries(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            with open(command[-1], "wb") as output:
                output.write(b"clip")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.subprocess.run", side_effect=fake_run):
                boundaries, outputs = split_video_into_batches(
                    "part.mp4", temp_dir, 90, 3, "ffmpeg", "adaptive",
                    boundaries=[0, 25, 65, 90],
                )

        self.assertEqual(boundaries, [0.0, 25.0, 65.0, 90.0])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            [call[call.index("-ss") + 1] for call in calls],
            ["0.000", "25.000", "65.000"],
        )

    def test_evidence_windows_use_right_side_lookahead_only(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            with open(command[-1], "wb") as output:
                output.write(b"clip")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.subprocess.run", side_effect=fake_run):
                windows, outputs = split_video_into_overlapping_batches(
                    "video.mp4", temp_dir, 2000, [0, 900, 1800, 2000],
                    "ffmpeg", "evidence", overlap_seconds=180,
                )
        self.assertEqual(windows, [(0.0, 1080.0), (900.0, 1980.0), (1800.0, 2000.0)])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            [call[call.index("-ss") + 1] for call in calls],
            ["0.000", "900.000", "1800.000"],
        )

    def test_video_upload_returns_ascii_filename(self):
        response = flask_app.test_client().post(
            "/api/upload-file",
            data={
                "type": "video",
                "file": (io.BytesIO(b"small test video"), "video tiếng Việt.mp4"),
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        payload["filename"].encode("ascii")
        uploaded_path = os.path.join(flask_app.root_path, payload["filename"])
        try:
            self.assertTrue(os.path.isfile(uploaded_path))
        finally:
            if os.path.isfile(uploaded_path):
                os.remove(uploaded_path)

    def test_split_endpoint_falls_back_to_timeline_rows(self):
        script = "[TIME=00:01-00:04] Cau mot.\n[TIME=00:05-00:08] Cau hai."
        with patch("app.split_text_into_sentences", side_effect=ValueError("bad token")):
            response = flask_app.test_client().post(
                "/api/split-sentences", json={"text": script}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), script.splitlines())

    def test_api_key_validation_and_quota_classification(self):
        self.assertEqual(validate_gemini_api_key("AIzaSy_valid-Key_123"), "")
        self.assertIn("Unicode", validate_gemini_api_key("AIzaSy_key_tiếng_Việt"))
        self.assertIn("dieu khien", validate_gemini_api_key("AIzaSy123\nnext"))
        daily = "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        minute = "429 RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProjectPerModel-FreeTier retryDelay: 30s"
        self.assertTrue(is_daily_project_gemini_quota_error(daily))
        self.assertFalse(is_per_minute_gemini_quota_error(daily))
        self.assertTrue(is_per_minute_gemini_quota_error(minute))
        self.assertTrue(is_transient_gemini_error("503 UNAVAILABLE"))
        self.assertFalse(is_transient_gemini_error(minute))
        self.assertTrue(is_gemini_auth_error("401 UNAUTHENTICATED API_KEY_INVALID"))
        try:
            raise RuntimeError(daily)
        except RuntimeError as cause:
            try:
                raise ValueError("Gemini khong tra JSON hop le sau 3 lan") from cause
            except ValueError as wrapped:
                self.assertTrue(is_daily_project_gemini_quota_error(wrapped))

    def test_chapter_checkpoint_resumes_only_completed_matching_prefix(self):
        windows = [
            {"start_seconds": 0.0, "end_seconds": 900.0},
            {"start_seconds": 870.0, "end_seconds": 1800.0},
            {"start_seconds": 1770.0, "end_seconds": 2700.0},
        ]
        checkpoint = {
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "completed": 2,
            "total": 3,
            "windows": windows,
            "video_name": "movie.mp4",
            "target_words": 3000,
            "evidence": [
                {"events": [{"evidence_id": "EV001"}]},
                {"events": [{"evidence_id": "EV002"}]},
            ],
        }
        resumed = resumable_chapter_evidence(
            checkpoint, windows, "movie.mp4", 3000,
        )
        self.assertEqual(len(resumed), 2)
        self.assertEqual(resumed[-1]["events"][0]["evidence_id"], "EV002")
        self.assertEqual(
            resumable_chapter_evidence(checkpoint, windows, "other.mp4", 3000),
            [],
        )

    def test_part_checkpoint_resumes_only_identical_locked_plan(self):
        checkpoint = {
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "responses": ["{\"parts\":[]}"],
            "boundaries": [0, 100, 200, 300],
            "target_words": 900,
            "video_name": "movie.mp4",
            "model": "gemini-test",
            "thinking_effort": "low",
            "style": "dramatic",
            "event_ids": ["E1", "E2"],
        }
        resumed = resumable_part_responses(
            checkpoint, [0, 100, 200, 300], 900, "movie.mp4",
            "gemini-test", "low", "dramatic", ["E1", "E2"],
        )
        self.assertEqual(resumed, checkpoint["responses"])
        self.assertEqual(
            resumable_part_responses(
                checkpoint, [0, 100, 200, 300], 900, "movie.mp4",
                "gemini-test", "low", "dramatic", ["E1", "DIFFERENT"],
            ),
            [],
        )

    def test_story_plan_checkpoint_requires_same_evidence_and_settings(self):
        checkpoint = {
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "plan": {
                "boundaries": [0, 100, 200, 300],
                "editorial_contract_version": 1,
                "editorial_anchors": {
                    "opening": ["EV001"], "inciting_incident": ["EV002"],
                    "protagonist_goal": ["EV003"], "climax": ["EV040"],
                    "resolution": ["EV041"], "ending": ["EV042"],
                },
            },
            "video_name": "movie.mp4",
            "target_words": 900,
            "model": "gemini-test",
            "thinking_effort": "low",
            "style": "dramatic",
            "evidence_event_count": 42,
        }
        plan = resumable_story_plan(
            checkpoint, "movie.mp4", 900, "gemini-test", "low", "dramatic", 42,
        )
        self.assertEqual(plan, checkpoint["plan"])
        self.assertIsNone(resumable_story_plan(
            checkpoint, "movie.mp4", 900, "gemini-test", "low", "dramatic", 41,
        ))
        transcript_plan = copy.deepcopy(checkpoint)
        transcript_plan["plan"]["events_by_part"] = [[{
            "evidence_id": "EV001", "source": "transcript_anchor"
        }], [], []]
        self.assertIsNone(resumable_story_plan(
            transcript_plan, "movie.mp4", 900, "gemini-test", "low", "dramatic", 42,
        ))

    def test_check_key_validates_without_generation_quota_probe(self):
        class Models:
            def list(self):
                return []

            def generate_content(self, **kwargs):
                raise AssertionError("check-key must not call GenerateContent")

        with patch("app.genai.Client", return_value=SimpleNamespace(models=Models())):
            response = flask_app.test_client().post("/api/check-key", json={
                "api_key": "AIzaSy_valid-Key_123",
            })
        payload = response.get_json()
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["usable"])
        self.assertEqual(payload["status"], "usable")

    def test_retry_delay_honors_google_retry_info(self):
        error = "429 RESOURCE_EXHAUSTED retryDelay: '58.5s' input_token_count PerMinute"
        self.assertEqual(gemini_retry_delay_seconds(error, 0), 59.5)

    def test_minute_quota_pacer_waits_for_rolling_usage_to_expire(self):
        pacer = GeminiMinuteQuotaPacer(
            safe_limit=220000, window_seconds=60, guard_seconds=2,
        )
        pacer.record(160000, now=100.0)
        self.assertEqual(pacer.required_wait(80000, now=100.0), 62.0)
        self.assertEqual(pacer.required_wait(80000, now=162.0), 0.0)
        self.assertGreater(
            estimate_gemini_video_input_tokens(2100, DETAIL_VIDEO_ANALYSIS_FPS),
            160000,
        )

    def test_quota_reservations_include_inflight_and_replace_estimates_with_actual_usage(self):
        clock = [100.0]
        pacer = GeminiMinuteQuotaPacer(safe_limit=100, clock_fn=lambda: clock[0])
        ticket = pacer.reserve_request('first', 70)
        self.assertGreater(pacer.required_wait(40), 0)
        clock[0] = 200.0
        self.assertGreater(pacer.required_wait(40), 0)  # Slow request is still in flight.
        pacer.complete_request(ticket, 20)
        self.assertEqual(pacer.required_wait(80), 0)
        self.assertGreater(pacer.required_wait(81), 0)
        second = pacer.reserve_request('second', 50)
        pacer.complete_request(second)  # Timeout/absent metadata keeps the estimate.
        self.assertEqual(pacer.required_wait(30), 0)
        self.assertGreater(pacer.required_wait(31), 0)

    def test_generation_releases_quota_reservation_on_success_and_failure(self):
        pacer = GeminiMinuteQuotaPacer(safe_limit=100)
        class Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(text='ok', usage_metadata={'prompt_token_count': 20})
        generate_gemini_text_with_recovery(SimpleNamespace(models=Models()), 'model', ['prompt'],
            None, 'test', quota_pacer=pacer, estimated_input_tokens=40)
        self.assertEqual(pacer._inflight, {})
        self.assertEqual(pacer.required_wait(80), 0)
        class FailedModels:
            def generate_content(self, **kwargs):
                raise RuntimeError('invalid request')
        with self.assertRaisesRegex(RuntimeError, 'invalid request'):
            generate_gemini_text_with_recovery(SimpleNamespace(models=FailedModels()), 'model', ['prompt'],
                None, 'test', quota_pacer=pacer, estimated_input_tokens=40)
        self.assertEqual(pacer._inflight, {})
        self.assertEqual(pacer.required_wait(40), 0)
        self.assertGreater(pacer.required_wait(41), 0)

    def test_minute_quota_retries_even_when_normal_attempt_budget_is_one(self):
        outcomes = [
            RuntimeError(
                "429 RESOURCE_EXHAUSTED input_token_count PerMinute "
                "retryDelay: '2s'"
            ),
            SimpleNamespace(text="ok"),
        ]
        calls = []
        sleeps = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs["model"])
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        result = generate_gemini_text_with_recovery(
            SimpleNamespace(models=Models()),
            "gemini-selected", ["prompt"], None, "test",
            max_attempts=1, sleep_fn=sleeps.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["gemini-selected", "gemini-selected"])
        self.assertEqual(sum(sleeps), 3.0)

    def test_503_uses_extended_overload_backoff(self):
        delays = [
            gemini_retry_delay_seconds("503 UNAVAILABLE high demand", attempt)
            for attempt in range(5)
        ]
        self.assertEqual(delays, [10.0, 20.0, 40.0, 60.0, 90.0])
        self.assertEqual(GEMINI_GENERATION_MAX_ATTEMPTS, 3)

    def test_batch_capacity_resume_retries_the_same_operation(self):
        outcomes = [RuntimeError("503 UNAVAILABLE high demand"), "ok"]
        calls = []
        sleeps = []
        messages = []

        def operation():
            calls.append("same-batch")
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        result = run_with_gemini_capacity_resume(
            operation,
            "batch 2",
            update_progress=messages.append,
            cycles=2,
            resume_delay=12,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["same-batch", "same-batch"])
        self.assertEqual(sum(sleeps), 12)
        self.assertTrue(any("giu ket qua cu" in message for message in messages))
        self.assertEqual(BATCH_REQUEST_COOLDOWN_SECONDS, 15)
        self.assertEqual(GEMINI_CAPACITY_RESUME_DELAY_SECONDS, 120)

    def test_batch_capacity_resume_does_not_retry_non_transient_error(self):
        calls = []

        def operation():
            calls.append("batch")
            raise ValueError("invalid recap JSON")

        with self.assertRaisesRegex(ValueError, "invalid recap JSON"):
            run_with_gemini_capacity_resume(
                operation,
                "batch",
                cycles=2,
                resume_delay=12,
                sleep_fn=lambda _: None,
            )
        self.assertEqual(calls, ["batch"])

    def test_default_generation_retries_503_twice_without_model_change(self):
        calls = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs["model"])
                raise RuntimeError("503 UNAVAILABLE high demand")

        sleeps = []
        with self.assertRaisesRegex(RuntimeError, "503 UNAVAILABLE"):
            generate_gemini_text_with_recovery(
                SimpleNamespace(models=Models()),
                "gemini-selected",
                ["prompt"],
                None,
                "test",
                sleep_fn=sleeps.append,
            )
        self.assertEqual(calls, ["gemini-selected"] * 3)
        self.assertEqual(sum(sleeps), 360.0)
        self.assertTrue(all(delay <= 5.0 for delay in sleeps))

    def test_story_plan_503_steps_down_each_retry_and_waits_one_minute(self):
        outcomes = [
            RuntimeError("503 UNAVAILABLE high demand"),
            RuntimeError("503 UNAVAILABLE high demand"),
            SimpleNamespace(text="plan-ok"),
        ]
        calls = []
        sleeps = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs["model"])
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        result, model = generate_story_plan_with_model_fallback(
            SimpleNamespace(models=Models()),
            ["locked evidence"],
            None,
            "story plan",
            stream=False,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(result, "plan-ok")
        self.assertEqual(STORY_PLAN_PRIMARY_MODEL, "gemini-3.8-flash")
        self.assertEqual(model, "gemini-3.6-flash")
        self.assertEqual(calls, [
            STORY_PLAN_PRIMARY_MODEL,
            STORY_PLAN_FALLBACK_MODEL,
            "gemini-3.6-flash",
        ])
        self.assertEqual(sum(sleeps), 120.0)
        self.assertTrue(all(delay <= 5.0 for delay in sleeps))

    def test_story_plan_exhausts_five_503_retries_and_records_every_model(self):
        calls, sleeps, recorded = [], [], []
        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs["model"])
                self_outer.assertEqual(kwargs["contents"], ["same locked evidence"])
                raise RuntimeError("503 UNAVAILABLE high demand")
        self_outer = self
        with self.assertRaisesRegex(RuntimeError, "503"):
            generate_story_plan_with_model_fallback(
                SimpleNamespace(models=Models()), ["same locked evidence"], None, "plan",
                stream=False, sleep_fn=sleeps.append,
                model_callback=lambda model, attempt: recorded.append((model, attempt)),
            )
        expected = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
        self.assertEqual(calls, expected)
        self.assertEqual(list(STORY_PLAN_MODEL_SEQUENCE), expected)
        self.assertEqual(recorded, [(model, i) for i, model in enumerate(expected)])
        self.assertEqual(sum(sleeps), 300.0)

    def test_story_plan_does_not_change_model_for_non_503_error(self):
        calls = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs["model"])
                raise ValueError("invalid request")

        with self.assertRaisesRegex(ValueError, "invalid request"):
            generate_story_plan_with_model_fallback(
                SimpleNamespace(models=Models()),
                ["locked evidence"],
                None,
                "story plan",
                stream=False,
                sleep_fn=lambda _: None,
            )
        self.assertEqual(calls, [STORY_PLAN_PRIMARY_MODEL])

    def test_planning_disconnect_and_timeout_retry_current_working_model(self):
        outcomes = [RuntimeError("Server disconnected without sending a response."),
                    RuntimeError("read timeout"), SimpleNamespace(text="ok")]
        calls, sleeps = [], []
        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs["model"])
                result = outcomes.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result
        result, model = generate_story_plan_with_model_fallback(
            SimpleNamespace(models=Models()), ["same request"], None, "plan",
            stream=False, sleep_fn=sleeps.append,
            model_sequence=("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"),
        )
        self.assertEqual((result, model), ("ok", "gemini-3.6-flash"))
        self.assertEqual(calls, ["gemini-3.6-flash"] * 3)
        self.assertEqual(sum(sleeps), 120)


    def test_generation_uses_requested_503_delay_schedule(self):
        outcomes = [
            RuntimeError("503 UNAVAILABLE high demand"),
            RuntimeError("503 UNAVAILABLE high demand"),
            RuntimeError("503 UNAVAILABLE high demand"),
            RuntimeError("503 UNAVAILABLE high demand"),
            SimpleNamespace(text="ok"),
        ]
        calls = []
        sleeps = []
        messages = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs["model"])
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        result = generate_gemini_text_with_recovery(
            SimpleNamespace(models=Models()),
            "gemini-selected",
            ["same prompt"],
            None,
            "test",
            update_progress=messages.append,
            sleep_fn=sleeps.append,
            resume_503_forever=True,
            capacity_retry_delay_seconds=60,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["gemini-selected"] * 5)
        self.assertEqual(sum(sleeps), 480.0)
        self.assertGreaterEqual(
            sum("tam gian doan" in message for message in messages), 4
        )

    def test_capacity_503_is_distinct_from_read_timeout(self):
        self.assertTrue(is_gemini_capacity_503_error(
            RuntimeError("503 UNAVAILABLE model high demand")
        ))
        self.assertFalse(is_gemini_capacity_503_error(
            RuntimeError("The read operation timed out")
        ))


    def test_generate_recovery_retries_only_when_allowed(self):
        outcomes = [RuntimeError("503 UNAVAILABLE"), SimpleNamespace(text="ok")]

        class Models:
            def generate_content(self, **kwargs):
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        client = SimpleNamespace(models=Models())
        result = generate_gemini_text_with_recovery(
            client, "model", ["prompt"], None, "test",
            max_attempts=2, sleep_fn=lambda _: None,
        )
        self.assertEqual(result, "ok")

    def test_upload_retries_remote_protocol_disconnect(self):
        outcomes = [
            RuntimeError("RemoteProtocolError: Server disconnected without sending a response"),
            SimpleNamespace(name="files/video"),
        ]

        class Files:
            def upload(self, **kwargs):
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        messages = []
        result = upload_gemini_file_with_recovery(
            SimpleNamespace(files=Files()), "video.mp4",
            update_progress=messages.append,
            max_attempts=2,
            sleep_fn=lambda _: None,
        )
        self.assertEqual(result.name, "files/video")
        self.assertIn("thu lai 2/2", messages[0])

    def test_wait_for_uploaded_file_handles_eventual_active(self):
        states = [RuntimeError("404 not found"), SimpleNamespace(state="PROCESSING"), SimpleNamespace(state="ACTIVE")]

        class Files:
            def get(self, **kwargs):
                value = states.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value

        result = wait_for_gemini_file(
            SimpleNamespace(files=Files()), "file", max_wait_seconds=5,
            poll_interval=1, sleep_fn=lambda _: None,
        )
        self.assertEqual(result.state, "ACTIVE")

    def test_upload_staging_uses_ascii_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "video tiếng Việt.mp4")
            staging = os.path.join(temp_dir, "ascii_stage")
            os.makedirs(staging)
            with open(source, "wb") as output:
                output.write(b"video")
            staged = stage_gemini_upload(source, staging, "source_video")
            staged.encode("ascii")
            self.assertTrue(os.path.isfile(staged))

    def test_model_selection_filters_non_text_models(self):
        models = [
            SimpleNamespace(name="models/gemini-3.5-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-3.1-flash-image", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-robotics", supported_actions=["bidiGenerateContent"]),
        ]
        self.assertEqual(build_model_candidates("gemini-3.5-flash", models), ["gemini-3.5-flash"])
        self.assertEqual(resolve_selected_gemini_model("models/gemini-3.5-flash"), "gemini-3.5-flash")

    def test_all_gemini_models_are_listed_but_specialized_models_are_disabled(self):
        models = [
            SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-3.1-flash-image", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-live", supported_actions=["bidiGenerateContent"]),
        ]
        rows = describe_gemini_models(models)
        self.assertEqual([row["id"] for row in rows], [
            "gemini-3.8-flash", "gemini-3.1-flash-image", "gemini-live",
        ])
        self.assertEqual([row["selectable"] for row in rows], [True, False, False])

    def test_thinking_effort_is_user_selectable(self):
        from google.genai import types as genai_types

        low = thinking_config_for_effort(genai_types, "gemini-3.8-flash", "low")
        high = thinking_config_for_effort(genai_types, "gemini-3.8-flash", "high")
        legacy = thinking_config_for_effort(genai_types, "gemini-2.5-flash", "medium")
        self.assertEqual(low.thinking_level.value, "LOW")
        self.assertEqual(high.thinking_level.value, "HIGH")
        self.assertEqual(legacy.thinking_budget, 4096)

    def test_analysis_rotates_only_on_daily_quota(self):
        task_id = create_task()
        daily = RuntimeError(
            "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        try:
            with patch("app._analyze_video_bg_once", side_effect=[daily, None]) as run:
                analyze_video_bg(
                    task_id, ["key-one", "key-two"], "model", "video.mp4", "style", ""
                )
            self.assertEqual([call.args[1] for call in run.call_args_list], ["key-one", "key-two"])
            self.assertEqual(
                [row["status"] for row in TASKS[task_id]["api_key_states"]],
                ["exhausted", "usable"],
            )
        finally:
            TASKS.pop(task_id, None)

    def test_daily_quota_rotation_preserves_checkpoint_for_next_api(self):
        task_id = create_task()
        daily = RuntimeError(
            "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        calls = []

        def run_once(current_task_id, api_key, *args, **kwargs):
            calls.append(api_key)
            if len(calls) == 1:
                TASKS[current_task_id]["chapter_evidence_checkpoint"] = {
                    "completed": 2, "total": 7,
                    "evidence": [{"events": [{}]}, {"events": [{}]}],
                }
                raise daily
            self.assertEqual(
                TASKS[current_task_id]["chapter_evidence_checkpoint"]["completed"], 2
            )

        try:
            with patch("app._analyze_video_bg_once", side_effect=run_once):
                analyze_video_bg(
                    task_id, ["key-one", "key-two"], "model", "video.mp4", "style", ""
                )
            self.assertEqual(calls, ["key-one", "key-two"])
        finally:
            TASKS.pop(task_id, None)

    def test_analysis_rotates_when_daily_quota_is_wrapped_as_json_error(self):
        task_id = create_task()
        try:
            try:
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED "
                    "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                )
            except RuntimeError as cause:
                try:
                    raise ValueError(
                        "Gemini khong tra JSON hop le cho doan 1 sau 3 lan"
                    ) from cause
                except ValueError as wrapped:
                    quota_error = wrapped
            with patch(
                "app._analyze_video_bg_once", side_effect=[quota_error, None]
            ) as run:
                analyze_video_bg(
                    task_id,
                    ["key-one", "key-two"],
                    "model", "video.mp4", "style", "",
                )
            self.assertEqual(
                [call.args[1] for call in run.call_args_list],
                ["key-one", "key-two"],
            )
        finally:
            TASKS.pop(task_id, None)

    def test_analyze_endpoint_rejects_duplicate_keys(self):
        response = flask_app.test_client().post("/api/analyze-video", json={
            "api_keys": ["AIzaSy_same", "AIzaSy_same"],
            "video_name": "source.mp4",
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("API 2", response.get_json()["message"])


if __name__ == "__main__":
    unittest.main()
