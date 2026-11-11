import copy
import json
import unittest

from story_planning import (run_two_stage_planning, validate_spine, validate_cluster_detail,
                            SpineBudgetError, balance_spine_budget)
from recap_pipeline import normalize_story_plan_from_evidence
from detail_repair import inspect_detail_rows, apply_detail_repairs


class TwoStagePlanningTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"events": [{
            "evidence_id": f"EV{i}", "start_seconds": (i-1)*40, "end_seconds": i*40,
            "event": f"Hành động sự kiện {i}", "cause": f"Nguyên nhân riêng {i}",
            "consequence": f"Kết quả riêng {i}",
        } for i in range(1, 4)]}]
        self.spine = {
            "boundary_1_seconds": 40, "boundary_2_seconds": 80,
            "parts": [{"part": i, "summary": f"Phần {i}"} for i in range(1, 4)],
            "selected_event_clusters": [{"cluster_id": f"C{i}", "summary": f"Cụm {i}",
                "event_ids": [f"EV{i}"], "sentence_count": 2, "importance": "main"} for i in range(1, 4)],
        }
        self.backbone = {"editorial_anchors": {
            "opening": ["EV1"], "inciting_incident": ["EV1"], "protagonist_goal": ["EV1"],
            "climax": ["EV2"], "resolution": ["EV3"], "ending": ["EV3"]},
            "event_importance": {f"EV{i}": "main" for i in range(1, 4)}}
        self.details = {f"C{i}": {"cluster_id": f"C{i}", "sentences": [
            {"source_event_ids": [f"EV{i}"], "focus": f"Nguyên nhân riêng {i}"},
            {"source_event_ids": [f"EV{i}"], "focus": f"Kết quả riêng {i}"},
        ]} for i in range(1, 4)}

    def normalize(self, text):
        return normalize_story_plan_from_evidence(text, self.chunks, 120, 120)

    def run_plan(self, generate, **kwargs):
        # These tests focus on grouping/detail behavior. The separate backbone
        # stage gets a valid fixture; dedicated tests below exercise its calls.
        def with_backbone(prompt, label):
            if "lap ke hoach 3/5:" in label:
                return json.dumps(self.backbone), "model"
            return generate(prompt, label)
        return run_two_stage_planning(self.chunks, 120, 120, "editorial instructions",
                                     with_backbone, json.loads, self.normalize, **kwargs)

    def test_two_stages_merge_with_exact_ai_owned_sentence_foci(self):
        calls = []
        def generate(prompt, label):
            calls.append(label)
            payload = self.spine if "2/5:" in label else self.details[label.split()[-1]]
            return json.dumps(payload), "model"
        text, _ = self.run_plan(generate)
        plan = self.normalize(text)
        events = [event for part in plan["events_by_part"] for event in part]
        self.assertEqual(len(events), 6)
        self.assertEqual([event["narration_focus"] for event in events],
                         [s["focus"] for detail in self.details.values() for s in detail["sentences"]])
        self.assertEqual([event["planning_cluster_id"] for event in events], ["C1", "C1", "C2", "C2", "C3", "C3"])
        self.assertEqual(plan["scene_contract"]["covered_source_event_count"], 3)
        self.assertEqual(len(calls), 4)

    def test_transport_failure_keeps_spine_and_completed_clusters(self):
        checkpoints, calls = [], []
        def failing(prompt, label):
            calls.append(label)
            if label.endswith("C2"):
                raise RuntimeError("503 UNAVAILABLE")
            return json.dumps(self.spine if "2/5:" in label else self.details["C1"]), "model"
        with self.assertRaisesRegex(RuntimeError, "503"):
            self.run_plan(failing, save=checkpoints.append)
        saved = checkpoints[-1]
        self.assertEqual(set(saved["details"]), {"C1"})
        resumed_calls = []
        def resumed(prompt, label):
            resumed_calls.append(label)
            return json.dumps(self.details[label.split()[-1]]), "model"
        self.run_plan(resumed, checkpoint=saved)
        self.assertEqual([label.split()[-1] for label in resumed_calls], ["C2", "C3"])

    def test_bad_cluster_is_retried_without_replanning_other_clusters(self):
        calls, bad_once = [], [True]
        def generate(prompt, label):
            calls.append(label)
            if "2/5:" in label:
                payload = self.spine
            else:
                payload = copy.deepcopy(self.details[label.split()[-1]])
                if label.endswith("C2") and bad_once and bad_once[0]:
                    bad_once[0] = False
                    payload["sentences"] = []
            return json.dumps(payload), "model"
        self.run_plan(generate)
        self.assertEqual(sum("2/5:" in label for label in calls), 1)
        self.assertEqual(sum(label.endswith("C1") for label in calls), 1)
        self.assertEqual(sum(label.endswith("C2") for label in calls), 2)

    def test_spine_cannot_omit_events_or_change_global_count(self):
        bad = copy.deepcopy(self.spine)
        bad["selected_event_clusters"][0]["sentence_count"] = 1
        with self.assertRaisesRegex(ValueError, "SPINE_SENTENCE_TOTAL"):
            validate_spine(bad, self.chunks[0]["events"], 6)
        bad = copy.deepcopy(self.spine)
        bad["selected_event_clusters"].pop()
        with self.assertRaisesRegex(ValueError, "SPINE_EVENTS_MISSING"):
            validate_spine(bad, self.chunks[0]["events"], 6)

    def test_detail_cannot_repeat_focus_or_import_other_cluster_event(self):
        bad = copy.deepcopy(self.details["C1"])
        bad["sentences"][1]["focus"] = bad["sentences"][0]["focus"]
        with self.assertRaisesRegex(ValueError, "REPEATED_FOCUS"):
            validate_cluster_detail(bad, self.spine["selected_event_clusters"][0])
        bad["sentences"][1]["source_event_ids"] = ["EV3"]
        with self.assertRaisesRegex(ValueError, "EVENT_OR_FOCUS"):
            validate_cluster_detail(bad, self.spine["selected_event_clusters"][0])

    def test_budget_failure_reports_actual_counts_and_source_facts(self):
        bad = copy.deepcopy(self.spine)
        bad["selected_event_clusters"][0]["sentence_count"] = 4
        with self.assertRaises(SpineBudgetError) as raised:
            validate_spine(bad, self.chunks[0]["events"], 6)
        row = raised.exception.report["clusters"][0]
        self.assertEqual((row["allocated"], row["capacity"], row["excess"]), (4, 3, 1))
        self.assertEqual(row["event_facts"][0]["event_id"], "EV1")
        self.assertEqual(len(row["event_facts"][0]["facts"]), 3)
        self.assertIn("allocated=4 capacity=3 excess=1", str(raised.exception))

    def test_backend_balances_without_extra_model_call_or_changing_story(self):
        bad = copy.deepcopy(self.spine)
        for cluster, count in zip(bad["selected_event_clusters"], [4, 1, 1]):
            cluster["sentence_count"] = count
        calls, logs = [], []
        def generate(prompt, label):
            calls.append(label)
            if "2/5:" in label:
                payload = bad
            else:
                cid = label.split()[-1]
                index = int(cid[1:])
                count = [3, 2, 1][index-1]
                facts = [f"Nguyên nhân riêng {index}", f"Kết quả riêng {index}", f"Hành động sự kiện {index}"]
                payload = {"cluster_id": cid, "sentences": [
                    {"source_event_ids": [f"EV{index}"], "focus": fact} for fact in facts[:count]
                ]}
            return json.dumps(payload), "model"
        text, _ = self.run_plan(generate, log=logs.append)
        merged = json.loads(text)
        self.assertEqual([c["sentence_count"] for c in merged["selected_event_clusters"]], [3, 2, 1])
        for old, new in zip(bad["selected_event_clusters"], merged["selected_event_clusters"]):
            self.assertEqual({k: v for k, v in old.items() if k != "sentence_count"},
                             {k: v for k, v in new.items() if k != "sentence_count"})
        self.assertEqual(sum("chia cum ke" in label for label in calls), 1)
        self.assertEqual(sum("sua rieng phan bo" in label for label in calls), 0)
        self.assertTrue(any("can bang tong 6/6" in line for line in logs))

    def test_balancing_does_not_invent_capacity_or_reduce_global_target(self):
        original = copy.deepcopy(self.spine)
        with self.assertRaisesRegex(ValueError, "SPINE_CAPACITY_TOTAL"):
            balance_spine_budget(original, self.chunks[0]["events"], 10)
        self.assertEqual(original, self.spine)
        balanced = balance_spine_budget(original, self.chunks[0]["events"], 5)
        self.assertEqual(sum(c["sentence_count"] for c in balanced["selected_event_clusters"]), 5)
        self.assertTrue(all(c["sentence_count"] >= 1 for c in balanced["selected_event_clusters"]))

    def test_old_budget_draft_resumes_locally_without_replanning_spine(self):
        bad = copy.deepcopy(self.spine)
        for cluster, count in zip(bad["selected_event_clusters"], [4, 1, 1]):
            cluster["sentence_count"] = count
        saved = []
        def failing(prompt, label):
            if "5/5:" in label:
                raise RuntimeError("503 UNAVAILABLE")
            return json.dumps(self.spine), "model"
        with self.assertRaisesRegex(RuntimeError, "503"):
            self.run_plan(failing, save=saved.append)
        checkpoint = copy.deepcopy(saved[-1])
        checkpoint.pop("spine")
        checkpoint["spine_draft"] = bad
        labels = []
        def resumed(prompt, label):
            labels.append(label)
            cid = label.split()[-1]
            index = int(cid[1:])
            count = [3, 2, 1][index-1]
            facts = [f"Nguyên nhân riêng {index}", f"Kết quả riêng {index}", f"Hành động sự kiện {index}"]
            payload = {"cluster_id": cid, "sentences": [
                {"source_event_ids": [f"EV{index}"], "focus": fact} for fact in facts[:count]
            ]}
            return json.dumps(payload), "model"
        self.run_plan(resumed, checkpoint=checkpoint)
        self.assertTrue(all("5/5:" in label for label in labels))

    def test_cluster_counts_are_flexible_while_global_total_stays_exact(self):
        labels = []
        def generate(prompt, label):
            labels.append(label)
            if "2/5:" in label:
                return json.dumps(self.spine), "model"
            cid = label.split()[-1]
            index = int(cid[1:])
            facts = [f"Nguyên nhân riêng {index}", f"Kết quả riêng {index}", f"Hành động sự kiện {index}"]
            return json.dumps({"cluster_id": cid, "sentences": [
                {"source_event_ids": [f"EV{index}"], "focus": fact} for fact in facts[:index]
            ]}), "model"
        text, _ = self.run_plan(generate)
        merged = json.loads(text)
        self.assertEqual([c["sentence_count"] for c in merged["selected_event_clusters"]], [1, 2, 3])
        self.assertEqual(len(labels), 4)
        self.assertEqual(sum(len(detail["sentences"]) for detail in merged["cluster_details"].values()), 6)

    def test_balance_preserves_already_written_cluster_counts(self):
        plan = balance_spine_budget(self.spine, self.chunks[0]["events"], 6,
                                    fixed_counts={"C1": 1})
        counts = [c["sentence_count"] for c in plan["selected_event_clusters"]]
        self.assertEqual(counts[0], 1)
        self.assertEqual(sum(counts), 6)

    def test_grouping_is_checkpointed_before_roles_and_budget(self):
        calls, saved = [], []
        def generate(prompt, label):
            calls.append(label)
            if "3/5:" in label:
                self.assertTrue(saved[-1].get("grouping"))
                self.assertIn("CUM DA KHOA", prompt)
                payload = self.backbone
            elif "2/5:" in label:
                self.assertFalse(saved)
                self.assertIn("nguyen nhan-hanh dong-ket qua", prompt)
                payload = self.spine
            else:
                payload = self.details[label.split()[-1]]
            return json.dumps(payload), "model"
        run_two_stage_planning(self.chunks, 120, 120, "editorial instructions",
                               generate, json.loads, self.normalize, save=saved.append)
        self.assertEqual(len(calls), 5)
        self.assertIn("2/5:", calls[0])
        self.assertIn("3/5:", calls[1])
        self.assertTrue(all("5/5:" in label for label in calls[2:]))

    def test_role_transport_failure_resumes_without_regrouping(self):
        saved = []
        def failing(prompt, label):
            if "3/5:" in label:
                raise RuntimeError("Server disconnected")
            return json.dumps(self.spine), "model"
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            run_two_stage_planning(self.chunks, 120, 120, "editorial instructions",
                                   failing, json.loads, self.normalize, save=saved.append)
        labels = []
        def resume(prompt, label):
            labels.append(label)
            if "3/5:" in label:
                return json.dumps(self.backbone), "model"
            return json.dumps(self.spine if "2/5:" in label else self.details[label.split()[-1]]), "model"
        run_two_stage_planning(self.chunks, 120, 120, "editorial instructions",
                               resume, json.loads, self.normalize, checkpoint=saved[-1])
        self.assertFalse(any("2/5:" in label for label in labels))

    def test_open_story_has_no_fabricated_terminal_role(self):
        self.backbone["editorial_anchors"]["resolution"] = []
        self.backbone["editorial_anchors"]["ending"] = []
        def generate(prompt, label):
            return json.dumps(self.spine if "2/5:" in label else self.details[label.split()[-1]]), "model"
        text, _ = self.run_plan(generate)
        plan = normalize_story_plan_from_evidence(
            text, self.chunks, 120, 120, require_editorial_contract=True)
        self.assertEqual(plan["editorial_contract_version"], 2)
        self.assertEqual(plan["editorial_anchors"]["ending"], [])
        self.assertFalse(plan["scene_contract"]["ending_inferred"])
        self.assertFalse(any(e["story_role"] == "ending"
                             for part in plan["events_by_part"] for e in part))

    def test_shared_event_can_contain_climax_and_outcome(self):
        self.backbone["editorial_anchors"]["climax"] = ["EV3"]
        def generate(prompt, label):
            return json.dumps(self.spine if "2/5:" in label else self.details[label.split()[-1]]), "model"
        text, _ = self.run_plan(generate)
        normalize_story_plan_from_evidence(text, self.chunks, 120, 120,
                                           require_editorial_contract=True)

    def test_reversed_roles_repair_without_regrouping(self):
        saved, calls = [], []
        def generate(prompt, label):
            calls.append(label)
            if "2/5:" in label:
                return json.dumps(self.spine), "model"
            if "3/5:" in label:
                payload = copy.deepcopy(self.backbone)
                if sum("3/5:" in item for item in calls) == 1:
                    payload["editorial_anchors"]["resolution"] = ["EV1"]
                return json.dumps(payload), "model"
            return json.dumps(self.details[label.split()[-1]]), "model"
        run_two_stage_planning(self.chunks, 120, 120, "editorial instructions",
                               generate, json.loads, self.normalize, save=saved.append)
        self.assertEqual(sum("2/5:" in item for item in calls), 1)
        self.assertEqual(sum("3/5:" in item for item in calls), 2)

    def prepare_order_error_case(self):
        self.chunks[0]["events"][2]["end_seconds"] = 100
        self.chunks[0]["events"].append({
            "evidence_id": "EV4", "start_seconds": 100, "end_seconds": 120,
            "event": "Đội cứu viện xuất hiện", "cause": "Nhận được tín hiệu cầu cứu",
            "consequence": "Mọi người được giải thoát",
        })
        self.spine["selected_event_clusters"][0]["sentence_count"] = 1
        self.spine["selected_event_clusters"][1]["sentence_count"] = 1
        self.spine["selected_event_clusters"][2].update(event_ids=["EV3", "EV4"], sentence_count=4)
        self.details["C1"]["sentences"] = self.details["C1"]["sentences"][:1]
        self.details["C2"]["sentences"] = self.details["C2"]["sentences"][:1]
        self.bad_detail = {"cluster_id": "C3", "sentences": [
            {"source_event_ids": ["EV4"], "focus": "Nhận được tín hiệu cầu cứu"},
            {"source_event_ids": ["EV3"], "focus": "Nguyên nhân riêng 3"},
            {"source_event_ids": ["EV4"], "focus": "Đội cứu viện xuất hiện"},
            {"source_event_ids": ["EV4"], "focus": "Mọi người được giải thoát"},
        ]}
        self.order_patch = {"cluster_id": "C3", "repairs": [
            {"row_index": 1, "source_event_ids": ["EV3"], "focus": "Nguyên nhân riêng 3"},
            {"row_index": 2, "source_event_ids": ["EV3"], "focus": "Kết quả riêng 3"},
        ]}

    def test_order_error_patches_only_bad_rows_and_preserves_completed_clusters(self):
        self.prepare_order_error_case()
        calls, saved = [], []
        def generate(prompt, label):
            calls.append(label)
            if "2/5:" in label:
                payload = self.spine
            elif "sua rieng" in label:
                self.assertIn('"row_index": 1', prompt)
                self.assertIn('"row_index": 2', prompt)
                self.assertIn("expected_id_order", prompt)
                self.assertIn("read_only", prompt)
                payload = self.order_patch
            else:
                payload = self.bad_detail if label.endswith("C3") else self.details[label.split()[-1]]
            return json.dumps(payload), "model"
        text, _ = self.run_plan(generate, save=saved.append)
        details = json.loads(text)["cluster_details"]
        self.assertEqual(details["C1"], self.details["C1"])
        self.assertEqual(details["C2"], self.details["C2"])
        self.assertEqual(details["C3"]["sentences"][2:], self.bad_detail["sentences"][2:])
        self.assertEqual(sum("cum 3/3 C3" in label for label in calls), 1)
        self.assertEqual(sum("sua rieng" in label for label in calls), 1)
        self.assertEqual(sum(len(value["sentences"]) for value in details.values()), 6)
        self.assertTrue(any("C3" in state.get("detail_drafts", {}) for state in saved))

    def test_order_repair_resumes_draft_after_disconnect_without_rewriting_cluster(self):
        self.prepare_order_error_case()
        saved = []
        def failing(prompt, label):
            if "sua rieng" in label:
                raise RuntimeError("Server disconnected")
            payload = self.spine if "2/5:" in label else (
                self.bad_detail if label.endswith("C3") else self.details[label.split()[-1]])
            return json.dumps(payload), "model"
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            self.run_plan(failing, save=saved.append)
        self.assertEqual(saved[-1]["detail_drafts"]["C3"], self.bad_detail)
        calls = []
        def resumed(prompt, label):
            calls.append(label)
            return json.dumps(self.order_patch), "model"
        self.run_plan(resumed, checkpoint=saved[-1])
        self.assertEqual(len(calls), 1)
        self.assertIn("sua rieng 2 o cum C3", calls[0])

    def test_patch_rejects_unrequested_rows_and_inspector_does_not_reorder_content(self):
        self.prepare_order_error_case()
        original = copy.deepcopy(self.bad_detail)
        issues = inspect_detail_rows(self.bad_detail, self.spine["selected_event_clusters"][2])
        self.assertEqual([row["row_index"] for row in issues], [1, 2])
        self.assertEqual(self.bad_detail, original)
        bad_patch = copy.deepcopy(self.order_patch)
        bad_patch["repairs"][1]["row_index"] = 3
        with self.assertRaisesRegex(ValueError, "DETAIL_PATCH_ROW"):
            apply_detail_repairs(self.bad_detail, bad_patch, [1, 2])
        self.assertEqual(self.bad_detail, original)
