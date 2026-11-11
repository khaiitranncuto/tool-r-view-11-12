import copy
import json
import unittest

from evidence_granularity import separate_event_granularity
from recap_pipeline import normalize_chunk_evidence
from story_planning import expand_event_plans, validate_cluster_detail


class EvidenceGranularityTests(unittest.TestCase):
    def test_summary_is_context_and_does_not_compete_with_atomic_events(self):
        chunks = [{"events": [
            {"evidence_id": "OVERVIEW", "granularity": "summary", "start_seconds": 0,
             "end_seconds": 1200, "event": "Cả nhóm xây dựng cuộc sống mới"},
            {"evidence_id": "A", "granularity": "atomic", "start_seconds": 10,
             "end_seconds": 20, "event": "Dựng nhà", "depends_on": ["OVERVIEW"]},
        ]}]
        original = copy.deepcopy(chunks)
        cleaned, report = separate_event_granularity(chunks)
        self.assertEqual([row["evidence_id"] for row in cleaned[0]["events"]], ["A"])
        self.assertEqual(cleaned[0]["events"][0]["context_dependencies"], ["OVERVIEW"])
        self.assertEqual(cleaned[0]["events"][0]["depends_on"], [])
        self.assertEqual(report["summary_ids"], ["OVERVIEW"])
        self.assertEqual(cleaned[0]["story_context"][0]["event"], "Cả nhóm xây dựng cuộc sống mới")
        self.assertEqual(chunks, original)

    def test_duration_alone_does_not_remove_a_continuous_action(self):
        for kind in ("atomic", ""):
            chunks = [{"events": [{"evidence_id": "A", "granularity": kind,
                                   "start_seconds": 0, "end_seconds": 900,
                                   "event": "Theo dõi một cuộc chạy trốn liên tục"}]}]
            cleaned, _ = separate_event_granularity(chunks)
            self.assertEqual(len(cleaned[0]["events"]), 1)

    def test_ambiguous_legacy_parent_is_retained_for_review_without_fake_splits(self):
        parent = {"evidence_id": "P", "start_seconds": 0, "end_seconds": 900,
                  "event": "Cả nhóm khám phá", "cause": "Đến vùng đất mới", "consequence": "Xây dựng cuộc sống"}
        chunks = [{"events": [parent,
            {"evidence_id": "A", "start_seconds": 30, "end_seconds": 40, "event": "Dựng nhà"},
            {"evidence_id": "B", "start_seconds": 60, "end_seconds": 70, "event": "Tìm thức ăn"}]}]
        cleaned, report = separate_event_granularity(chunks)
        self.assertEqual(report["review_ids"], ["P"])
        self.assertEqual([r["evidence_id"] for r in cleaned[0]["events"]], ["A", "B"])
        self.assertEqual(cleaned[0]["event_review_candidates"][0]["event"], parent["event"])
        again, _ = separate_event_granularity(cleaned)
        self.assertEqual(cleaned, again)

    def test_video_normalizer_separates_explicit_context_and_uncertain_events(self):
        raw = {"events": [
            {"start": 0, "end": 40, "event": "Tổng quan", "granularity": "summary"},
            {"start": 5, "end": 9, "event": "Dựng nhà", "granularity": "atomic"},
            {"start": 12, "end": 20, "event": "Chưa rõ hành động", "granularity": "uncertain"},
        ]}
        result = normalize_chunk_evidence(json.dumps(raw), 100, 60, chunk_number=1)
        self.assertEqual([row["event"] for row in result["events"]], ["Dựng nhà"])
        self.assertEqual(len(result["story_context"]), 1)
        self.assertEqual(len(result["event_review_candidates"]), 1)

    def test_grouped_writer_cannot_scatter_one_event_across_sentence_order(self):
        cluster = {"cluster_id": "C1", "event_ids": ["A", "B"], "sentence_count": 3}
        output = {"cluster_id": "C1", "event_plans": [
            {"source_event_ids": ["B"], "focuses": ["Kết quả B"]},
            {"source_event_ids": ["A"], "focuses": ["Nguyên nhân A", "Hành động A"]},
        ]}
        detail = expand_event_plans(output, cluster)
        self.assertEqual([row["source_event_ids"] for row in detail["sentences"]], [["A"], ["A"], ["B"]])
        validate_cluster_detail(detail, cluster)
        output["event_plans"].append({"source_event_ids": ["A"], "focuses": ["Lặp A"]})
        # Repeated single-ID groups are unambiguous: the backend collects
        # them under that event, retaining source order and all distinct ideas.
        repaired = expand_event_plans(output, cluster)
        self.assertEqual([row['source_event_ids'] for row in repaired['sentences']],
                         [['A'], ['A'], ['A'], ['B']])
        validate_cluster_detail(repaired, cluster, allow_flexible_count=True)
