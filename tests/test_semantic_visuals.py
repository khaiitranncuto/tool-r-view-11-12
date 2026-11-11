import copy
import json
import unittest

from semantic_visuals import semantic_visual_plan
from recap_pipeline import (coerce_single_part_response, single_part_missing_requirements,
                            build_targeted_narration_repair_prompt, apply_targeted_narration_repairs,
                            payload_to_timeline, clamp_payload_narrations)
from tts_engine import parse_sentences_and_visual_beats, expand_visual_beats
from event_storyboard import validate_render_visual_grounding
from video_renderer import build_shot_spans


class SemanticVisualTests(unittest.TestCase):
    def test_link_only_repairs_preserve_text(self):
        from grounded_pipeline import preserve_structure_only_narrations
        previous = {'parts': [{'scenes': [{'scene_id': 'S1', 'tts': 'keep'},
                                         {'scene_id': 'S2', 'tts': 'wrong'}]}]}
        candidate = {'parts': [{'scenes': [{'scene_id': 'S1', 'tts': 'unwanted rewrite'},
                                          {'scene_id': 'S2', 'tts': 'fixed'}]}]}
        requirements = {'invalid_timestamp_scenes': [
            {'scene_id': 'S1', 'reasons': ['invalid_spoken_trigger:abc']},
            {'scene_id': 'S2', 'reasons': ['unsupported_narration:wrong action']}]}
        result = preserve_structure_only_narrations(previous, candidate, requirements)
        self.assertEqual([s['tts'] for s in result['parts'][0]['scenes']], ['keep', 'fixed'])

    def test_opening_trigger_is_backend_owned_without_dropping_shot(self):
        plan = semantic_visual_plan('The boy opens the door.',
            [{'start': '00:10', 'end': '00:15', 'shot_id': 'SHOT_000002'}],
            [{'visual_id': 'V001', 'trigger_text': 'wrong opening trigger'}])
        self.assertEqual(plan['visual_alignment_errors'], [])
        self.assertEqual(plan['visual_links'][0]['trigger_text'], '')
        self.assertEqual(plan['locked_visual_shots'][0]['shot_id'], 'SHOT_000002')

    def test_failed_repair_preserves_verified_scene_but_not_rejected_scene(self):
        from grounded_pipeline import preserve_verified_scenes
        old = {'parts': [{'scenes': [
            {'scene_id': 'S1', 'tts': 'verified', 'grounding_audit_passed': True},
            {'scene_id': 'S2', 'tts': 'wrong', 'grounding_audit_passed': False}]}]}
        new = {'parts': [{'scenes': [
            {'scene_id': 'S1', 'tts': 'new error', 'grounding_audit_passed': False},
            {'scene_id': 'S2', 'tts': 'fixed', 'grounding_audit_passed': True}]}]}
        kept = preserve_verified_scenes(old, new)['parts'][0]['scenes']
        self.assertEqual([s['tts'] for s in kept], ['verified', 'fixed'])
        self.assertEqual(new['parts'][0]['scenes'][0]['tts'], 'new error')

    def setUp(self):
        self.text = "Moana bị mắc chân vào san hô, rồi cô dùng đá đập vỡ chúng và bơi lên mặt nước."
        self.sources = [
            {"start": "00:10", "end": "00:15", "event": "Mắc chân"},
            {"start": "00:20", "end": "00:26", "event": "Dùng đá"},
            {"start": "00:30", "end": "00:35", "event": "Bơi lên"},
        ]
        self.links = [{"visual_id": "V001", "trigger_text": ""},
                      {"visual_id": "V002", "trigger_text": "rồi cô dùng đá"},
                      {"visual_id": "V003", "trigger_text": "và bơi lên"}]

    def test_action_links_survive_to_real_tts_boundaries(self):
        event = {"event_id": "E0001", "scene_id": "S0001", "visual_sources": self.sources}
        raw = json.dumps({"scenes": [{"event_id": "E0001", "tts": self.text,
                                      "visual_links": self.links}]})
        payload = json.loads(coerce_single_part_response(raw, 1, 60, 25,
                            locked_event_by_id={"E0001": event}))
        payload = clamp_payload_narrations(payload)
        timeline = payload_to_timeline(payload)
        self.assertTrue(validate_render_visual_grounding(timeline, 60))
        sentences, _, _, groups = parse_sentences_and_visual_beats(timeline.splitlines())
        tokens = self.text.split()
        cues = [[{"start": i * 0.25, "end": (i+1)*0.25, "content": word}
                 for i, word in enumerate(tokens)]]
        texts, durations, starts, ends = expand_visual_beats(sentences, [7.0], groups, cues)
        self.assertEqual(starts, [10, 20, 30])
        self.assertEqual(ends, [15, 26, 35])
        self.assertEqual(" ".join(texts), self.text)
        self.assertAlmostEqual(sum(durations), 7.0)
        self.assertAlmostEqual(durations[0], tokens.index("rồi") * 0.25 - 0.10)

    def test_missing_links_repair_only_defective_sentence(self):
        locked = {"E0001": {"event_id": "E0001", "scene_id": "S0001",
                            "visual_sources": self.sources}}
        raw = json.dumps({"scenes": [{"event_id": "E0001", "tts": self.text}]})
        payload = json.loads(coerce_single_part_response(raw, 1, 60, 25, locked_event_by_id=locked))
        requirements = single_part_missing_requirements(json.dumps(payload), 60, 25)
        self.assertEqual(requirements["invalid_timestamp_scenes"][0]["scene_id"], "S0001")
        prompt = build_targeted_narration_repair_prompt(payload, requirements, locked)
        self.assertIn("V003", prompt)
        repaired = apply_targeted_narration_repairs(payload, json.dumps({"scenes": [
            {"event_id": "E0001", "tts": self.text, "visual_links": self.links}
        ]}), ["E0001"], locked)
        repaired = json.loads(coerce_single_part_response(json.dumps(repaired), 1, 60, 25,
                                                        locked_event_by_id=locked))
        self.assertEqual(repaired["parts"][0]["scenes"][0]["visual_alignment_errors"], [])

    def test_unknown_or_backwards_source_is_not_invented(self):
        links = copy.deepcopy(self.links)
        links[1]["visual_id"] = "V999"
        self.assertTrue(semantic_visual_plan(self.text, self.sources, links)["visual_alignment_errors"])
        links = [self.links[2] | {"trigger_text": ""}, self.links[1]]
        self.assertTrue(semantic_visual_plan(self.text, self.sources, links)["visual_alignment_errors"])

    def test_primary_can_skip_unmentioned_early_source(self):
        plan = semantic_visual_plan("Cô dùng đá để giải thoát chân mình.", self.sources,
                                    [{"visual_id": "V002", "trigger_text": ""}])
        self.assertEqual(plan["start"], "00:00:20.00")
        self.assertEqual(plan["visual_alignment_errors"], [])

    def test_four_actions_not_capped_at_three_shots(self):
        sources = self.sources + [{"start": "00:40", "end": "00:45", "event": "Lên bờ"}]
        links = self.links + [{"visual_id": "V004", "trigger_text": "rồi lên bờ"}]
        text = self.text.rstrip(".") + " rồi lên bờ an toàn."
        plan = semantic_visual_plan(text, sources, links)
        self.assertEqual(len(plan["locked_visual_shots"]), 4)
        self.assertEqual(plan["visual_alignment_errors"], [])

    def test_renderer_plays_action_sequentially_without_uniform_sampling(self):
        edit = {"source_start": 10, "source_duration": 8, "visual_window_end": 25}
        scenes = [{"start": 10, "end": 11}, {"start": 11, "end": 17},
                  {"start": 17, "end": 25}]
        self.assertEqual(build_shot_spans(edit, scenes, preserve_action_order=True),
                         [(10, 1), (11, 6), (17, 1)])


if __name__ == "__main__":
    unittest.main()
