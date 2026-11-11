import copy
import unittest

from character_identity import profile, merge_profile
from evidence_writer import supported_facts, unsupported_dialogue_fact
from situation_recap import bind_character_registry


class GeminiNamesPolicyTests(unittest.TestCase):
    def test_gemini_name_survives_bad_or_missing_whisper(self):
        row = profile(dict(id='P02', name='Doraemon', confidence=1,
            name_source='gemini', name_evidence='lời thoại',
            neutral_alias='mèo máy màu xanh'), transcript=[], strict=True)
        self.assertEqual(row['display_name'], 'Doraemon')
        self.assertFalse(row['name_verified'])  # not falsely transcript-verified
        self.assertEqual(row['name_source'], 'gemini')

    def test_clip_local_id_collision_resolves_by_gemini_name(self):
        registry = [profile(dict(id=cid, name=name, confidence=1,
            name_source='gemini', neutral_alias=name), strict=True)
            for cid, name in [('P08', 'Dorami'), ('P09', 'Time Patrol Officer')]]
        payload = {'characters': [dict(id='P09', name='Dorami', match_confidence=.99,
            identity_evidence={'shot_ids': ['SHOT_000001']})], 'events': []}
        result = bind_character_registry(payload, registry, {'SHOT_000001'}, [])
        self.assertEqual(result['characters'][0]['canonical_id'], 'P08')
        self.assertEqual(result['characters'][0]['display_name'], 'Dorami')

    def test_conflicting_name_does_not_pollute_existing_person(self):
        base = profile(dict(id='P01', name='Nobita', name_source='gemini', confidence=1), strict=True)
        before = copy.deepcopy(base)
        result = merge_profile(base, {'identity_conflict': 'conflicting_name',
            'name_candidates': [{'name': "Nobita's Mom"}]})
        self.assertEqual(result, before)

    def test_e0042_speech_content_excluded_before_writing(self):
        visual = dict(fact_id='F1', text='Nhóm bạn ăn hamburger và khoai tây chiên',
                      shot_ids=['SHOT_000502'], evidence_kind='visual')
        speech = dict(fact_id='F2', text='Doraemon giải thích về lịch sử di cư của con người',
                      shot_ids=['SHOT_000506'], evidence_kind='visual', spoken_text='')
        self.assertEqual(supported_facts({'observed_facts': [visual, speech]}), [visual])
        self.assertEqual(speech['evidence_kind'], 'visual')  # original is immutable
        self.assertFalse(unsupported_dialogue_fact(dict(speech, spoken_text='Con người di cư.')))
