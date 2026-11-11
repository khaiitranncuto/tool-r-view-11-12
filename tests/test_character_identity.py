import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

from character_identity import (VERSION, compact_registry, safe_name_mentions,
    verify_name_evidence, cache_reference_frames, needs_identity_rebuild, render_identity_context)
from situation_recap import character_registry, bind_character_registry
from grounded_pipeline import bind_observations, shot_index, observation_response_schema
from recap_pipeline import (normalize_chunk_evidence, normalize_global_evidence_audit,
    apply_verified_evidence_results, _narration_text_normalizer)


def transcript():
    return [{'start_seconds': 1.0, 'end_seconds': 3.0, 'text': 'Nobita, lại đây!'}]


def evidence(name='Nobita', association='direct_address'):
    return {'kind': 'dialogue', 'quote': f'{name}, lại đây!', 'start_seconds': 1, 'end_seconds': 3,
            'shot_ids': ['SHOT_000001'], 'association': association}


def registry():
    return [{'id': 'P01', 'name': 'Nobita', 'confidence': .99, 'name_evidence': 'lời thoại',
        'neutral_alias': 'cậu bé đeo kính', 'stable_features': ['kính tròn', 'tóc đen'],
        'face_features': ['mặt tròn'], 'visual_signature': 'kính tròn, tóc đen'}]


def payload():
    return {'characters': [{'id': 'C1', 'canonical_id': 'P01', 'name': 'Nobita',
        'alias': 'cậu bé đeo kính', 'match_confidence': .99, 'name_evidence': evidence(),
        'identity_evidence': {'matched_features': ['kính tròn', 'tóc đen'],
            'shot_ids': ['SHOT_000001'], 'reference_seconds': [1.0]},
        'face_features': ['mặt tròn'], 'appearance_variants': ['áo xanh']}],
        'events': [{'event': '[CHAR=C1] mở cửa', 'characters': ['C1'], 'grounded_facts': [
            {'fact_id': 'F1', 'text': '[CHAR=C1] mở cửa', 'actor_ids': ['C1'], 'shot_ids': ['SHOT_000001']}]}]}


class CharacterIdentityTests(unittest.TestCase):
    def test_candidate_name_cannot_leak_through_story_context(self):
        people = [{'id': 'P04', 'name_status': 'candidate', 'proposed_name': 'Gian',
                   'display_name': 'cậu bé to con'}]
        original = 'Gian dùng cỗ máy thời gian. [CHAR=P04] trở về.'
        rendered = render_identity_context(original, people)
        self.assertEqual(rendered, 'cậu bé to con dùng cỗ máy thời gian. cậu bé to con trở về.')
        self.assertEqual(original, 'Gian dùng cỗ máy thời gian. [CHAR=P04] trở về.')

    def test_structured_fact_literal_names_and_codes_are_not_search_replaced(self):
        raw = payload()
        literal = 'Biển ghi Gian, thời gian và mã C1.'
        raw['characters'][0].update(name='Gian', match_confidence=.1)
        raw['events'][0]['event'] = literal
        raw['events'][0]['grounded_facts'][0].update(text=literal, actor_ids=[])
        bound = bind_character_registry(raw, registry(), {'SHOT_000001'}, transcript())
        self.assertEqual(bound['events'][0]['grounded_facts'][0]['text'], literal)
        chunk = normalize_chunk_evidence(json.dumps(dict(bound, events=[dict(bound['events'][0],
            start_seconds=0, end_seconds=6, visual_shots=[])]), ensure_ascii=False), 0, 6, chunk_number=1)
        fixed = apply_verified_evidence_results([chunk], {'canonical_characters': [], 'identity_links': []}, [])
        self.assertEqual(fixed[0]['events'][0]['grounded_facts'][0]['text'], literal)

    def test_candidate_name_is_not_promoted_by_compatibility_path_or_leaked_as_alias(self):
        people = registry()
        people[0]['aliases'] = ['Nobi Nobita']
        strict = character_registry({'characters': people}, transcript_segments=transcript())
        self.assertEqual(strict[0]['name_status'], 'candidate')
        self.assertEqual(strict[0]['aliases'], [])
        self.assertEqual(character_registry({'characters': strict})[0]['name_status'], 'candidate')

    def test_name_spans_never_rewrite_time_space_or_cascade(self):
        text = 'Gian dùng cỗ máy thời gian đi qua không gian. Thời Gian vẫn trôi.'
        result = safe_name_mentions(text, {'Gian': 'cậu bé to con', 'cậu bé to con': 'sai'})
        self.assertEqual(result, 'cậu bé to con dùng cỗ máy thời gian đi qua không gian. Thời Gian vẫn trôi.')
        self.assertEqual(text, 'Gian dùng cỗ máy thời gian đi qua không gian. Thời Gian vẫn trôi.')
        normalize = _narration_text_normalizer([{'id': 'P01', 'name': 'Gian',
            'aliases': ['gian'], 'descriptive_alias': 'cậu bé to con'}])
        self.assertEqual(normalize('Cỗ máy thời gian vượt không gian.'), 'Cỗ máy thời gian vượt không gian.')

    def test_raw_fact_and_summary_survive_both_identity_binding_stages(self):
        raw = payload()
        raw['characters'][0].update(name='Gian', name_evidence='lời thoại', match_confidence=.2)
        original = 'Nhóm bạn đi cỗ máy thời gian trở về. Gian bước ra.'
        raw['events'][0].update(event=original)
        raw['events'][0]['grounded_facts'][0]['text'] = original
        bound = bind_character_registry(raw, registry())
        self.assertEqual(raw['events'][0]['event'], original)
        fact = bound['events'][0]['grounded_facts'][0]
        self.assertEqual(fact['raw_text'], original)
        self.assertIn('cỗ máy thời gian', fact['text'])
        chunk = normalize_chunk_evidence(json.dumps(dict(bound, events=[dict(bound['events'][0],
            start_seconds=0, end_seconds=6, visual_shots=[])]), ensure_ascii=False), 0, 6, chunk_number=1)
        final = apply_verified_evidence_results([chunk], {'canonical_characters': [], 'identity_links': []}, [])
        self.assertIn('thời gian', final[0]['events'][0]['grounded_facts'][0]['text'])
        self.assertEqual(final[0]['events'][0]['grounded_facts'][0]['raw_text'], original)

    def test_exact_quote_time_and_association_confirm_name_without_new_id(self):
        raw = payload()
        bound = bind_character_registry(raw, registry(), {'SHOT_000001'}, transcript())
        c = bound['characters'][0]
        self.assertEqual(c['canonical_id'], 'P01')
        self.assertEqual(c['name_status'], 'confirmed')
        self.assertTrue(c['name_verified'])
        fact = bound['events'][0]['grounded_facts'][0]
        self.assertEqual(fact['raw_text'], '[CHAR=C1] mở cửa')
        self.assertEqual(fact['text'], 'Nobita mở cửa')
        self.assertEqual(fact['canonical_actor_ids'], ['P01'])
        self.assertEqual(c['verified_reference_seconds'], [1.0])

    def test_third_person_and_generic_claim_do_not_confirm_name(self):
        for proof in (evidence(association='third_person'), 'lời thoại',
                      dict(evidence(), quote='Doraemon, lại đây!')):
            raw = payload(); raw['characters'][0]['name_evidence'] = proof
            c = bind_character_registry(raw, registry(), {'SHOT_000001'}, transcript())['characters'][0]
            self.assertEqual(c['canonical_id'], 'P01')
            self.assertEqual(c['identity_match_status'], 'matched')
            self.assertNotEqual(c['name_status'], 'confirmed')
            self.assertEqual(c['display_name'], 'cậu bé đeo kính')

    def test_quote_and_face_from_different_times_cannot_be_joined(self):
        self.assertTrue(verify_name_evidence('Nobita', evidence(), transcript(), {'SHOT_000001'}))
        for proof in (dict(evidence(), start_seconds=30, end_seconds=32),
                      dict(evidence(), shot_ids=['SHOT_000006']),
                      dict(evidence(), quote='Nobita nói điều không có trong transcript')):
            self.assertFalse(verify_name_evidence('Nobita', proof, transcript(), {'SHOT_000001', 'SHOT_000006'}))
        wide = dict(evidence(), start_seconds=0, end_seconds=20, shot_ids=['SHOT_000002'])
        self.assertFalse(verify_name_evidence('Nobita', wide, transcript(), {'SHOT_000002'}))

    def test_rephrasing_one_feature_is_not_two_independent_identity_features(self):
        raw = payload()
        raw['characters'][0]['identity_evidence']['matched_features'] = ['kính tròn', 'đeo kính tròn']
        c = bind_character_registry(raw, registry(), {'SHOT_000001'}, transcript())['characters'][0]
        self.assertEqual(c['identity_match_status'], 'pending')

    def test_weak_view_does_not_erase_confirmed_global_name_or_pollute_features(self):
        first = bind_character_registry(payload(), registry(), {'SHOT_000001'}, transcript())
        saved = character_registry({'characters': registry()}, [{'characters': first['characters']}], transcript())
        weak = payload()
        weak['characters'][0].update(match_confidence=.2, face_features=['dấu hiệu sai'])
        weak['characters'][0]['identity_evidence']['matched_features'] = ['không rõ']
        second = bind_character_registry(weak, saved, {'SHOT_000001'}, transcript())
        self.assertEqual(second['characters'][0]['current_view_status'], 'pending')
        remembered = character_registry({'characters': saved}, [{'characters': second['characters']}], transcript())
        self.assertEqual(remembered[0]['name'], 'Nobita')
        self.assertEqual(remembered[0]['name_status'], 'confirmed')
        self.assertNotIn('dấu hiệu sai', remembered[0]['face_features'])

    def test_clothes_change_keeps_same_identity_and_accumulates_variants(self):
        first = bind_character_registry(payload(), registry(), {'SHOT_000001'}, transcript())
        saved = character_registry({'characters': registry()}, [{'characters': first['characters']}], transcript())
        changed = payload(); changed['characters'][0]['appearance_variants'] = ['áo da thú']
        second = bind_character_registry(changed, saved, {'SHOT_000001'}, transcript())
        c = second['characters'][0]
        self.assertEqual(c['canonical_id'], 'P01')
        self.assertEqual(c['name'], 'Nobita')
        self.assertIn('áo xanh', c['appearance_variants'])
        self.assertIn('áo da thú', c['appearance_variants'])

    def test_identical_visual_candidates_remain_ambiguous(self):
        people = registry() + [dict(registry()[0], id='P02', name='Doraemon')]
        raw = payload(); raw['characters'][0]['name_evidence'] = 'không rõ'
        c = bind_character_registry(raw, people, {'SHOT_000001'}, transcript())['characters'][0]
        self.assertEqual(c['identity_match_status'], 'pending')
        self.assertEqual(c['identity_conflict'], 'ambiguous_visual_match')

    def test_unnamed_person_gets_stable_alias_and_later_name_keeps_id(self):
        raw = payload(); raw['characters'][0].update(canonical_id='', name='', name_evidence={}, alias='cậu bé đeo kính')
        first = bind_character_registry(raw, [], {'SHOT_000001'}, transcript())
        cid = first['characters'][0]['canonical_id']
        saved = character_registry({}, [{'characters': first['characters']}], transcript())
        named = payload(); named['characters'][0]['canonical_id'] = cid
        second = bind_character_registry(named, saved, {'SHOT_000001'}, transcript())
        self.assertEqual(second['characters'][0]['canonical_id'], cid)
        self.assertEqual(second['characters'][0]['name'], 'Nobita')
        self.assertEqual(second['characters'][0]['neutral_alias'], first['characters'][0]['neutral_alias'])

    def test_global_audit_cannot_promote_unverified_candidate_name(self):
        raw = payload(); raw['characters'][0]['name_evidence'] = 'lời thoại'
        bound = bind_character_registry(raw, registry(), {'SHOT_000001'}, transcript())
        chunk = normalize_chunk_evidence(json.dumps(dict(bound, events=[dict(bound['events'][0],
            start_seconds=0, end_seconds=6, visual_shots=[])]), ensure_ascii=False), 0, 6, chunk_number=1)
        audit = normalize_global_evidence_audit(json.dumps({'canonical_characters': [], 'identity_links': []}), [chunk], 6)
        self.assertFalse(any(c.get('name') == 'Nobita' for c in audit['canonical_characters']))

    def test_unknown_actor_token_is_quarantined_without_inventing_person(self):
        raw = {'characters': [{'id': 'C1'}], 'events': [{'event': 'Open door', 'content_kind': 'story',
            'observations': [{'text': '[CHAR=C9] mở cửa', 'actor_ids': ['C9'], 'shot_ids': ['SHOT_000001']}]}]}
        bound = bind_observations(raw, shot_index(6), 0, 6, 1)
        self.assertEqual(bound['events'], [])
        self.assertEqual(bound['_grounding_rejections'][0]['reason'], 'unknown_actor_reference')

    def test_compact_roster_excludes_frames_and_history(self):
        full = dict(registry()[0], name_status='candidate', proposed_name='Nobita',
            reference_frames=[{'local_path': 'private/local.jpg'}], appearances=[{'big': 'history'}])
        compact = compact_registry([full])[0]
        self.assertNotIn('reference_frames', compact)
        self.assertNotIn('appearances', compact)
        self.assertNotIn('local_path', json.dumps(compact))
        self.assertEqual(compact['candidate_name'], 'Nobita')

    def test_old_normalized_checkpoint_requires_local_rebuild_but_current_one_does_not(self):
        old = {'evidence': [{'events': ['cached']}], 'character_registry_version': 1}
        self.assertTrue(needs_identity_rebuild(old))
        modern = {'evidence': [{'character_registry_version': VERSION}], 'character_registry_version': VERSION}
        self.assertFalse(needs_identity_rebuild(modern))

    def test_reference_frames_are_cached_locally_and_never_reencoded_on_reuse(self):
        from tts_engine import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / 'source.mp4')
            subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                'color=c=blue:s=160x90:r=2:d=3', '-c:v', 'libx264', source], capture_output=True, check=True)
            person = {'id': 'P01', 'reference_shot_ids': ['SHOT_000001'], 'verified_reference_seconds': [1.0]}
            first = cache_reference_frames(source, [person], directory, ffmpeg, 3)
            self.assertTrue(Path(first[0]['reference_frames'][0]['local_path']).is_file())
            with patch('subprocess.run', side_effect=AssertionError('must reuse local image')):
                second = cache_reference_frames(source, [person], directory, ffmpeg, 3)
            self.assertEqual(first, second)
            self.assertNotIn('reference_frames', compact_registry(first)[0])

    def test_reference_picture_uses_existing_sampled_clip_grid_and_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'; source.write_bytes(b'source')
            sampled = Path(directory) / 'sampled.mp4'; sampled.write_bytes(b'existing 2fps clip')
            person = {'id': 'P01', 'reference_shot_ids': ['SHOT_000081'],
                      'verified_reference_seconds': [483.0]}
            def encode(command, **kwargs):
                self.assertEqual(command[command.index('-i') + 1], str(sampled))
                self.assertEqual(float(command[command.index('-ss') + 1]), 3.5)
                Path(command[-1]).write_bytes(b'cached reference')
                return subprocess.CompletedProcess(command, 0)
            with patch('subprocess.run', side_effect=encode):
                result = cache_reference_frames(str(source), [person], directory, 'ffmpeg', 1000,
                    sampled_clips=[(str(sampled), 479.6, 989.2)])
            self.assertAlmostEqual(result[0]['reference_frames'][0]['source_seconds'], 483.1)


if __name__ == '__main__':
    unittest.main()
