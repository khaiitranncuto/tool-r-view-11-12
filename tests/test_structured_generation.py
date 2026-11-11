import copy
import json
import unittest
from unittest.mock import Mock

from google.genai import types
from structured_generation import generate_with_schema_recovery
from character_identity import dialogue_name_mentions
from recap_pipeline import build_part_story_context


class ProviderError(Exception):
    def __init__(self, code, message='INVALID_ARGUMENT'):
        super().__init__(message)
        self.code = code


class StructuredGenerationTests(unittest.TestCase):
    def test_actual_sdk_stream_request_schema_rejection_and_json_recovery(self):
        import httpx
        from google import genai
        bodies = []
        def handler(request):
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                return httpx.Response(400, json={'error': {'code': 400,
                    'status': 'INVALID_ARGUMENT', 'message': 'Request contains an invalid argument.'}})
            data = {'candidates': [{'content': {'parts': [{'text': '{"scenes":[]}'}],
                'role': 'model'}, 'finishReason': 'STOP'}]}
            return httpx.Response(200, headers={'Content-Type': 'text/event-stream'},
                text='data: ' + json.dumps(data) + '\n\n')
        client = genai.Client(api_key='offline-only', http_options=types.HttpOptions(
            client_args={'transport': httpx.MockTransport(handler)}))
        try:
            def generate(config):
                return ''.join(chunk.text or '' for chunk in client.models.generate_content_stream(
                    model='gemini-3.1-flash-lite', contents='offline test', config=config))
            self.assertEqual(generate_with_schema_recovery(generate, self.config()), '{"scenes":[]}')
            self.assertEqual(len(bodies), 2)
            self.assertIn('responseJsonSchema', bodies[0]['generationConfig'])
            fallback = bodies[1]['generationConfig']
            self.assertNotIn('responseJsonSchema', fallback)
            self.assertNotIn('responseSchema', fallback)
            self.assertEqual(fallback['responseMimeType'], 'application/json')
            self.assertEqual(fallback['maxOutputTokens'], 400)
        finally:
            client.close()

    def config(self):
        return types.GenerateContentConfig(response_mime_type='application/json',
            temperature=.2, response_json_schema={'type': 'object'},
            max_output_tokens=400)

    def test_success_does_not_add_a_request(self):
        generate = Mock(return_value='{"scenes":[]}')
        config = self.config()
        self.assertEqual(generate_with_schema_recovery(generate, config), '{"scenes":[]}')
        generate.assert_called_once_with(config)

    def test_schema_400_retries_once_without_schema_and_keeps_options(self):
        generate = Mock(side_effect=[ProviderError(400), '{"scenes":[]}'])
        report = Mock()
        config = self.config()
        generate_with_schema_recovery(generate, config, report)
        fallback = generate.call_args_list[1].args[0]
        self.assertIsNone(fallback.response_json_schema)
        self.assertIsNone(fallback.response_schema)
        self.assertEqual(fallback.response_mime_type, 'application/json')
        self.assertEqual(fallback.max_output_tokens, config.max_output_tokens)
        self.assertEqual(fallback.temperature, config.temperature)
        self.assertIsNotNone(config.response_json_schema)
        report.assert_called_once()

    def test_non_schema_errors_are_not_swallowed_or_retried_here(self):
        for error in [ProviderError(429), ProviderError(503), ProviderError(401),
                      ProviderError(400, 'API_KEY_INVALID'), RuntimeError('network')]:
            generate = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                generate_with_schema_recovery(generate, self.config())
            self.assertEqual(generate.call_count, 1)

    def test_second_failure_propagates_without_loop(self):
        generate = Mock(side_effect=ProviderError(400))
        with self.assertRaises(ProviderError):
            generate_with_schema_recovery(generate, self.config())
        self.assertEqual(generate.call_count, 2)

    def test_no_schema_means_no_schema_retry(self):
        generate = Mock(side_effect=ProviderError(400))
        with self.assertRaises(ProviderError):
            generate_with_schema_recovery(generate, types.GenerateContentConfig())
        self.assertEqual(generate.call_count, 1)

    def test_dialogue_typo_is_mention_not_new_name_or_visible_actor(self):
        original = 'Trước hết chúng ta phải tìm nôpita.'
        self.assertEqual(dialogue_name_mentions(original, ['Nobita', 'Doraemon']), {'Nobita'})
        self.assertEqual(dialogue_name_mentions('giao trong thời gian', ['Gian']), set())
        self.assertEqual(dialogue_name_mentions('nopita', ['Nobita', 'Nopita']), {'Nopita'})
        self.assertEqual(dialogue_name_mentions('nopita', ['Nobita', 'Nopito']), set())

    def test_dialogue_mention_adds_allowed_name_without_visible_identity(self):
        event = dict(event_id='E0007', start_seconds=6, end_seconds=12,
            characters=['P02'], situation_narration=True, spoken_text='phải tìm nôpita',
            observed_facts=[{'text': 'Suneo bàn bạc.'}])
        plan = dict(boundaries=[0, 20, 40, 60], summaries=['', '', ''],
            events_by_part=[[event], [], []], characters=[
                {'id': 'P01', 'name': 'Nobita'}, {'id': 'P02', 'name': 'Suneo'}])
        original = copy.deepcopy(plan)
        row = build_part_story_context(plan, 1)['required_events'][0]
        self.assertIn('Nobita', row['allowed_character_names'])
        self.assertIn('Nobita', row['mentioned_character_names'])
        self.assertNotIn('Nobita', row['visible_character_names'])
        self.assertEqual(plan, original)
