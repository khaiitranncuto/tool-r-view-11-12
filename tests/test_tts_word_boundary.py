import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import edge_tts
from tts_engine import generate_tts_chunk_with_timing


class WordTimingTests(unittest.TestCase):
    def test_request_uses_supported_word_timing_and_keeps_individual_cues(self):
        requested = []

        class OfflineCommunicate:
            def __init__(self, text, voice, **kwargs):
                # Construct the installed client config, without connecting.
                client = real_communicate(text, voice, **kwargs)
                requested.append(client.tts_config.boundary)

            async def stream(self):
                yield {'type': 'audio', 'data': b'x' * 10000}
                for i, word in enumerate(['Open', 'door']):
                    yield {'type': 'WordBoundary', 'text': word,
                           'offset': i * 5000000, 'duration': 4000000}

        real_communicate = edge_tts.Communicate
        with tempfile.TemporaryDirectory() as directory:
            with patch('edge_tts.Communicate', OfflineCommunicate):
                result = asyncio.run(generate_tts_chunk_with_timing(
                    'Open door.', 'vi-VN-HoaiMyNeural', '+20%',
                    str(Path(directory) / 'audio.mp3'), max_retries=1))
        self.assertEqual(requested, ['WordBoundary'])
        self.assertEqual([cue.content for cue in result.cues], ['Open', 'door'])


if __name__ == '__main__':
    unittest.main()
