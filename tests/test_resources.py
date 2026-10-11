import threading
import types
import unittest
from unittest.mock import patch

from audio_capture import AudioCapture
from transcriber import Transcriber
from ai_client import DeepSeekClient, DeepSeekSettings, AIError


class ResourceTests(unittest.TestCase):
    def test_audio_close_even_when_stop_raises(self):
        closed = []
        class Stream:
            def stop(self):
                raise RuntimeError('device removed')
            def close(self):
                closed.append(True)
        capture = AudioCapture()
        capture._stream = Stream()
        capture.stop()
        self.assertEqual(closed, [True])
        self.assertFalse(capture.is_capturing)

    def test_cpu_model_loading_does_not_query_gpu(self):
        options, statuses = [], []
        module = types.ModuleType('funasr')
        module.AutoModel = lambda **kwargs: options.append(kwargs) or object()
        transcriber = Transcriber(device='cpu')
        self.addCleanup(transcriber.close)
        with patch.dict('sys.modules', {'funasr': module}), \
             patch('transcriber._gpu_info', side_effect=AssertionError('CPU mode queried GPU')):
            transcriber.load_model(statuses.append)
        self.assertEqual(options[0]['device'], 'cpu')
        self.assertIn('CPU', statuses[-1])

    def test_network_stream_cannot_use_simulation_settings(self):
        with self.assertRaises(AIError):
            DeepSeekClient(opener=lambda *a, **k: self.fail('Network called')).stream(
                DeepSeekSettings(simulation=True), '问题', threading.Event(), lambda _: None)
