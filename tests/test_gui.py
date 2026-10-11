import os
import io
import json
import threading
import time
import unittest
from unittest.mock import patch

from gui import InterviewAssistantGUI
from ai_client import DeepSeekSettings
from ai_client import DeepSeekClient


@unittest.skipIf(os.name != 'nt' and not os.environ.get('DISPLAY'), 'GUI requires a display')
class GUITests(unittest.TestCase):
    def setUp(self):
        with patch('gui._default_compute_device', return_value='cpu'):
            self.gui = InterviewAssistantGUI()
        self.gui.root.withdraw()
        self.addCleanup(self.gui.root.destroy)

    def test_key_masked_and_auto_opt_in(self):
        self.assertEqual(self.gui._api_key_entry.cget('show'), '*')
        self.assertFalse(self.gui._auto_ai_var.get())
        self.gui._api_key_var.set('synthetic-test-key')
        self.assertEqual(self.gui.get_ai_settings().api_key, 'synthetic-test-key')

    def test_stale_answer_event_is_ignored(self):
        self.gui.begin_ai_request(2)
        self.gui._apply('ai', {'request_id': 1, 'kind': 'delta', 'text': '旧回答'})
        self.gui._apply('ai', {'request_id': 2, 'kind': 'delta', 'text': '新回答'})
        self.assertEqual(self.gui._answer_box.get('1.0', 'end').strip(), '新回答')

    def test_transcript_acceptance_gate_rejects_stopped_session(self):
        self.gui.on_transcript_received = lambda text, generation: generation == 3
        self.gui._apply('append', ('旧文本', 2))
        self.gui._apply('append', ('新文本', 3))
        self.assertEqual(self.gui.get_current_text(), '新文本')

    def test_background_answer_uses_gui_queue(self):
        self.gui.begin_ai_request(1)
        thread = threading.Thread(target=lambda: self.gui.update_ai_event({'request_id': 1, 'kind': 'delta', 'text': '分块'}))
        thread.start()
        thread.join(1)
        self.gui._poll_updates()
        self.assertIn('分块', self.gui._answer_box.get('1.0', 'end'))

    def test_controller_auto_trigger_and_local_simulation(self):
        import main
        with patch('main._detect_device', return_value='cpu'), \
             patch('gui._default_compute_device', return_value='cpu'), \
             patch('main.AudioCapture.list_devices', return_value=[]):
            app = main.InterviewAssistant()
        app._gui.root.withdraw()
        self.addCleanup(app._close)
        app._listening = True
        app._listening_generation = 5
        with patch.object(app, '_start_ai') as start:
            app._accept_transcript('模拟问题', 5)
            start.assert_not_called()
            app._gui._auto_ai_var.set(True)
            app._accept_transcript('模拟问题', 5)
            start.assert_called_once_with('模拟问题')
            self.assertFalse(app._accept_transcript('旧问题', 4))
        app._simulate_ai()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            app._gui.root.update()
            if '请补充自己的真实经历' in app._gui._answer_box.get('1.0', 'end'):
                break
            time.sleep(.01)
        self.assertIn('本地模拟', app._gui._answer_box.get('1.0', 'end'))

    def test_apply_key_does_not_send_network_request(self):
        invoked = []
        self.gui.on_apply_ai = lambda: invoked.append('apply')
        self.gui._api_key_var.set('synthetic-test-key')
        self.gui._on_apply_ai()
        self.assertEqual(invoked, ['apply'])

    def test_metrics_display(self):
        self.gui._apply('asr_metrics', {'queue_ms': 12., 'inference_ms': 80., 'audio_seconds': 2.})
        self.assertIn('80', self.gui._asr_metrics_var.get())

    def test_synthetic_audio_to_automatic_sse_answer(self):
        import main
        import numpy as np
        with patch('main._detect_device', return_value='cpu'), \
             patch('gui._default_compute_device', return_value='cpu'), \
             patch('main._get_hw_info', return_value='模拟硬件'), \
             patch('main.AudioCapture.list_devices', return_value=[]):
            app = main.InterviewAssistant()
        app._gui.root.withdraw()
        self.addCleanup(app._close)
        app._listening = True
        app._listening_generation = 4
        app._gui._auto_ai_var.set(True)
        app._gui._api_key_var.set('synthetic-test-key')
        class Model:
            def generate(self, **kwargs):
                return [{'text': '合成音频问题'}]
        app._transcriber._model = Model()
        app._transcriber._loaded = True
        body = ('data: ' + json.dumps({'choices': [{'delta': {'content': '模拟 AI 回答'}}]}, ensure_ascii=False) + '\n\ndata: [DONE]\n\n').encode()
        requests = []
        def local_opener(request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(body)
        app._ai.client.real_client = DeepSeekClient(opener=local_opener)
        callback = lambda text: app._on_transcript(text, 4)
        for _ in range(4):
            app._transcriber.process_chunk(np.full(1600, .1, dtype=np.float32), callback)
        for _ in range(12):
            app._transcriber.process_chunk(np.zeros(1600, dtype=np.float32), callback)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            app._gui.root.update()
            if '模拟 AI 回答' in app._gui._answer_box.get('1.0', 'end'):
                break
            time.sleep(.01)
        self.assertIn('合成音频问题', app._gui.get_current_text())
        self.assertIn('模拟 AI 回答', app._gui._answer_box.get('1.0', 'end'))
        self.assertEqual(requests[0]['messages'][-1]['content'], '合成音频问题')
        self.assertEqual(len(requests), 1)
