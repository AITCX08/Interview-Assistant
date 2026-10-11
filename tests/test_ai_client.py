import io
import json
import threading
import time
import unittest
from urllib.error import HTTPError, URLError

from ai_client import DeepSeekClient, DeepSeekSettings, AIError, AnswerService, RoutingClient


def event(content):
    return ('data: ' + json.dumps({'choices': [{'delta': {'content': content}}]}, ensure_ascii=False) + '\n\n').encode()


class AITests(unittest.TestCase):
    def settings(self):
        return DeepSeekSettings(api_key='synthetic-test-key')

    def test_stream_is_incremental_non_thinking_and_official_endpoint(self):
        body = b': heartbeat\n\n' + event('先') + event('说明思路。') + b'data: [DONE]\n\n'
        response = io.BytesIO(body)
        requests, received = [], []
        def opener(request, timeout):
            requests.append(request)
            return response
        metrics = DeepSeekClient(opener=opener).stream(self.settings(), '模拟问题', threading.Event(), received.append)
        self.assertEqual(received, ['先', '说明思路。'])
        payload = json.loads(requests[0].data)
        self.assertTrue(payload['stream'])
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertEqual(requests[0].full_url, 'https://api.deepseek.com/chat/completions')
        self.assertNotIn('synthetic-test-key', repr(self.settings()))
        self.assertGreaterEqual(metrics['total_ms'], metrics['first_token_ms'])
        self.assertTrue(response.closed)

    def test_missing_key_rejected_before_network(self):
        def forbidden(*args, **kwargs):
            self.fail('Network called')
        with self.assertRaises(AIError):
            DeepSeekClient(opener=forbidden).stream(DeepSeekSettings(), '问题', threading.Event(), lambda _: None)

    def test_http_error_does_not_leak_key_or_body(self):
        for status in (401, 429, 500):
            def opener(*args, **kwargs):
                raise HTTPError('https://api.deepseek.com', status, 'synthetic-test-key', {}, io.BytesIO(b'secret'))
            with self.assertRaises(AIError) as error:
                DeepSeekClient(opener=opener).stream(self.settings(), '问题', threading.Event(), lambda _: None)
            self.assertNotIn('synthetic-test-key', str(error.exception))
            self.assertNotIn('secret', str(error.exception))

    def test_invalid_json_and_truncated_stream_are_errors(self):
        for body in (b'data: {wrong}\n\n', event('部分回答')):
            with self.assertRaises(AIError):
                DeepSeekClient(opener=lambda *a, **k: io.BytesIO(body)).stream(self.settings(), '问题', threading.Event(), lambda _: None)

    def test_network_timeout_is_sanitized(self):
        def opener(*args, **kwargs):
            raise URLError(TimeoutError('synthetic-test-key'))
        with self.assertRaises(AIError) as error:
            DeepSeekClient(opener=opener).stream(self.settings(), '问题', threading.Event(), lambda _: None)
        self.assertNotIn('synthetic-test-key', str(error.exception))

    def test_cancel_closes_stream(self):
        cancelled = threading.Event()
        response = io.BytesIO(event('一') + event('二') + b'data: [DONE]\n\n')
        received = []
        def callback(text):
            received.append(text)
            cancelled.set()
        with self.assertRaises(AIError):
            DeepSeekClient(opener=lambda *a, **k: response).stream(self.settings(), '问题', cancelled, callback)
        self.assertEqual(received, ['一'])
        self.assertTrue(response.closed)

    def test_multiline_sse_and_empty_choices(self):
        response = io.BytesIO(b'data: {"choices":\ndata: []}\n\n' + event('好') + b'data: [DONE]\n\n')
        received = []
        DeepSeekClient(opener=lambda *a, **k: response).stream(self.settings(), '问题', threading.Event(), received.append)
        self.assertEqual(received, ['好'])

    def test_latest_question_replaces_pending_and_suppresses_old_events(self):
        gate, started = threading.Event(), threading.Event()
        calls, events = [], []
        class Client:
            def stream(self, settings, prompt, cancel, on_delta):
                calls.append(prompt)
                if prompt == '旧问题':
                    started.set()
                    gate.wait(1)
                on_delta(prompt)
                return {'first_token_ms': 1., 'total_ms': 2.}
        service = AnswerService(Client(), events.append)
        self.addCleanup(service.close)
        service.submit(self.settings(), '旧问题')
        self.assertTrue(started.wait(1))
        service.submit(self.settings(), '中间问题')
        latest = service.submit(self.settings(), '最新问题')
        gate.set()
        deadline = time.monotonic() + 1
        while not any(e['kind'] == 'done' and e['request_id'] == latest for e in events) and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(calls, ['旧问题', '最新问题'])
        self.assertEqual([e['text'] for e in events if e['kind'] == 'delta'], ['最新问题'])

    def test_blank_prompt_does_not_replace_active_request(self):
        service = AnswerService(DeepSeekClient(), lambda _: None)
        self.addCleanup(service.close)
        with self.assertRaises(AIError):
            service.submit(self.settings(), '')

    def test_local_simulation_without_key_never_uses_transport(self):
        def forbidden(*args, **kwargs):
            self.fail('Real transport called during local simulation')
        received = []
        settings = DeepSeekSettings(simulation=True)
        metrics = RoutingClient(DeepSeekClient(opener=forbidden)).stream(settings, '模拟问题', threading.Event(), received.append)
        self.assertIn('本地模拟', ''.join(received))
        self.assertGreater(metrics['total_ms'], metrics['first_token_ms'])
