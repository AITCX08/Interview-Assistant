import io
import json
import threading
import unittest
from ai_client import AIError, DeepSeekClient, DeepSeekSettings


class MessageTransportTests(unittest.TestCase):
    def test_custom_messages_arrive_unchanged_and_stream_metrics_return(self):
        seen = []
        body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        def opener(request, timeout):
            seen.append(json.loads(request.data))
            return io.BytesIO(body)
        client = DeepSeekClient(opener=opener)
        messages = [{'role': 'system', 'content': 'synthetic instruction'},
                    {'role': 'user', 'content': 'synthetic document'}]
        output = []
        result = client.stream_messages(DeepSeekSettings(api_key='synthetic'), messages,
                                       threading.Event(), output.append)
        self.assertEqual(seen[0]['messages'], messages)
        self.assertEqual(output, ['ok'])
        self.assertGreaterEqual(result['total_ms'], result['first_token_ms'])

    def test_invalid_or_unbounded_messages_never_open_network(self):
        client = DeepSeekClient(opener=lambda *a, **k: self.fail('unexpected network'))
        for messages in ([], [{'role': 'tool', 'content': 'x'}],
                         [{'role': 'user', 'content': 'x' * 40001}],
                         [{'role': 'user', 'content': 'x'}] * 41,
                         [{'role': 'user', 'content': None}],
                         [{'role': 'user', 'content': 'x', 'name': 'extra'}]):
            with self.subTest(messages_type=len(messages)), self.assertRaises(AIError):
                client.stream_messages(DeepSeekSettings(api_key='synthetic'), messages,
                                       threading.Event(), lambda _: None)

    def test_cancelled_request_does_not_open_network(self):
        cancel = threading.Event()
        cancel.set()
        client = DeepSeekClient(opener=lambda *a, **k: self.fail('unexpected network'))
        with self.assertRaises(AIError):
            client.stream_messages(DeepSeekSettings(api_key='synthetic'),
                                  [{'role': 'user', 'content': 'x'}], cancel, lambda _: None)
