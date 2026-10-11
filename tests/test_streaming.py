import threading
import time
import unittest
import numpy as np
from transcriber import Transcriber


class StreamingTests(unittest.TestCase):
    def make(self, gate=None, pending=20):
        class Model:
            def __init__(self):
                self.calls = []
                self.started = threading.Event()
            def generate(self, input, cache, is_final, **kwargs):
                self.calls.append((cache, is_final, input.copy()))
                self.started.set()
                if gate:
                    gate.wait(2)
                cache['count'] = cache.get('count', 0) + 1
                return [{'text': '请解释' if cache['count'] == 1 else '缓存原理'}]
        partial = []
        t = Transcriber(streaming=True, on_partial=partial.append, max_pending=pending)
        t._model, t._loaded = Model(), True
        t.set_vad_params(.012, .6)
        self.addCleanup(t.close)
        return t, partial

    def feed(self, t, callback, speech=12, silence=6):
        for _ in range(speech):
            t.process_chunk(np.full(1600, .1, np.float32), callback)
        for _ in range(silence):
            t.process_chunk(np.zeros(1600, np.float32), callback)

    def wait_idle(self, t):
        deadline = time.monotonic() + 2
        while t._jobs.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(t._jobs.unfinished_tasks, 0)

    def test_partial_before_end_final_once_and_cache_per_utterance(self):
        t, partial = self.make()
        final = []
        self.feed(t, final.append, speech=6, silence=0)
        self.wait_idle(t)
        self.assertTrue(partial)
        self.assertEqual(final, [])
        self.feed(t, final.append, speech=0)
        self.wait_idle(t)
        self.assertEqual(final, ['请解释缓存原理'])
        first_cache = t._model.calls[0][0]
        self.feed(t, final.append)
        self.wait_idle(t)
        self.assertIsNot(first_cache, t._model.calls[-1][0])
        self.assertEqual(sum(call[1] for call in t._model.calls), 2)

    def test_stop_suppresses_inflight_partial_and_final(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        t, partial = self.make(gate)
        final = []
        self.feed(t, final.append, speech=6, silence=0)
        self.assertTrue(t._model.started.wait(1))
        t.reset_vad()
        gate.set()
        self.wait_idle(t)
        self.assertEqual(partial, [])
        self.assertEqual(final, [])

    def test_audio_discontinuity_discards_current_utterance(self):
        t, partial = self.make()
        final = []
        self.feed(t, final.append, speech=6, silence=0)
        self.wait_idle(t)
        self.assertTrue(t.has_pending_speech)
        t.discard_utterance()
        self.feed(t, final.append, speech=6)
        self.wait_idle(t)
        self.assertEqual(final, [])
        self.assertFalse(t.has_pending_speech)

    def test_overflow_invalidates_whole_utterance(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        t, partial = self.make(gate, pending=1)
        final = []
        self.feed(t, final.append, speech=6, silence=0)
        self.assertTrue(t._model.started.wait(1))
        self.feed(t, final.append, speech=24)
        gate.set()
        self.wait_idle(t)
        self.assertEqual(final, [])
        self.feed(t, final.append, speech=6, silence=0)
        self.wait_idle(t)
        self.feed(t, final.append, speech=0)
        self.wait_idle(t)
        self.assertEqual(len(final), 1)


class QuestionTests(unittest.TestCase):
    def make(self):
        from question_gate import QuestionGate
        return QuestionGate()

    def test_acknowledgments_and_statements_skip(self):
        gate = self.make()
        for text in ('好的', '嗯嗯', '明白了', '谢谢', '我们公司在北京'):
            self.assertIsNone(gate.push(text, 1))
        self.assertIsNone(gate.poll(10))

    def test_complete_question_and_interview_imperative(self):
        gate = self.make()
        self.assertEqual(gate.push('请解释一下缓存的原理', 1), '请解释一下缓存的原理')
        self.assertEqual(gate.push('介绍一下你做过的项目', 2), '介绍一下你做过的项目')
        self.assertEqual(gate.push('说一下你的优化思路', 3), '说一下你的优化思路')

    def test_incomplete_question_merges(self):
        gate = self.make()
        self.assertIsNone(gate.push('请解释缓存原理，以及', 1))
        self.assertEqual(gate.push('如何解决缓存击穿', 1.5), '请解释缓存原理，以及如何解决缓存击穿')

    def test_background_prefix_merge_timeout_and_reset(self):
        gate = self.make()
        self.assertIsNone(gate.push('我先说一下背景', 1))
        self.assertEqual(gate.push('如何处理并发请求', 1.5), '我先说一下背景如何处理并发请求')
        self.assertIsNone(gate.push('请解释缓存，以及', 2))
        self.assertIsNone(gate.poll(2.5))
        self.assertEqual(gate.poll(5), '请解释缓存，以及')
        gate.push('请解释缓存，以及', 20)
        gate.reset()
        self.assertIsNone(gate.poll(25))

    def test_duplicates_suppressed_but_later_repetition_allowed(self):
        gate = self.make()
        self.assertIsNotNone(gate.push('什么是缓存？', 1))
        self.assertIsNone(gate.push('什么是缓存', 2))
        self.assertIsNotNone(gate.push('什么是缓存', 20))

    def test_max_buffer_bounded(self):
        gate = self.make()
        gate.push('我先说一下背景' + '背景' * 5000, 1)
        self.assertLessEqual(len(gate.pending), 1200)


class BenchmarkPrivacyTests(unittest.TestCase):
    def test_near_matching_text_is_canonicalized_before_cloud(self):
        from scripts.benchmark_streaming import allowed_prompt, EXPECTED
        self.assertEqual(allowed_prompt(EXPECTED + '附加文字'), EXPECTED)

    def test_unrelated_text_rejected(self):
        from scripts.benchmark_streaming import allowed_prompt
        with self.assertRaises(ValueError):
            allowed_prompt('完全无关的个人信息')
