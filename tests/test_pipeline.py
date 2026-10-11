import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

from audio_capture import AudioCapture
from transcriber import Transcriber


class FakeModel:
    def __init__(self, gate=None):
        self.gate = gate
        self.started = threading.Event()
        self.inputs = []

    def generate(self, input, **kwargs):
        self.inputs.append(input.copy())
        self.started.set()
        if self.gate:
            self.gate.wait(2)
        return [{'text': str(int(input[0]))}]


class PipelineTests(unittest.TestCase):
    def make_transcriber(self, model=None):
        transcriber = Transcriber()
        transcriber._model = model or FakeModel()
        transcriber._loaded = True
        self.addCleanup(lambda: getattr(transcriber, 'close', lambda: None)())
        return transcriber

    def test_buffer_keeps_numpy_blocks(self):
        transcriber = self.make_transcriber()
        transcriber.process_chunk(np.full(1600, .1, dtype=np.float32), lambda _: None)
        self.assertIsInstance(transcriber._buffer[0], np.ndarray)

    def test_exact_silence_boundary(self):
        transcriber = self.make_transcriber()
        done = threading.Event()
        for _ in range(4):
            transcriber.process_chunk(np.full(1600, .1, dtype=np.float32), lambda _: done.set())
        for _ in range(11):
            transcriber.process_chunk(np.zeros(1600, dtype=np.float32), lambda _: done.set())
        self.assertEqual(len(transcriber._model.inputs), 0)
        transcriber.process_chunk(np.zeros(1600, dtype=np.float32), lambda _: done.set())
        self.assertTrue(done.wait(1))

    def test_reset_suppresses_inflight_result(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        model = FakeModel(gate)
        transcriber = self.make_transcriber(model)
        received = []
        transcriber._buffer = [np.ones(1600, dtype=np.float32)]
        transcriber._flush(received.append)
        self.assertTrue(model.started.wait(1))
        transcriber.reset_vad()
        gate.set()
        time.sleep(.08)
        self.assertEqual(received, [])

    def test_pending_queue_bounded_and_latest_preserved(self):
        gate = threading.Event()
        model = FakeModel(gate)
        transcriber = self.make_transcriber(model)
        received = []
        for value in range(1, 9):
            transcriber._buffer = [np.full(1600, value, dtype=np.float32)]
            transcriber._flush(received.append)
            if value == 1:
                self.assertTrue(model.started.wait(1))
        self.assertLessEqual(transcriber.pending_count, 2)
        gate.set()
        deadline = time.monotonic() + 1
        while len(received) < 3 and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(received, ['1', '7', '8'])

    def test_segment_cap_counts_silence_too(self):
        transcriber = self.make_transcriber()
        transcriber._max_segment_frames = 6
        done = threading.Event()
        for _ in range(4):
            transcriber.process_chunk(np.full(1600, .1, dtype=np.float32), lambda _: done.set())
        transcriber.process_chunk(np.zeros(1600, dtype=np.float32), lambda _: done.set())
        transcriber.process_chunk(np.full(1600, .1, dtype=np.float32), lambda _: done.set())
        self.assertTrue(done.wait(.3))

    def test_audio_callback_queue_is_bounded_and_not_blocking(self):
        capture = AudioCapture()
        self.addCleanup(capture.stop)
        gate = threading.Event()
        self.addCleanup(gate.set)
        class FakeStream:
            def __init__(self, **kwargs):
                self.callback = kwargs['callback']
            def start(self): pass
            def stop(self): pass
            def close(self): pass
        with patch('audio_capture.sd.InputStream', FakeStream):
            release = threading.Timer(.05, gate.set)
            release.start()
            self.addCleanup(release.join)
            capture.start(callback=lambda _: gate.wait(1))
            block = np.ones((1600, 1), dtype=np.float32)
            started = time.perf_counter()
            for _ in range(500):
                capture._stream.callback(block, 1600, None, None)
            elapsed = time.perf_counter() - started
            self.assertLessEqual(capture.audio_queue.qsize(), 20)
            self.assertLess(elapsed, .5)

    def test_failed_audio_start_leaves_stopped(self):
        capture = AudioCapture()
        self.addCleanup(capture.stop)
        with patch('audio_capture.sd.InputStream', side_effect=RuntimeError('unavailable')):
            with self.assertRaises(RuntimeError):
                capture.start()
        self.assertFalse(capture.is_capturing)

    def test_driver_overflow_reports_discontinuity_off_callback(self):
        capture = AudioCapture()
        self.addCleanup(capture.stop)
        notified = threading.Event()
        capture.on_discontinuity = notified.set
        class FakeStream:
            def __init__(self, **kwargs): self.callback = kwargs['callback']
            def start(self): pass
            def stop(self): pass
            def close(self): pass
        class Overflow:
            input_overflow = True
        with patch('audio_capture.sd.InputStream', FakeStream):
            capture.start(callback=lambda _: None)
            capture._stream.callback(np.ones((1600, 1), np.float32), 1600, None, Overflow())
            self.assertTrue(notified.wait(1))
            self.assertEqual(capture.input_overflows, 1)
