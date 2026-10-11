"""Offline synthetic pipeline benchmark; never opens audio or HTTP devices."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from transcriber import Transcriber
from ai_client import AnswerService, DeepSeekSettings, RoutingClient


class SyntheticASR:
    def generate(self, input, **kwargs):
        time.sleep(.025)
        return [{'text': '请说明项目的设计思路。'}]


def run_trial(silence):
    done = threading.Event()
    result = {}
    def receive(event):
        if event['kind'] == 'done':
            result['ai'] = event['metrics']
            done.set()
        elif event['kind'] == 'error':
            result['error'] = event['text']
            done.set()
    class ForbiddenNetwork:
        def stream(self, *args, **kwargs):
            raise AssertionError('Real network forbidden during simulation')
    ai = AnswerService(RoutingClient(ForbiddenNetwork()), receive)
    asr = Transcriber(on_metrics=lambda metrics: result.update(asr=metrics))
    try:
        asr._model, asr._loaded = SyntheticASR(), True
        asr.set_vad_params(.012, silence)
        block = (.1 * np.sin(2 * np.pi * 200 * np.arange(1600) / 16000)).astype(np.float32)
        on_text = lambda text: ai.submit(DeepSeekSettings(simulation=True), text)
        started = time.perf_counter()
        for _ in range(8):
            asr.process_chunk(block, on_text)
        for _ in range(math.ceil(silence / .1 - 1e-9)):
            asr.process_chunk(np.zeros(1600, dtype=np.float32), on_text)
        if not done.wait(3) or 'error' in result:
            raise RuntimeError('Synthetic pipeline did not complete')
        result['accelerated_wall_ms'] = (time.perf_counter() - started) * 1000
        # Audio is injected faster than real time. Add its logical silence wait;
        # actual end-to-end audio/hardware/network performance is not measured.
        result['logical_silence_ms'] = silence * 1000
        result['estimated_first_answer_ms'] = silence * 1000 + result['asr']['queue_ms'] + result['asr']['inference_ms'] + result['ai']['first_token_ms']
        return result
    finally:
        asr.close()
        ai.close()


def buffer_comparison():
    chunks = [np.ones(1600, dtype=np.float32) for _ in range(200)]
    legacy = []
    started = time.perf_counter()
    for chunk in chunks:
        legacy.extend(chunk.tolist())
    legacy_ms = (time.perf_counter() - started) * 1000
    legacy_bytes = sys.getsizeof(legacy) + sum(sys.getsizeof(x) for x in legacy)
    blocks = []
    started = time.perf_counter()
    for chunk in chunks:
        blocks.append(chunk.copy())
    blocks_ms = (time.perf_counter() - started) * 1000
    blocks_bytes = sys.getsizeof(blocks) + sum(sys.getsizeof(x) for x in blocks)
    return {'audio_seconds': 20, 'legacy_list_ms': legacy_ms, 'numpy_blocks_ms': blocks_ms,
            'legacy_bytes': legacy_bytes, 'numpy_blocks_bytes': blocks_bytes,
            'scope': 'Only buffering primitive comparison, not whole ASR speed'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='.local/latency-report.json')
    args = parser.parse_args()
    trials = {str(s): [run_trial(s) for _ in range(5)] for s in (1.2, .6)}
    summaries = {}
    for silence, values in trials.items():
        summaries[silence] = {
            'samples': len(values),
            'queue_ms_p50': statistics.median(v['asr']['queue_ms'] for v in values),
            'mock_inference_ms_p50': statistics.median(v['asr']['inference_ms'] for v in values),
            'mock_ai_first_token_ms_p50': statistics.median(v['ai']['first_token_ms'] for v in values),
            'mock_ai_total_ms_p50': statistics.median(v['ai']['total_ms'] for v in values),
            'estimated_first_answer_ms_p50': statistics.median(v['estimated_first_answer_ms'] for v in values),
        }
    report = {'kind': 'offline-synthetic-only', 'python': sys.version.split()[0],
              'limits': 'Synthetic ASR sleeps 25 ms; simulated AI emits every 80 ms. No real ASR/cloud speed or recognition accuracy measured. UI scheduling/network overhead excluded from estimate.',
              'summaries': summaries, 'buffer_comparison': buffer_comparison(), 'trials': trials}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'summaries': summaries}, ensure_ascii=False))


if __name__ == '__main__':
    main()
