"""Explicit synthetic VB-CABLE benchmark; logs/results only in an ignored RunId.

Usage: --run-dir .local/<unique-id> --audio <synthetic-question.wav>
        --input-device 2 --output-device 5 [--real-ai]
Only the hard-coded synthetic question is allowed through the real-AI gate.
"""
import argparse
import difflib
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import unicodedata
import wave

EXPECTED = '请解释如何提高检索增强生成系统的检索准确率，并说明你会怎样评估优化效果。'


def normalized(text):
    return ''.join(c for c in text if not unicodedata.category(c).startswith(('P', 'Z')) and not c.isspace())


def allowed_prompt(text):
    ratio = difflib.SequenceMatcher(None, normalized(text), normalized(EXPECTED)).ratio()
    if ratio < .85:
        raise ValueError('Transcript does not match the permitted synthetic question')
    return EXPECTED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--input-device', type=int, required=True)
    parser.add_argument('--output-device', type=int, required=True)
    parser.add_argument('--real-ai', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    run = Path(args.run_dir).resolve()
    if not run.is_relative_to(project / '.local'):
        raise ValueError('Run directory must be under .local')
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'worker-pid.txt').exists() or (run / 'exit-code.txt').exists():
        raise ValueError('RunId already used')
    sys.path.insert(0, str(project))
    code, app = 1, None
    report = {'actual_device_capture': True, 'real_ai': args.real_ai, 'private_resume_sent': False, 'cases': []}
    report['source_sha256'] = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
                               for name in ('main.py', 'transcriber.py', 'audio_capture.py', 'question_gate.py')}
    with (run / 'run.log').open('w', encoding='utf-8', buffering=1) as log:
        sys.stdout = sys.stderr = log
        (run / 'worker-pid.txt').write_text(str(os.getpid()))

        def save(stage):
            (run / 'progress.json').write_text(json.dumps({'stage': stage}), encoding='utf-8')
            (run / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

        try:
            import numpy as np
            import sounddevice as sd
            import main as controller
            from audio_capture import AudioCapture
            with wave.open(args.audio, 'rb') as wav:
                if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError('Expected mono PCM16 at 16 kHz')
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32) / 32768
            if 'cable output' not in sd.query_devices(args.input_device)['name'].lower():
                raise ValueError('Input must be the explicit VB-CABLE Output device')
            if 'cable input' not in sd.query_devices(args.output_device)['name'].lower():
                raise ValueError('Output must be the explicit VB-CABLE Input device')
            report['audio_seconds'] = len(pcm) / 16000
            for streaming in (False, True):
                mode = 'streaming' if streaming else 'offline'
                save('loading ' + mode)
                app = controller.InterviewAssistant()
                app._gui.root.withdraw()
                if streaming:
                    app._transcriber.close()
                    app._gui._streaming_var.set(True)
                    app._transcriber = app._new_transcriber('zh', app._gui.get_compute_device())
                app._gui._auto_ai_var.set(True)
                app._gui._use_resume_var.set(False)
                metric_events, partial_events = [], []
                original_metrics = app._transcriber.on_metrics
                app._transcriber.on_metrics = lambda m: (metric_events.append(m), original_metrics(m))
                original_partial = app._transcriber.on_partial
                app._transcriber.on_partial = lambda t: (partial_events.append((time.perf_counter(), t)), original_partial(t))
                original_accept = app._gui.on_transcript_received
                def guarded(text, generation):
                    # Canonical synthetic prompt only: incidental audio text can
                    # never enter a paid request, even if similarity is high.
                    return original_accept(allowed_prompt(text), generation)
                app._gui.on_transcript_received = guarded
                if not args.real_ai:
                    original_start = app._start_ai
                    app._start_ai = lambda text: original_start(text, simulation=True)
                loading_started = time.perf_counter()
                app._model_loading = True
                loader = threading.Thread(target=app._load_model_bg, daemon=True)
                loader.start()
                until = time.perf_counter() + 600
                while loader.is_alive() and time.perf_counter() < until:
                    app._gui.root.update()
                    time.sleep(.02)
                if not app._transcriber.is_loaded:
                    raise RuntimeError('Model loading failed or timed out')
                report[mode + '_load_ms'] = (time.perf_counter() - loading_started) * 1000
                app._gui.root.update()
                app._gui._device_var.set(next(k for k, v in app._gui._device_map.items() if v == args.input_device))
                for iteration in range(2):
                    save(mode + ' case ' + str(iteration))
                    metric_events.clear()
                    partial_events.clear()
                    capture_times, events, playback_errors = [], [], []
                    original_capture = app._audio.start
                    def tapped_start(device_id=None, callback=None):
                        def tapped(chunk):
                            if AudioCapture.rms(chunk) > .012:
                                capture_times.append(time.perf_counter())
                            callback(chunk)
                        return original_capture(device_id, tapped)
                    app._audio.start = tapped_start
                    original_event = app._ai.on_event
                    app._ai.on_event = lambda e: (events.append((time.perf_counter(), e)), original_event(e))
                    app._gui._current_box.delete('1.0', 'end')
                    app._start_listening()
                    if not app._audio.is_capturing:
                        raise RuntimeError('Capture did not start')
                    def play():
                        try:
                            time.sleep(.3)
                            with sd.OutputStream(device=args.output_device, channels=2, samplerate=16000, dtype='float32') as output:
                                output.write(np.repeat(pcm[:, None], 2, axis=1))
                        except Exception:
                            playback_errors.append(True)
                    playback = threading.Thread(target=play, daemon=True)
                    playback.start()
                    until, transcript_at = time.perf_counter() + 90, None
                    while time.perf_counter() < until:
                        app._gui.root.update()
                        if transcript_at is None and app._gui.get_current_text():
                            transcript_at = time.perf_counter()
                        if any(e['kind'] in ('done', 'error') for _, e in events):
                            break
                        time.sleep(.005)
                    last = capture_times[-1] if capture_times else None
                    first = next((t for t, e in events if e['kind'] == 'delta'), None)
                    terminal = next(((t, e) for t, e in events if e['kind'] in ('done', 'error')), None)
                    def after_end(t):
                        return (t - last) * 1000 if t and last else None
                    transcript = app._gui.get_current_text()
                    ratio = difflib.SequenceMatcher(None, normalized(transcript), normalized(EXPECTED)).ratio()
                    report['cases'].append({'mode': mode, 'iteration': iteration, 'transcript': transcript,
                        'similarity': ratio, 'partial_before_end': any(t < last and text for t, text in partial_events) if last else False,
                        'end_to_text_ms': after_end(transcript_at), 'end_to_first_ms': after_end(first),
                        'end_to_done_ms': after_end(terminal[0]) if terminal else None,
                        'audio_dropped_chunks': app._audio.dropped_chunks,
                        'input_overflows': app._audio.input_overflows, 'asr_metrics': list(metric_events),
                        'ai': terminal[1] if terminal else None})
                    app._stop_listening()
                    app._audio.start = original_capture
                    app._ai.on_event = original_event
                    playback.join(2)
                    save(mode + ' case complete')
                    if playback_errors or not terminal or terminal[1]['kind'] != 'done' or ratio < .85:
                        raise RuntimeError('Synthetic chain did not complete correctly')
                app._close()
                app = None
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            code = 0
        except Exception:
            traceback.print_exc()
            report['error'] = '验证未完成，诊断保存在本机日志'
        finally:
            if app:
                app._close()
            report['exit_code'] = code
            save('finished')
            (run / 'exit-code.txt').write_text(str(code))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
