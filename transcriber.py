"""Local FunASR ASR with bounded FIFO inference and stale-result isolation."""
from __future__ import annotations
import math
import queue
import threading
import time
import numpy as np

SILENCE_THRESHOLD, SILENCE_SECONDS = .012, 1.2
MIN_SPEECH_SECONDS, MAX_SEGMENT_SECONDS = .4, 30.
SAMPLE_RATE, CHUNK_SECONDS = 16000, .1
_MODELS = {'zh': 'paraformer-zh', 'en': 'paraformer-en', 'auto': 'iic/SenseVoiceSmall'}


def _detect_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def _gpu_info():
    import torch
    if not torch.cuda.is_available():
        return ''
    index = torch.cuda.current_device()
    memory = torch.cuda.get_device_properties(index).total_memory // (1024 ** 2)
    return f'{torch.cuda.get_device_name(index)} ({memory} MB)'


class Transcriber:
    def __init__(self, language='zh', device='cpu', max_pending=2,
                 on_metrics=None, on_error=None, reuse_resources=True,
                 streaming=False, on_partial=None):
        if language not in _MODELS or device not in ('cpu', 'cuda') or max_pending < 1:
            raise ValueError('Invalid ASR settings')
        self.reuse_resources = bool(reuse_resources)
        if streaming and language != 'zh':
            raise ValueError('流式模式目前仅支持中文；请切回整句模式')
        self.streaming = bool(streaming)
        self.on_partial = on_partial or (lambda _: None)
        self._stream_id = 0
        self._drop_utterance = False
        self.resource_reuse_active = False
        self.language, self._device = language, device
        self._model, self._loaded = None, False
        self._buffer: list[np.ndarray] = []
        self._speech_frames = self._silence_frames = self._total_frames = 0
        self._silence_threshold = SILENCE_THRESHOLD
        self._silence_frame_limit = math.ceil(SILENCE_SECONDS / CHUNK_SECONDS - 1e-9)
        self._min_speech_frames = math.ceil(MIN_SPEECH_SECONDS / CHUNK_SECONDS - 1e-9)
        self._max_segment_frames = math.ceil(MAX_SEGMENT_SECONDS / CHUNK_SECONDS)
        self._jobs = queue.Queue(maxsize=max_pending)
        self._state_lock = threading.RLock()
        self._closed = threading.Event()
        self._generation = 0
        self.on_metrics = on_metrics or (lambda _: None)
        self.on_error = on_error or (lambda _: None)
        self._worker = threading.Thread(target=self._work, name='asr-worker', daemon=True)
        self._worker.start()

    @property
    def is_loaded(self):
        return self._loaded and not self._closed.is_set()

    @property
    def pending_count(self):
        return self._jobs.qsize()

    @property
    def has_pending_speech(self):
        with self._state_lock:
            return bool(self._total_frames or self._drop_utterance or self._jobs.unfinished_tasks)

    def discard_utterance(self):
        """Invalidate state after a capture gap; resume after a silent boundary."""
        with self._state_lock:
            self.reset_vad()
            self._drop_utterance = self.streaming

    def load_model(self, progress_callback=None):
        from funasr import AutoModel
        from importlib.metadata import version, PackageNotFoundError
        from funasr_runtime import enable_resource_reuse
        notify = progress_callback or (lambda _: None)
        description = _gpu_info() if self._device == 'cuda' else ''
        if self._device == 'cuda' and not description:
            raise RuntimeError('已选择 CUDA，但没有可用 CUDA 设备')
        model_name = 'paraformer-zh-streaming' if self.streaming else _MODELS[self.language]
        notify(f'正在加载 FunASR [{model_name} / {self._device}]…')
        options = dict(model=model_name, device=self._device,
                       disable_update=True, disable_pbar=True, log_level='ERROR')
        if self.streaming:
            model = AutoModel(**options)
        else:
            try:
                model = AutoModel(vad_model='fsmn-vad', punc_model='ct-punc', **options)
            except Exception:
                notify('可选 VAD/标点模型加载失败，正在尝试精简模式…')
                model = AutoModel(**options)
        if self.reuse_resources and self.language in ('zh', 'en'):
            try:
                installed_version = version('funasr')
            except PackageNotFoundError:
                installed_version = ''
            self.resource_reuse_active = enable_resource_reuse(model, installed_version)
        with self._state_lock:
            if self._closed.is_set():
                return
            self._model, self._loaded = model, True
        reuse_status = '资源复用已启用' if self.resource_reuse_active else '兼容模式'
        notify(f'✅ FunASR 模型加载完成 [{description or "CPU"} / {reuse_status}]，可以开始监听')

    def process_chunk(self, chunk, on_transcript):
        if self.streaming:
            self._process_stream_chunk(chunk, on_transcript)
            return
        with self._state_lock:
            if not self.is_loaded or not chunk.size:
                return
            speech = float(np.sqrt(np.mean(chunk ** 2) + 1e-9)) > self._silence_threshold
            if not speech and not self._buffer:
                return
            self._buffer.append(np.asarray(chunk, dtype=np.float32).copy())
            self._total_frames += 1
            if speech:
                self._speech_frames += 1
                self._silence_frames = 0
            else:
                self._silence_frames += 1
            if self._total_frames >= self._max_segment_frames or self._silence_frames >= self._silence_frame_limit:
                if self._speech_frames >= self._min_speech_frames:
                    self._flush(on_transcript)
                else:
                    self._reset()

    def _process_stream_chunk(self, chunk, callback):
        with self._state_lock:
            if not self.is_loaded or not chunk.size:
                return
            speech = float(np.sqrt(np.mean(chunk ** 2) + 1e-9)) > self._silence_threshold
            if self._drop_utterance:
                self._silence_frames = 0 if speech else self._silence_frames + 1
                if self._silence_frames >= self._silence_frame_limit:
                    self._drop_utterance = False
                    self._reset()
                    self._stream_id += 1
                return
            if not speech and self._total_frames == 0:
                return
            self._buffer.append(np.asarray(chunk, dtype=np.float32).copy())
            self._total_frames += 1
            self._speech_frames += int(speech)
            self._silence_frames = 0 if speech else self._silence_frames + 1
            final = (self._silence_frames >= self._silence_frame_limit
                     or self._total_frames >= self._max_segment_frames)
            if final and self._speech_frames < self._min_speech_frames:
                self._reset()
                self._stream_id += 1
                return
            if len(self._buffer) >= 6 or final:
                job = {'audio': np.concatenate(self._buffer), 'callback': callback,
                       'generation': self._generation, 'utterance': self._stream_id,
                       'final': final, 'queued_at': time.perf_counter()}
                self._buffer.clear()
                try:
                    self._jobs.put_nowait(job)
                except queue.Full:
                    # Stateful audio cannot drop a middle chunk and continue.
                    self.reset_vad()
                    self._drop_utterance = not final
                    self.on_error('流式识别积压，本句已丢弃；请切换计算设备或整句模式')
                    return
            if final:
                self._reset()
                self._stream_id += 1

    def _reset(self):
        self._buffer.clear()
        self._speech_frames = self._silence_frames = self._total_frames = 0

    def _flush(self, on_transcript):
        with self._state_lock:
            if not self._buffer or self._closed.is_set():
                return
            audio = np.concatenate(self._buffer).astype(np.float32, copy=False)
            self._reset()
            job = (audio, on_transcript, self._generation, time.perf_counter())
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                try:
                    self._jobs.get_nowait()
                    self._jobs.task_done()
                except queue.Empty:
                    pass
                self._jobs.put_nowait(job)
                self.on_error('识别队列已满，已丢弃最早等待的语音段；请检查推理速度')

    def _work(self):
        stream_key, stream_cache, stream_text = None, {}, ''
        while not self._closed.is_set():
            try:
                job = self._jobs.get(timeout=.1)
            except queue.Empty:
                continue
            if isinstance(job, dict):
                audio, callback, generation, queued_at = (job['audio'], job['callback'], job['generation'], job['queued_at'])
            else:
                audio, callback, generation, queued_at = job
            try:
                with self._state_lock:
                    if generation != self._generation or not self.is_loaded:
                        continue
                    model = self._model
                started = time.perf_counter()
                if self.streaming:
                    key = (generation, job['utterance'])
                    if stream_key != key:
                        stream_key, stream_cache, stream_text = key, {}, ''
                    result = model.generate(input=audio, cache=stream_cache,
                                            is_final=job['final'], chunk_size=[0, 10, 5],
                                            encoder_chunk_look_back=4, decoder_chunk_look_back=1,
                                            disable_pbar=True)
                else:
                    result = model.generate(input=audio, batch_size_s=30, disable_pbar=True)
                finished = time.perf_counter()
                with self._state_lock:
                    if generation != self._generation or self._closed.is_set():
                        continue
                    self.on_metrics({'queue_ms': (started - queued_at) * 1000,
                                     'inference_ms': (finished - started) * 1000,
                                     'audio_seconds': audio.size / SAMPLE_RATE})
                    if self.streaming:
                        # Paraformer streaming returns only this block's tokens.
                        fragment = result[0].get('text', '').strip() if result else ''
                        stream_text += fragment
                        if job['final']:
                            self.on_partial('')
                            if stream_text.strip():
                                callback(stream_text.strip())
                            stream_key, stream_cache, stream_text = None, {}, ''
                        elif stream_text:
                            self.on_partial(stream_text)
                    elif result and result[0].get('text'):
                        text = result[0]['text'].strip()
                        if text:
                            callback(text)
            except Exception:
                with self._state_lock:
                    if generation == self._generation and not self._closed.is_set():
                        if self.streaming:
                            self.reset_vad()
                            self._drop_utterance = not job['final']
                            stream_key, stream_cache, stream_text = None, {}, ''
                        self.on_error('语音识别失败，请检查模型、计算设备或输入音频')
            finally:
                self._jobs.task_done()

    def set_language(self, lang):
        if lang not in _MODELS:
            raise ValueError('Unsupported language')
        with self._state_lock:
            if lang != self.language:
                self.reset_vad()
                self.language, self._loaded, self._model = lang, False, None

    def set_vad_params(self, threshold, silence_seconds):
        threshold, silence_seconds = float(threshold), float(silence_seconds)
        if not math.isfinite(threshold) or not math.isfinite(silence_seconds):
            raise ValueError('VAD settings must be finite')
        with self._state_lock:
            self._silence_threshold = min(.5, max(.001, threshold))
            self._silence_frame_limit = max(1, math.ceil(max(.1, min(3., silence_seconds)) / CHUNK_SECONDS - 1e-9))

    def reset_vad(self):
        with self._state_lock:
            self._generation += 1
            self._drop_utterance = False
            self._stream_id += 1
            self._reset()
            while True:
                try:
                    self._jobs.get_nowait()
                    self._jobs.task_done()
                except queue.Empty:
                    break

    def close(self):
        self._closed.set()
        self.reset_vad()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=.25)
