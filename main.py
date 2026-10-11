"""
main.py
-------
Entry point for Interview Assistant.

Wires together:
  AudioCapture  ──(raw audio)──▶  Transcriber  ──(text)──▶  GUI
"""

from __future__ import annotations

import sys
import threading
import numpy as np

from audio_capture import AudioCapture
from transcriber import Transcriber, _detect_device
from gui import InterviewAssistantGUI
from ai_client import AIError, AnswerService, RoutingClient


def _get_hw_info() -> str:
    """Detect and return a human-readable CPU or GPU description."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return f"GPU: {name}  ({mem:.1f} GB VRAM)"
    except Exception:
        pass
    # CPU fallback
    try:
        import platform
        cpu = platform.processor() or platform.machine()
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if "model name" in line:
                        cpu = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        return f"CPU: {cpu}"
    except Exception:
        return "硬件信息不可用"


def _check_deps() -> None:
    """Print a friendly error if required packages are missing."""
    missing = []
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")
    try:
        import funasr  # noqa: F401
    except ImportError:
        missing.append("funasr")
    try:
        import modelscope  # noqa: F401
    except ImportError:
        missing.append("modelscope")
    if missing:
        print(
            "\n❌  缺少依赖包，请先运行安装脚本：\n"
            "   Linux/macOS: bash setup.sh\n"
            "   Windows:     setup_windows.bat\n\n"
            f"   缺少的包: {', '.join(missing)}\n"
        )
        sys.exit(1)


class InterviewAssistant:
    """Top-level orchestrator."""

    def __init__(self) -> None:
        self._audio       = AudioCapture()
        self._gui         = InterviewAssistantGUI()
        self._listening   = False
        self._listening_generation = 0
        self._closed = False
        self._model_loading = False
        self._transcriber = self._new_transcriber('zh', _detect_device())
        self._ai = AnswerService(RoutingClient(), self._gui.update_ai_event)
        self._audio.on_error = self._gui.set_status

        # Wire GUI callbacks
        self._gui.on_start        = self._start_listening
        self._gui.on_stop         = self._stop_listening
        self._gui.on_reload_model = self._reload_model
        self._gui.on_run_ai = lambda: self._start_ai(self._gui.get_current_text())
        self._gui.on_cancel_ai = self._cancel_ai
        self._gui.on_apply_ai = self._apply_ai_config
        self._gui.on_simulate_ai = self._simulate_ai
        self._gui.on_transcript_received = self._accept_transcript
        self._gui.root.protocol('WM_DELETE_WINDOW', self._close)

        # Populate device list
        devices = AudioCapture.list_devices()
        if not devices:
            self._gui.set_status("⚠️ 未找到音频输入设备")
        else:
            self._gui.set_devices(devices)

        # Detect hardware info in background (torch import can be slow)
        threading.Thread(
            target=lambda: self._gui.set_hw_info(_get_hw_info()),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _new_transcriber(self, language, device):
        return Transcriber(language=language, device=device,
                           on_metrics=self._gui.update_asr_metrics, on_error=self._gui.set_status)

    def _load_model_bg(self, transcriber=None) -> None:
        """Load FunASR Paraformer in a background thread."""
        transcriber = transcriber or self._transcriber
        def notify(text):
            if not self._closed and transcriber is self._transcriber:
                self._gui.set_status(text)
        try:
            transcriber.load_model(progress_callback=notify)
        except Exception:
            notify('❌ 模型加载失败，请检查依赖、网络、模型缓存和计算设备')
        finally:
            if transcriber is self._transcriber:
                self._model_loading = False

    def _reload_model(self) -> None:
        """Rebuild Transcriber with current GUI settings and reload the model."""
        if self._model_loading:
            self._gui.set_status('当前模型仍在加载，请稍候再切换')
            return
        if self._listening:
            self._stop_listening()
        lang   = self._gui.get_language()
        device = self._gui.get_compute_device()
        self._gui.set_status(f"⏳ 正在重新加载模型（{lang} / {device}）…")
        self._transcriber.close()
        self._transcriber = self._new_transcriber(lang, device)
        self._model_loading = True
        threading.Thread(target=self._load_model_bg, args=(self._transcriber,), daemon=True).start()

    # ------------------------------------------------------------------
    # Listen control
    # ------------------------------------------------------------------

    def _start_listening(self) -> None:
        if self._gui.get_language() != self._transcriber.language:
            self._reload_model()
            return
        if not self._transcriber.is_loaded:
            self._gui.set_status("⏳ 模型尚未加载完成，请稍候…")
            return

        # Apply current language and VAD settings
        self._transcriber.set_vad_params(
            self._gui.get_vad_threshold(),
            self._gui.get_vad_silence(),
        )
        self._transcriber.reset_vad()
        self._listening_generation += 1
        generation = self._listening_generation
        transcriber = self._transcriber

        device_id = self._gui.get_selected_device_id()
        self._listening = True
        self._gui.set_listening(True)

        def _audio_cb(chunk: np.ndarray) -> None:
            rms = AudioCapture.rms(chunk)
            self._gui.update_volume(rms)
            transcriber.process_chunk(chunk, lambda text: self._on_transcript(text, generation))

        try:
            self._audio.start(device_id=device_id, callback=_audio_cb)
        except Exception as exc:
            self._listening = False
            self._listening_generation += 1
            self._transcriber.reset_vad()
            self._gui.set_listening(False)
            self._gui.set_status(f"❌ 无法打开音频设备: {exc}")

    def _stop_listening(self) -> None:
        self._listening = False
        self._listening_generation += 1
        self._audio.stop()
        self._transcriber.reset_vad()
        self._listening = False
        self._gui.set_listening(False)
        self._gui.update_volume(0.0)
        self._cancel_ai()

    # ------------------------------------------------------------------
    # Transcription callback (background thread → GUI queue)
    # ------------------------------------------------------------------

    def _on_transcript(self, text: str, generation=None) -> None:
        self._gui.append_transcription(text, generation)

    def _accept_transcript(self, text, generation):
        # This callback runs on the Tk main thread, never in the ASR worker.
        if self._closed or not self._listening or generation != self._listening_generation:
            return False
        if self._gui._auto_ai_var.get():
            self._start_ai(text)
        return True

    def _apply_ai_config(self):
        try:
            self._gui.get_ai_settings().validate()
            self._cancel_ai()
            self._gui.invalidate_ai_request('配置已应用，Key 仅保留在本次运行内存中')
        except (AIError, ValueError) as exc:
            message = str(exc) if isinstance(exc, AIError) else '回答上限须为整数'
            self._gui.invalidate_ai_request(message)

    def _start_ai(self, prompt, simulation=False):
        try:
            settings = self._gui.get_ai_settings(simulation=simulation)
            request_id = self._ai.submit(settings, prompt)
            self._gui.begin_ai_request(request_id, simulation=simulation)
        except (AIError, ValueError) as exc:
            message = str(exc) if isinstance(exc, AIError) else '回答上限须为整数'
            self._gui.invalidate_ai_request(message)

    def _simulate_ai(self):
        self._start_ai(self._gui.get_current_text() or '请说明一个项目的实现思路。', simulation=True)

    def _cancel_ai(self):
        self._ai.cancel()
        self._gui.invalidate_ai_request()

    def _close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_listening()
        self._transcriber.close()
        self._ai.close()
        self._gui._api_key_var.set('')
        self._gui._closing = True
        self._gui.root.destroy()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        # Load model in background so GUI is immediately responsive
        self._model_loading = True
        threading.Thread(target=self._load_model_bg, args=(self._transcriber,), daemon=True).start()
        self._gui.run()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _check_deps()
    app = InterviewAssistant()
    app.run()
