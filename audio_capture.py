"""
audio_capture.py
----------------
Captures system audio (loopback) from video conferencing software.

推荐方案 (Windows): VB-CABLE Virtual Audio Device
  安装: https://vb-audio.com/Cable/ (VBCABLE_Driver_Pack45.exe)
  安装后会新增两个设备：
    • CABLE Input  (虚拟扬声器) — 在会议软件中将「扬声器」设置为此设备
    • CABLE Output (虚拟麦克风) — 本工具从此设备读取音频

  ⚠️  使用 CABLE Input 作为扬声器后听不到声音的解决方法：
  控制面板 → 声音 → 录制 → CABLE Output → 属性
  → 侦听 → 勾选「侦听此设备」→ 选择你的真实耳机/扬声器 → 确定

On Linux:  Select a PulseAudio/PipeWire "monitor" source.
On macOS:  Install BlackHole (https://github.com/ExistentialAudio/BlackHole)
           then select it as the input device.
"""

import sounddevice as sd
import numpy as np
import queue
import threading
from typing import Optional, Callable, List, Dict


class AudioCapture:
    """Manages audio device listing and real-time capture."""

    SAMPLE_RATE = 16000
    CHUNK_MS = 100        # 100 ms per callback chunk

    def __init__(self):
        self._chunk_frames = int(self.SAMPLE_RATE * self.CHUNK_MS / 1000)
        self._stream: Optional[sd.InputStream] = None
        self.is_capturing = False
        # Raw audio queue for consumers that want to pull data themselves
        self.audio_queue: queue.Queue = queue.Queue(maxsize=20)
        self._worker = None
        self._stop_event = threading.Event()
        self.dropped_chunks = 0
        self.on_error = lambda _: None

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> List[Dict]:
        """
        Return all usable input devices including system-audio monitors.

        Each entry: {id, name, channels, sample_rate, is_monitor}
        """
        devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] < 1:
                continue
            name: str = dev["name"]
            name_lower = name.lower()
            # VB-CABLE Output is the virtual mic we read from.
            # Also keep Linux monitor sources and other loopback devices.
            is_vbcable = "cable output" in name_lower
            is_monitor = (
                is_vbcable
                or "monitor" in name_lower
                or "loopback" in name_lower
                or "what u hear" in name_lower
            )
            devices.append(
                {
                    "id": idx,
                    "name": name,
                    "channels": dev["max_input_channels"],
                    "sample_rate": int(dev["default_samplerate"]),
                    "is_monitor": is_monitor,
                    "is_vbcable": is_vbcable,
                }
            )
        return devices

    @staticmethod
    def preferred_device(devices: List[Dict]) -> Optional[int]:
        """
        Return the id of the best default device for meeting audio capture.
        Priority: VB-CABLE Output > other monitors > first available.
        """
        # 1st priority: VB-CABLE Output
        for dev in devices:
            if dev.get("is_vbcable"):
                return dev["id"]
        # 2nd priority: any monitor/loopback
        for dev in devices:
            if dev["is_monitor"]:
                return dev["id"]
        return devices[0]["id"] if devices else None

    # ------------------------------------------------------------------
    # Capture control
    # ------------------------------------------------------------------

    def start(
        self,
        device_id: Optional[int] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        """
        Begin capturing audio.

        Args:
            device_id: sounddevice device index. None = system default.
            callback:  Called with each (chunk_frames,) float32 ndarray.
        """
        if self.is_capturing:
            return

        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def _sd_callback(indata, frames, time_info, status):
            if not self.is_capturing or stop_event.is_set():
                return
            mono = indata[:, 0].astype(np.float32, copy=True)
            try:
                self.audio_queue.put_nowait(mono)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                self.dropped_chunks += 1
                try:
                    self.audio_queue.put_nowait(mono)
                except queue.Full:
                    self.dropped_chunks += 1

        def dispatch():
            last_dropped = 0
            while not stop_event.is_set():
                try:
                    mono = self.audio_queue.get(timeout=.1)
                except queue.Empty:
                    continue
                if stop_event.is_set():
                    break
                try:
                    if self.dropped_chunks > last_dropped:
                        last_dropped = self.dropped_chunks
                        self.on_error('音频处理积压，已丢弃过期音频块')
                    if callback:
                        callback(mono)
                except Exception:
                    self.on_error('音频处理失败，请停止并重新开始监听')

        try:
            self._stream = sd.InputStream(
                device=device_id, channels=1, samplerate=self.SAMPLE_RATE,
                blocksize=self._chunk_frames, dtype='float32', callback=_sd_callback,
            )
            self.is_capturing = True
            self._worker = threading.Thread(target=dispatch, name='audio-worker', daemon=True)
            self._worker.start()
            self._stream.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop capturing and release the audio stream."""
        self.is_capturing = False
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                self.on_error('音频设备停止失败，正在尝试释放设备')
            finally:
                try:
                    self._stream.close()
                except Exception:
                    self.on_error('音频设备释放失败，请检查设备连接')
            self._stream = None
        if self._worker and threading.current_thread() is not self._worker:
            self._worker.join(timeout=.25)
        self._worker = None
        # Drain the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def rms(audio: np.ndarray) -> float:
        """Root-mean-square energy of an audio chunk."""
        return float(np.sqrt(np.mean(audio ** 2) + 1e-9))
