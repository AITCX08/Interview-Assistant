"""
gui.py
------
Floating Tkinter window for Interview Assistant.

Layout
------
┌─────────────────────────────────────────────────────┐
│  🎙 Interview Assistant                  [置顶 ✓]   │
├─────────────────────────────────────────────────────┤
│  音频设备: [设备选择 ▼]  语言:[中文▼]  [● 开始监听] │
│  状态: 就绪          音量 ████░░░░░░                 │
├─────────────────────────────────────────────────────┤
│  实时识别                                            │
│  ╔═════════════════════════════════════════════╗    │
│  ║  (转录内容实时追加显示)                      ║    │
│  ╚═════════════════════════════════════════════╝    │
│         [  🗑 删除  ]        [  📋 复制到历史  ]     │
├─────────────────────────────────────────────────────┤
│  历史对话                               [清空历史]   │
│  ╔═════════════════════════════════════════════╗    │
│  ║ [14:23] 你好，欢迎参加今天的会议…           ║    │
│  ║ [14:25] 我们来讨论一下第三季度目标…         ║    │
│  ╚═════════════════════════════════════════════╝    │
└─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import scrolledtext, ttk, filedialog
from typing import Callable, Dict, List, Optional
from ai_client import DeepSeekSettings, MODELS
from credentials import KeyStore, CredentialError
from resume_context import extract_resume, ResumeError
from extension_api import CORE_API_VERSION, ExtensionContext
from local_data import export_text, LocalDataError


def _default_compute_device() -> str:
    """'cuda' if a GPU is available, else 'cpu' — used as the dropdown default."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Catppuccin Mocha palette – easy on the eyes for long sessions
# ---------------------------------------------------------------------------
C = {
    "base":     "#1e1e2e",
    "surface0": "#313244",
    "surface1": "#45475a",
    "overlay":  "#6c7086",
    "text":     "#cdd6f4",
    "subtext":  "#a6adc8",
    "blue":     "#89b4fa",
    "green":    "#a6e3a1",
    "red":      "#f38ba8",
    "yellow":   "#f9e2af",
    "peach":    "#fab387",
    "mauve":    "#cba6f7",
}

# Model display name → internal language key
_LANG_DISPLAY = {
    "中文 (paraformer-zh)":  "zh",
    "英文 (paraformer-en)":  "en",
    "多语言 (SenseVoice)":   "auto",
}


class InterviewAssistantGUI:
    """Main application window."""

    def __init__(self, key_store=None) -> None:
        self._key_store = key_store or KeyStore()
        self._resume_generation = 0
        self._resume_busy = False
        self.root = tk.Tk()
        self.root.title("Interview Assistant · 面试助手")
        self.root.geometry("820x920")
        self.root.minsize(760, 720)
        self.root.configure(bg=C["base"])
        self.root.attributes("-topmost", True)

        # Public callbacks – set by main.py before calling run()
        self.on_start:        Optional[Callable[[], None]] = None
        self.on_stop:         Optional[Callable[[], None]] = None
        self.on_reload_model: Optional[Callable[[], None]] = None
        self.on_run_ai = None
        self.on_cancel_ai = None
        self.on_apply_ai = None
        self.on_simulate_ai = None
        self.on_transcript_received = None
        self.on_partial_received = None
        self.on_segment_discarded = None
        self._active_ai_request = None
        self._closing = False
        self._extensions = []
        self._extensions_closed = False
        self._poll_timer = None
        self.root.bind('<Destroy>', self._on_root_destroy, add='+')

        # Internal state
        self._is_listening = False
        self._device_map: Dict[str, int] = {}
        self._update_q: queue.SimpleQueue = queue.SimpleQueue()

        self._build_style()
        self._build_ui()
        self._poll_updates()

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        for widget in ("TCombobox", "TProgressbar"):
            style.configure(
                widget,
                fieldbackground=C["surface0"],
                background=C["surface0"],
                foreground=C["text"],
                troughcolor=C["surface1"],
                bordercolor=C["surface1"],
                darkcolor=C["surface0"],
                lightcolor=C["surface0"],
            )
        style.map("TCombobox", fieldbackground=[("readonly", C["surface0"])])
        style.configure('TNotebook', background=C['base'], bordercolor=C['surface1'])
        style.configure('TNotebook.Tab', background=C['surface1'], foreground=C['text'], padding=(10, 5))
        style.map('TNotebook.Tab', background=[('selected', C['surface0'])],
                  foreground=[('selected', C['blue'])])

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = dict(padx=14, pady=8)

        # ── Header ────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=C["base"])
        hdr.pack(fill=tk.X, **pad)

        tk.Label(
            hdr, text="🎙  Interview Assistant",
            font=("Helvetica", 15, "bold"),
            bg=C["base"], fg=C["blue"],
        ).pack(side=tk.LEFT)

        self._topmost_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            hdr, text="置顶", variable=self._topmost_var,
            bg=C["base"], fg=C["subtext"], activebackground=C["base"],
            selectcolor=C["surface0"], cursor="hand2",
            command=lambda: self.root.attributes("-topmost", self._topmost_var.get()),
        ).pack(side=tk.RIGHT)

        # ── Audio controls ────────────────────────────────────────────
        af = tk.LabelFrame(
            self.root, text="  音频设置  ",
            bg=C["base"], fg=C["blue"],
            font=("Helvetica", 10, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        af.pack(fill=tk.X, padx=14, pady=(0, 6))

        row1 = tk.Frame(af, bg=C["base"])
        row1.pack(fill=tk.X, padx=10, pady=(8, 4))

        tk.Label(row1, text="音频设备:", bg=C["base"], fg=C["text"],
                 font=("Helvetica", 10)).pack(side=tk.LEFT)

        self._device_var = tk.StringVar()
        self._device_combo = ttk.Combobox(
            row1, textvariable=self._device_var,
            width=32, state="readonly",
        )
        self._device_combo.pack(side=tk.LEFT, padx=(6, 14))

        tk.Label(row1, text="语音模型:", bg=C["base"], fg=C["text"],
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        self._lang_var = tk.StringVar(value="zh")
        _default_model = next(k for k, v in _LANG_DISPLAY.items() if v == "zh")
        self._model_display_var = tk.StringVar(value=_default_model)
        _mc = ttk.Combobox(
            row1, textvariable=self._model_display_var,
            values=list(_LANG_DISPLAY.keys()),
            width=22, state="readonly",
        )
        _mc.pack(side=tk.LEFT, padx=(6, 0))
        _mc.bind(
            "<<ComboboxSelected>>",
            lambda _: self._lang_var.set(
                _LANG_DISPLAY.get(self._model_display_var.get(), "zh")
            ),
        )

        row2 = tk.Frame(af, bg=C["base"])
        row2.pack(fill=tk.X, padx=10, pady=(0, 10))

        self._listen_btn = tk.Button(
            row2, text="▶  开始监听",
            bg=C["green"], fg=C["base"],
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, padx=16, pady=6,
            cursor="hand2",
            command=self._toggle_listen,
        )
        self._listen_btn.pack(side=tk.LEFT)

        tk.Label(row2, text="音量:", bg=C["base"], fg=C["subtext"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(18, 5))
        self._vol_bar = ttk.Progressbar(row2, length=130, maximum=100)
        self._vol_bar.pack(side=tk.LEFT)

        self._status_lbl = tk.Label(
            row2, text="就绪",
            bg=C["base"], fg=C["yellow"],
            font=("Helvetica", 9),
        )
        self._status_lbl.pack(side=tk.RIGHT)

        # ── Model & hardware settings ─────────────────────────────────
        self._build_model_panel()
        self._asr_metrics_var = tk.StringVar(value='识别：排队 — / 推理 —')
        self._ai_metrics_var = tk.StringVar(value='AI：首字 — / 完成 —')
        metrics_row = tk.Frame(self.root, bg=C['base'])
        metrics_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        for var in (self._asr_metrics_var, self._ai_metrics_var):
            tk.Label(metrics_row, textvariable=var, bg=C['base'], fg=C['subtext']).pack(side=tk.LEFT, padx=(0, 20))
        self._notebook = ttk.Notebook(self.root)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        transcript_tab = tk.Frame(self._notebook, bg=C['base'])
        self._ai_tab = tk.Frame(self._notebook, bg=C['base'])
        self._notebook.add(transcript_tab, text='语音转写')
        self._notebook.add(self._ai_tab, text='DeepSeek 建议与配置')

        # ── Current transcription ─────────────────────────────────────
        cf = tk.LabelFrame(
            transcript_tab, text="  实时识别  ",
            bg=C["base"], fg=C["blue"],
            font=("Helvetica", 10, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        cf.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 6))

        self._current_box = scrolledtext.ScrolledText(
            cf, height=7, wrap=tk.WORD,
            bg=C["surface0"], fg=C["text"],
            font=("Helvetica", 12),
            relief=tk.FLAT, padx=10, pady=10,
            insertbackground=C["text"],
            selectbackground=C["blue"], selectforeground=C["base"],
        )
        self._current_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 4))

        btn_row = tk.Frame(cf, bg=C["base"])
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 10))

        self._del_btn = tk.Button(
            btn_row, text="🗑  删除",
            bg=C["red"], fg="white",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, padx=20, pady=6,
            cursor="hand2",
            command=self._delete_current,
        )
        self._del_btn.pack(side=tk.LEFT)

        self._copy_btn = tk.Button(
            btn_row, text="📋  复制到历史",
            bg=C["blue"], fg=C["base"],
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, padx=20, pady=6,
            cursor="hand2",
            command=self._copy_to_history,
        )
        self._copy_btn.pack(side=tk.RIGHT)
        tk.Button(btn_row, text='导出当前文字', bg=C['surface1'], fg=C['text'],
                  relief=tk.FLAT, command=lambda: self._export_text(self.get_current_text())).pack(side=tk.RIGHT, padx=8)

        # ── History ───────────────────────────────────────────────────
        hf = tk.LabelFrame(
            transcript_tab, text="  历史对话  ",
            bg=C["base"], fg=C["blue"],
            font=("Helvetica", 10, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        hf.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        hist_top = tk.Frame(hf, bg=C["base"])
        hist_top.pack(fill=tk.X, padx=10, pady=(6, 2))

        tk.Label(hist_top, text="点击「复制到历史」后，对话自动记录在此",
                 bg=C["base"], fg=C["overlay"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT)

        tk.Button(
            hist_top, text="清空历史",
            bg=C["surface1"], fg=C["subtext"],
            font=("Helvetica", 9), relief=tk.FLAT, padx=10, pady=3,
            cursor="hand2",
            command=self._clear_history,
        ).pack(side=tk.RIGHT)
        tk.Button(hist_top, text='导出历史', bg=C['surface1'], fg=C['text'], relief=tk.FLAT,
                  command=lambda: self._export_text(self._history_box.get('1.0', tk.END))).pack(side=tk.RIGHT, padx=8)

        self._history_box = scrolledtext.ScrolledText(
            hf, height=7, wrap=tk.WORD,
            bg=C["surface0"], fg=C["text"],
            font=("Helvetica", 11),
            relief=tk.FLAT, padx=10, pady=10,
            state=tk.DISABLED,
            selectbackground=C["mauve"], selectforeground=C["base"],
        )
        self._history_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Text tags for history colouring
        self._history_box.tag_configure(
            "ts", foreground=C["overlay"], font=("Helvetica", 9)
        )
        self._history_box.tag_configure(
            "body", foreground=C["text"], font=("Helvetica", 11)
        )
        self._history_box.tag_configure(
            "sep", foreground=C["surface1"], font=("Helvetica", 7)
        )
        self._build_ai_panel()
        self._build_resume_panel()
        try:
            saved = self._key_store.load()
            self._api_key_var.set(saved)
            self._remember_key_var.set(bool(saved))
        except CredentialError:
            self._ai_status_lbl.configure(text='本机 Key 读取失败，请重新填写')

    def _build_ai_panel(self):
        settings = tk.LabelFrame(self._ai_tab, text=' DeepSeek 配置 ', bg=C['base'], fg=C['blue'])
        settings.pack(fill=tk.X, padx=10, pady=10)
        settings.columnconfigure(1, weight=1)
        self._api_key_var = tk.StringVar()
        self._ai_model_var = tk.StringVar(value=MODELS[0])
        self._ai_tokens_var = tk.StringVar(value='512')
        self._auto_ai_var = tk.BooleanVar(value=False)
        self._show_key_var = tk.BooleanVar(value=False)
        tk.Label(settings, text='API Key', bg=C['base'], fg=C['text']).grid(row=0, column=0, padx=8, pady=8)
        self._api_key_entry = tk.Entry(settings, textvariable=self._api_key_var, show='*',
            bg=C['surface0'], fg=C['text'], insertbackground=C['text'],
            selectbackground=C['blue'], selectforeground=C['base'], relief=tk.FLAT,
            highlightthickness=1, highlightbackground=C['surface1'], highlightcolor=C['blue'])
        self._api_key_entry.grid(row=0, column=1, sticky='ew', padx=8, pady=8)
        tk.Checkbutton(settings, text='显示', variable=self._show_key_var,
            command=lambda: self._api_key_entry.configure(show='' if self._show_key_var.get() else '*'),
            bg=C['base'], fg=C['text'], selectcolor=C['surface0'],
            activebackground=C['base'], activeforeground=C['text']).grid(row=0, column=2, padx=8)
        tk.Label(settings, text='模型', bg=C['base'], fg=C['text']).grid(row=1, column=0, padx=8, pady=6)
        ttk.Combobox(settings, textvariable=self._ai_model_var, values=MODELS, state='readonly').grid(row=1, column=1, sticky='ew', padx=8)
        self._remember_key_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings, text='本机加密记住 Key', variable=self._remember_key_var,
            bg=C['base'], fg=C['text'], selectcolor=C['surface0'],
            activebackground=C['base'], activeforeground=C['text']).grid(row=1, column=2, padx=8)
        tk.Label(settings, text='回答上限', bg=C['base'], fg=C['text']).grid(row=2, column=0, padx=8, pady=6)
        tk.Spinbox(settings, from_=64, to=2048, increment=64, textvariable=self._ai_tokens_var,
            bg=C['surface0'], fg=C['text'], buttonbackground=C['surface1'],
            insertbackground=C['text'], highlightbackground=C['surface1'], highlightcolor=C['blue'],
            width=10).grid(row=2, column=1, sticky='w', padx=8)
        tk.Button(settings, text='应用配置', command=self._on_apply_ai,
            bg=C['blue'], fg=C['base'], relief=tk.FLAT).grid(row=2, column=2, padx=8, pady=6)
        tk.Checkbutton(settings, text='识别完成后自动生成建议（上传当前语音段文字到 DeepSeek）',
            variable=self._auto_ai_var, bg=C['base'], fg=C['text'], selectcolor=C['surface0'],
            activebackground=C['base'], activeforeground=C['text']).grid(row=3, column=0, columnspan=3, sticky='w', padx=8, pady=8)
        tk.Label(settings, text='默认非思考模式、流式输出。自动模式可判定并合并问题；手动模式发送当前转写区文字。',
            bg=C['base'], fg=C['subtext'], wraplength=660, justify=tk.LEFT).grid(row=4, column=0, columnspan=3, sticky='w', padx=8, pady=(0, 8))
        buttons = tk.Frame(self._ai_tab, bg=C['base'])
        buttons.pack(fill=tk.X, padx=10)
        for text, callback, color in (
            ('生成建议', lambda: self.on_run_ai and self.on_run_ai(), 'green'),
            ('取消回答', lambda: self.on_cancel_ai and self.on_cancel_ai(), 'red'),
            ('本地模拟（不联网）', lambda: self.on_simulate_ai and self.on_simulate_ai(), 'mauve'),
        ):
            tk.Button(buttons, text=text, command=callback, bg=C[color], fg=C['base'],
                      relief=tk.FLAT, padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 10))
        self._ai_status_lbl = tk.Label(self._ai_tab, text='尚未发起 AI 请求', bg=C['base'], fg=C['subtext'],
                                       wraplength=680, justify=tk.LEFT)
        self._ai_status_lbl.pack(fill=tk.X, padx=10, pady=8)
        self._answer_box = scrolledtext.ScrolledText(self._ai_tab, height=10, wrap=tk.WORD,
            bg=C['surface0'], fg=C['text'], insertbackground=C['text'],
            selectbackground=C['blue'], selectforeground=C['base'],
            font=('Helvetica', 12), relief=tk.FLAT, padx=10, pady=10, state=tk.DISABLED)
        self._answer_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _build_resume_panel(self):
        panel = tk.Frame(self._notebook, bg=C['base'])
        self._notebook.add(panel, text='简历上下文（本地）')
        controls = tk.Frame(panel, bg=C['base'])
        controls.pack(fill=tk.X, padx=10, pady=10)
        self._resume_import_btn = tk.Button(controls, text='导入 PDF / DOCX', command=self._import_resume,
            bg=C['blue'], fg=C['base'], relief=tk.FLAT)
        self._resume_import_btn.pack(side=tk.LEFT)
        tk.Button(controls, text='清空简历上下文', command=self._clear_resume,
            bg=C['surface1'], fg=C['text'], relief=tk.FLAT).pack(side=tk.LEFT, padx=12)
        self._use_resume_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text='携带编辑后的简历上下文（真实回答时发送至 DeepSeek）',
            variable=self._use_resume_var, command=self._context_changed,
            bg=C['base'], fg=C['text'], selectcolor=C['surface0'],
            activebackground=C['base'], activeforeground=C['text']).pack(anchor='w', padx=10)
        tk.Label(panel, text='自动移除常见电话、邮箱、标注姓名与链接，无法保证完整匿名。请手动删除姓名、单位等身份线索后再勾选。\n预览可编辑；最多 12000 字符。仅本次运行保存，不上传文件，不保存提取文本。',
            wraplength=690, justify=tk.LEFT, bg=C['base'], fg=C['subtext']).pack(fill=tk.X, padx=10, pady=10)
        self._resume_status_var = tk.StringVar(value='尚未导入简历；上下文关闭')
        tk.Label(panel, textvariable=self._resume_status_var, bg=C['base'], fg=C['subtext']).pack(anchor='w', padx=10)
        self._resume_box = scrolledtext.ScrolledText(panel, wrap=tk.WORD, bg=C['surface0'], fg=C['text'],
            insertbackground=C['text'], selectbackground=C['blue'], selectforeground=C['base'], relief=tk.FLAT)
        self._resume_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._resume_box.bind('<<Modified>>', self._resume_edited)

    def _context_changed(self):
        if self.on_cancel_ai:
            self.on_cancel_ai()

    def _resume_edited(self, event=None):
        if self._resume_box.edit_modified():
            self._context_changed()
            self._resume_box.edit_modified(False)

    def _clear_resume(self):
        self._resume_generation += 1
        self._use_resume_var.set(False)
        self._resume_box.delete('1.0', tk.END)
        self._resume_status_var.set('简历上下文已清空')
        self._context_changed()

    def _import_resume(self):
        if self._resume_busy:
            return
        path = filedialog.askopenfilename(parent=self.root, filetypes=[('简历文档', '*.pdf *.docx')])
        if path:
            self.import_resume_path(path)

    def import_resume_path(self, path):
        if self._resume_busy:
            return
        self._clear_resume()
        generation = self._resume_generation
        self._resume_busy = True
        self._resume_import_btn.configure(state=tk.DISABLED)
        self._resume_status_var.set('正在本地解析…')
        def work():
            try:
                result = extract_resume(path)
                event = (generation, result, None)
            except ResumeError as exc:
                event = (generation, None, str(exc))
            self._update_q.put(('resume', event))
        threading.Thread(target=work, daemon=True).start()

    def get_ai_settings(self, simulation=False):
        return DeepSeekSettings(api_key=self._api_key_var.get().strip(),
                                model=self._ai_model_var.get(), max_tokens=int(self._ai_tokens_var.get()),
                                simulation=simulation,
                                resume_context=self._resume_box.get('1.0', tk.END).strip() if self._use_resume_var.get() else '')

    def get_current_text(self):
        return self._current_box.get('1.0', tk.END).strip()

    def _export_text(self, text):
        try:
            if export_text(self.root, text):
                self.set_status('已导出所选文字到本机')
        except LocalDataError as exc:
            self.set_status(str(exc))

    def mount_extension(self, title, factory):
        if self._extensions_closed:
            raise RuntimeError('扩展已关闭')
        parent = tk.Frame(self._notebook, bg=C['base'])
        context = ExtensionContext(CORE_API_VERSION, self.get_ai_settings,
            lambda: self._resume_box.get('1.0', tk.END).strip(), self.get_current_text, self.post_to_ui)
        try:
            extension = factory(parent, context)
        except Exception:
            parent.destroy()
            raise
        self._notebook.add(parent, text=title)
        self._extensions.append(extension)
        return extension

    def post_to_ui(self, callback):
        if not self._closing and not self._extensions_closed:
            self._update_q.put(('extension', callback))

    def close_extensions(self):
        if self._extensions_closed:
            return
        self._extensions_closed = True
        for extension in self._extensions:
            if callable(getattr(extension, 'close', None)):
                try:
                    extension.close()
                except Exception:
                    self.set_status('扩展关闭失败')
        self._extensions.clear()

    def _cancel_poll(self):
        if self._poll_timer is not None:
            try:
                self.root.after_cancel(self._poll_timer)
            except tk.TclError:
                pass
            self._poll_timer = None

    def _on_root_destroy(self, event):
        if event.widget is self.root:
            self._closing = True
            self._cancel_poll()

    def shutdown(self):
        self._closing = True
        self._cancel_poll()
        self.close_extensions()
        self._api_key_var.set('')

    def _on_apply_ai(self):
        if self.on_apply_ai:
            self.on_apply_ai()

    def begin_ai_request(self, request_id, simulation=False):
        self._active_ai_request = request_id
        self._answer_box.configure(state=tk.NORMAL)
        self._answer_box.delete('1.0', tk.END)
        self._answer_box.configure(state=tk.DISABLED)
        self._ai_metrics_var.set('AI：首字 — / 完成 —')
        self._ai_status_lbl.configure(text='本地模拟中，不联网' if simulation else '等待 DeepSeek 流式回答…')
        self._notebook.select(self._ai_tab)

    def invalidate_ai_request(self, message='回答已取消'):
        self._active_ai_request = None
        self._ai_status_lbl.configure(text=message)

    def update_ai_event(self, event):
        self._update_q.put(('ai', event))

    def update_asr_metrics(self, metrics):
        self._update_q.put(('asr_metrics', metrics))

    def _build_model_panel(self) -> None:
        mf = tk.LabelFrame(
            self.root, text="  模型参数与硬件  ",
            bg=C["base"], fg=C["mauve"],
            font=("Helvetica", 10, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        mf.pack(fill=tk.X, padx=14, pady=(0, 6))

        row1 = tk.Frame(mf, bg=C["base"])
        row1.pack(fill=tk.X, padx=10, pady=(8, 4))

        tk.Label(row1, text="计算设备:", bg=C["base"], fg=C["text"],
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        self._compute_device_var = tk.StringVar(value=_default_compute_device())
        self._reuse_resources_var = tk.BooleanVar(value=True)
        self._question_gate_var = tk.BooleanVar(value=True)
        self._streaming_var = tk.BooleanVar(value=False)
        ttk.Combobox(
            row1, textvariable=self._compute_device_var,
            values=["cpu", "cuda"],
            width=6, state="readonly",
        ).pack(side=tk.LEFT, padx=(6, 16))

        tk.Button(
            row1, text="↺  重新加载模型",
            bg=C["mauve"], fg=C["base"],
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, padx=16, pady=5,
            cursor="hand2",
            command=self._on_reload_model,
        ).pack(side=tk.LEFT)

        row2 = tk.Frame(mf, bg=C["base"])
        row2.pack(fill=tk.X, padx=10, pady=(0, 4))

        mode_row = tk.Frame(mf, bg=C['base'])
        mode_row.pack(fill=tk.X, padx=10, pady=4)
        tk.Checkbutton(mode_row, text='中文流式识别（重载生效）', variable=self._streaming_var,
                       bg=C['base'], fg=C['text'], selectcolor=C['surface0'],
                       activebackground=C['base'], activeforeground=C['text']).pack(side=tk.LEFT)
        tk.Checkbutton(mode_row, text='仅完整问题自动回答', variable=self._question_gate_var,
                       bg=C['base'], fg=C['text'], selectcolor=C['surface0'],
                       activebackground=C['base'], activeforeground=C['text']).pack(side=tk.LEFT, padx=10)
        self._partial_var = tk.StringVar(value='')
        tk.Label(mf, textvariable=self._partial_var, bg=C['base'], fg=C['text'],
                 anchor='w', wraplength=560).pack(fill=tk.X, padx=10)

        tk.Label(row2, text="静音阈值:", bg=C["base"], fg=C["text"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._vad_threshold_var = tk.DoubleVar(value=0.012)
        tk.Scale(
            row2, from_=0.001, to=0.05, resolution=0.001,
            orient=tk.HORIZONTAL, length=140,
            variable=self._vad_threshold_var,
            bg=C["base"], fg=C["text"], troughcolor=C["surface1"],
            highlightthickness=0,
            font=("Helvetica", 8),
        ).pack(side=tk.LEFT, padx=(4, 18))

        tk.Label(row2, text="静音时长(s):", bg=C["base"], fg=C["text"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._vad_silence_var = tk.DoubleVar(value=.6)
        tk.Scale(
            row2, from_=0.3, to=3.0, resolution=0.1,
            orient=tk.HORIZONTAL, length=140,
            variable=self._vad_silence_var,
            bg=C["base"], fg=C["text"], troughcolor=C["surface1"],
            highlightthickness=0,
            font=("Helvetica", 8),
        ).pack(side=tk.LEFT, padx=(4, 0))

        row3 = tk.Frame(mf, bg=C["base"])
        row3.pack(fill=tk.X, padx=10, pady=(0, 10))

        self._hw_info_lbl = tk.Label(
            row3, text="硬件信息检测中…",
            bg=C["base"], fg=C["overlay"],
            font=("Helvetica", 9),
        )
        self._hw_info_lbl.pack(side=tk.LEFT)
        tk.Checkbutton(row3, text='复用模型资源（重载生效）', variable=self._reuse_resources_var,
            bg=C['base'], fg=C['text'], selectcolor=C['surface0'],
            activebackground=C['base'], activeforeground=C['text']).pack(side=tk.RIGHT, padx=8)
        tk.Button(row3, text='低延迟停顿：0.6 秒', command=lambda: self._vad_silence_var.set(.6),
                  bg=C['surface1'], fg=C['text'], relief=tk.FLAT).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Public API (called from background threads via the update queue)
    # ------------------------------------------------------------------

    def set_devices(self, devices: List[Dict]) -> None:
        """Populate the device dropdown. Call from any thread."""
        self._update_q.put(("devices", devices))

    def set_status(self, text: str) -> None:
        self._update_q.put(("status", text))

    def set_listening(self, value: bool) -> None:
        self._update_q.put(("listening", value))

    def update_volume(self, rms: float) -> None:
        """rms: 0.0 – 1.0 raw RMS; will be scaled for display."""
        self._update_q.put(("volume", min(100.0, rms * 600)))

    def append_transcription(self, text: str, generation=None) -> None:
        """Append a new recognised segment to the current text box."""
        self._update_q.put(("append", (text, generation)))

    def get_selected_device_id(self) -> Optional[int]:
        return self._device_map.get(self._device_var.get())

    def get_language(self) -> str:
        return self._lang_var.get()

    def get_compute_device(self) -> str:
        return self._compute_device_var.get()

    def get_vad_threshold(self) -> float:
        return self._vad_threshold_var.get()

    def get_vad_silence(self) -> float:
        return self._vad_silence_var.get()

    def set_hw_info(self, text: str) -> None:
        self._update_q.put(("hw_info", text))

    # ------------------------------------------------------------------
    # Button handlers (always on main thread via Tkinter events)
    # ------------------------------------------------------------------

    def _on_reload_model(self) -> None:
        if self.on_reload_model:
            self.on_reload_model()

    def _toggle_listen(self) -> None:
        if not self._is_listening:
            if self.on_start:
                self.on_start()
        else:
            if self.on_stop:
                self.on_stop()

    def _delete_current(self) -> None:
        self._current_box.delete("1.0", tk.END)

    def _copy_to_history(self) -> None:
        text = self._current_box.get("1.0", tk.END).strip()
        if not text:
            self.set_status("⚠️ 当前没有可复制的内容")
            return

        ts = datetime.now().strftime("%H:%M:%S")

        self._history_box.configure(state=tk.NORMAL)
        # Separator between entries
        if self._history_box.get("1.0", tk.END).strip():
            self._history_box.insert(tk.END, "\n" + "─" * 55 + "\n", "sep")
        self._history_box.insert(tk.END, f"[{ts}]  ", "ts")
        self._history_box.insert(tk.END, text + "\n", "body")
        self._history_box.see(tk.END)
        self._history_box.configure(state=tk.DISABLED)

        # Also copy to system clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

        # Clear current box
        self._delete_current()
        self.set_status("✅ 已复制到历史记录（同时写入剪贴板）")

    def _clear_history(self) -> None:
        self._history_box.configure(state=tk.NORMAL)
        self._history_box.delete("1.0", tk.END)
        self._history_box.configure(state=tk.DISABLED)
        self.set_status("历史记录已清空")

    # ------------------------------------------------------------------
    # Update loop – processes messages from background threads safely
    # ------------------------------------------------------------------

    def _poll_updates(self) -> None:
        self._cancel_poll()
        if self._closing:
            return
        for _ in range(100):
            try:
                action, value = self._update_q.get_nowait()
                self._apply(action, value)
            except queue.Empty:
                break
            except tk.TclError:
                return
            except Exception:
                self._status_lbl.configure(text='界面事件处理失败，请检查配置')
        self._poll_timer = self.root.after(40, self._poll_updates)

    def _apply(self, action: str, value) -> None:
        if action == 'extension':
            if not self._closing and not self._extensions_closed:
                value()
        elif action == 'resume':
            generation, document, error = value
            self._resume_busy = False
            self._resume_import_btn.configure(state=tk.NORMAL)
            if generation != self._resume_generation:
                return
            if error:
                self._resume_status_var.set(error)
            else:
                self._resume_box.insert('1.0', document.text)
                self._resume_status_var.set(f'已在本地提取 {len(document.text)} 字符；请检查预览，上下文仍关闭')
        elif action == "devices":
            self._device_map = {}
            names: List[str] = []
            for dev in value:
                if dev.get("is_vbcable"):
                    label = f"[VB-CABLE] {dev['name']}"
                elif dev["is_monitor"]:
                    label = f"[系统音频] {dev['name']}"
                else:
                    label = dev["name"]
                names.append(label)
                self._device_map[label] = dev["id"]
            self._device_combo["values"] = names
            if names:
                # Priority: VB-CABLE > other system-audio monitors > first
                preferred = next(
                    (n for n in names if n.startswith("[VB-CABLE]")),
                    next(
                        (n for n in names if n.startswith("[系统音频]")),
                        names[0],
                    ),
                )
                self._device_combo.set(preferred)

        elif action == "status":
            self._status_lbl.configure(text=value)

        elif action == "volume":
            self._vol_bar["value"] = value

        elif action == "listening":
            self._is_listening = value
            if value:
                self._listen_btn.configure(
                    text="⏹  停止监听", bg=C["red"]
                )
                self._status_lbl.configure(text="🎙 监听中…", fg=C["green"])
            else:
                self._listen_btn.configure(
                    text="▶  开始监听", bg=C["green"]
                )
                self._status_lbl.configure(text="已停止", fg=C["yellow"])

        elif action == "append":
            text, generation = value if isinstance(value, tuple) else (value, None)
            if self.on_transcript_received and not self.on_transcript_received(text, generation):
                return
            self._current_box.insert(tk.END, text + "\n")
            self._current_box.see(tk.END)

        elif action == 'partial':
            text, generation = value
            if self.on_partial_received and not self.on_partial_received(generation):
                return
            self._partial_var.set(('识别中：' + text[-180:]) if text else '')

        elif action == 'segment_discarded':
            self._partial_var.set('')
            if self.on_segment_discarded:
                self.on_segment_discarded()

        elif action == 'asr_metrics':
            self._asr_metrics_var.set(f"识别：排队 {value['queue_ms']:.0f} ms / 推理 {value['inference_ms']:.0f} ms")

        elif action == 'ai':
            if value['request_id'] != self._active_ai_request:
                return
            if value['kind'] == 'delta':
                if not self._answer_box.get('1.0', tk.END).strip() and 'elapsed_ms' in value:
                    self._ai_metrics_var.set(f"AI：首字 {value['elapsed_ms']:.0f} ms / 接收中")
                self._answer_box.configure(state=tk.NORMAL)
                self._answer_box.insert(tk.END, value['text'])
                self._answer_box.see(tk.END)
                self._answer_box.configure(state=tk.DISABLED)
                self._ai_status_lbl.configure(text='正在接收回答…')
            elif value['kind'] == 'done':
                metrics = value['metrics']
                self._ai_metrics_var.set(f"AI：首字 {metrics['first_token_ms']:.0f} ms / 完成 {metrics['total_ms']:.0f} ms")
                self._ai_status_lbl.configure(text='已到输出上限，回答可能不完整' if metrics.get('truncated') else '回答完成')
            elif value['kind'] == 'started':
                self._ai_status_lbl.configure(text='请求已开始，等待首字…')
            elif value['kind'] == 'error':
                self._ai_status_lbl.configure(text=value['text'])

        elif action == "hw_info":
            self._hw_info_lbl.configure(text=value)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()
