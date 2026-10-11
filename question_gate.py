"""Local bounded Chinese interview-question routing; no network or persistence."""
import re
import time


class QuestionGate:
    MAX_CHARS = 1200
    WAIT_SECONDS = 2.0
    DEDUP_SECONDS = 8.0
    _question = re.compile(r'什么|怎么|如何|为什么|为何|是否|能否|能不能|有没有|哪些|多少|区别|解释|介绍|描述|谈谈|说说|讲讲|说一下|讲一下|谈一下|举例|分析|设计|实现|比较|请说明|请问|问一下|[？?]|(?:吗|么|呢)[。！!\s]*$|\b(?:what|how|why|explain|describe|tell me)\b', re.I)
    _unfinished = re.compile(r'(?:以及|然后|如果|因为|比如|例如|首先|关于|但是|并且|还有|和|与|或者|分别|如下|尤其是|包括|：|:)[，,。\s]*$')
    _background = re.compile(r'先说.*背景|先介绍.*背景|(?:假设|假如|场景是|有这样一个场景)')
    _ack = re.compile(r'^(?:好|好的|好吧|嗯|嗯嗯|哦|明白|明白了|了解|了解了|谢谢|谢谢你|对|对的|是的|可以|没问题|ok|okay|thanks)[，,。.!！\s]*$', re.I)

    def __init__(self):
        self.reset()

    def reset(self):
        self.pending = ''
        self._since = None
        self._last = ''
        self._last_at = -float('inf')

    def _emit(self, text, now):
        self.pending, self._since = '', None
        normalized = re.sub(r'[\W_]+', '', text).lower()
        if normalized == self._last and now - self._last_at < self.DEDUP_SECONDS:
            return None
        self._last, self._last_at = normalized, now
        return text

    def _is_background(self, text):
        return bool(self._background.search(text) and not re.search(r'什么|怎么|如何|为什么|是否|[？?]', text))

    def push(self, text, now=None):
        now = time.monotonic() if now is None else now
        text = text.strip()[:self.MAX_CHARS]
        if not text or self._ack.fullmatch(text):
            return None
        # Never carry old context into a later, unrelated question.
        if self._since is not None and now - self._since >= self.WAIT_SECONDS:
            self.pending, self._since = '', None
        combined = (self.pending + text)[:self.MAX_CHARS]
        if self._unfinished.search(combined) or self._is_background(text):
            self.pending = combined
            if self._since is None:
                self._since = now
            return None
        if self._question.search(combined):
            return self._emit(combined, now)
        self.pending, self._since = '', None
        return None

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        if self._since is None or now - self._since < self.WAIT_SECONDS:
            return None
        text = self.pending
        self.pending, self._since = '', None
        return self._emit(text, now) if self._question.search(text) and not self._is_background(text) else None
