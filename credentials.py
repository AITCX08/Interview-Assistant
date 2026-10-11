"""Per-user Windows DPAPI credential storage; never plaintext."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path

class CredentialError(Exception):
    pass

class Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]

class KeyStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'InterviewAssistant' / 'deepseek-key.dpapi'

    @staticmethod
    def _crypt(data, encrypt):
        if os.name != 'nt':
            raise CredentialError('加密记住 Key 仅支持 Windows')
        buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        source, target = Blob(len(data), buf), Blob()
        dll = ctypes.WinDLL('crypt32', use_last_error=True)
        if encrypt:
            ok = dll.CryptProtectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target))
        else:
            ok = dll.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target))
        if not ok:
            raise CredentialError('本机 Key 加密或解密失败，请重新填写')
        try:
            return ctypes.string_at(target.data, target.size)
        finally:
            kernel = ctypes.WinDLL('kernel32')
            kernel.LocalFree.argtypes = [ctypes.c_void_p]
            kernel.LocalFree(target.data)

    def load(self):
        if not self.path.exists():
            return ''
        try:
            if self.path.stat().st_size > 16384:
                raise ValueError()
            return self._crypt(self.path.read_bytes(), False).decode('utf-8')
        except (OSError, ValueError, UnicodeError) as exc:
            raise CredentialError('本机 Key 读取失败，请重新填写') from exc

    def save(self, key):
        if not key.strip() or len(key) > 4096 or '\n' in key or '\r' in key:
            raise CredentialError('Key 格式不正确')
        encrypted = self._crypt(key.strip().encode('utf-8'), True)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_bytes(encrypted)
            os.replace(temporary, self.path)
        except OSError as exc:
            raise CredentialError('本机 Key 保存失败') from exc
