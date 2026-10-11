"""Optional encrypted user data and explicit local text export, available to all editions."""
import json
import os
from pathlib import Path
import tempfile
from tkinter import filedialog
from credentials import KeyStore, CredentialError

MAX_BYTES = 512 * 1024


class LocalDataError(Exception):
    pass


class EncryptedLocalStore:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, value):
        temporary = None
        try:
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            if len(data) > MAX_BYTES:
                raise LocalDataError('本机资料超过 512 KiB，请减少保存内容')
            data = KeyStore._crypt(data, True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix='.pending-', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            os.replace(temporary, self.path)
        except (OSError, ValueError, TypeError, CredentialError):
            raise LocalDataError('本机资料加密保存失败，未改为明文保存') from None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load(self):
        if not self.path.exists():
            return None
        try:
            if self.path.stat().st_size > MAX_BYTES + 8192:
                raise ValueError()
            data = KeyStore._crypt(self.path.read_bytes(), False)
            if len(data) > MAX_BYTES:
                raise ValueError()
            return json.loads(data.decode('utf-8'))
        except (OSError, ValueError, TypeError, UnicodeError, CredentialError):
            raise LocalDataError('本机资料读取失败；原文件已保留') from None

    def clear(self):
        """Only the caller's explicit delete action should invoke this."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            raise LocalDataError('本机资料删除失败') from None


def export_text(parent, text, *, markdown=False):
    extension = '.md' if markdown else '.txt'
    path = filedialog.asksaveasfilename(parent=parent, defaultextension=extension,
        initialfile='interview-notes' + extension,
        filetypes=[('文字资料', '*' + extension)], title='导出本地文字（包含当前内容）')
    if not path:
        return False
    try:
        Path(path).write_text(text, encoding='utf-8')
        return True
    except OSError:
        raise LocalDataError('文字导出失败，请选择可写位置') from None
