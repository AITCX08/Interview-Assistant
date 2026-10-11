"""Bounded local document extraction. No network, persistence or document execution."""
from dataclasses import dataclass, field
from pathlib import Path
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

MAX_CONTEXT = 12000
class ResumeError(Exception):
    pass

@dataclass(frozen=True)
class ResumeDocument:
    text: str = field(repr=False)
    pages: int = 0

def redact_contacts(text):
    text = re.sub(r'[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[邮箱已移除]', text)
    text = re.sub(r'(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)', '[电话已移除]', text)
    text = re.sub(r'(?<!\d)\d{17}[\dXx](?!\w)', '[证件已移除]', text)
    text = re.sub(r'(?m)^\s*(?:姓名|Name)\s*[:：].*$', '[姓名已移除]', text, flags=re.I)
    text = re.sub(r'https?://\S+', '[链接已移除]', text)
    return text.strip()

def extract_resume(path):
    try:
        path = Path(path)
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ResumeError('文档须小于 10 MB')
        if path.suffix.lower() == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ResumeError('请先在本机解除 PDF 密码保护')
            if len(reader.pages) > 20:
                raise ResumeError('文档最多支持 20 页')
            parts=[]
            for page in reader.pages:
                parts.append(page.extract_text() or '')
                if sum(map(len, parts)) > 100000:
                    raise ResumeError('文档文本过长')
            text='\n'.join(parts)
            pages=len(reader.pages)
        elif path.suffix.lower() == '.docx':
            with ZipFile(path) as archive:
                info=archive.getinfo('word/document.xml')
                if info.file_size > 5 * 1024 * 1024:
                    raise ResumeError('文档内容过大')
                data=archive.read(info)
            if b'<!DOCTYPE' in data or b'<!ENTITY' in data:
                raise ResumeError('文档包含不支持的 XML 内容')
            root=ET.fromstring(data)
            ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            text='\n'.join(''.join(p.itertext()) for p in root.iter(ns+'p'))
            pages=0
        else:
            raise ResumeError('仅支持 PDF 和 DOCX')
        if not text.strip():
            raise ResumeError('未提取到文本；扫描件需要先进行本地 OCR')
        text=redact_contacts(text)
        if len(text) > MAX_CONTEXT:
            raise ResumeError('文本超过 12000 字符，请精简文档后导入')
        return ResumeDocument(text, pages)
    except ResumeError:
        raise
    except Exception as exc:
        raise ResumeError('文档解析失败，请检查格式及本地 pypdf 依赖') from exc
