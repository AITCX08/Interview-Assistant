import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import patch
from credentials import KeyStore
from resume_context import extract_resume, redact_contacts, ResumeError
from ai_client import DeepSeekSettings, DeepSeekClient

class ResumeTests(unittest.TestCase):
    def test_contact_redaction(self):
        result = redact_contacts('姓名：测试候选人\n电话 13812345678 邮箱 sample@example.com\n项目：检索系统')
        self.assertNotIn('13812345678', result)
        self.assertNotIn('sample@example.com', result)
        self.assertNotIn('测试候选人', result)
        self.assertIn('检索系统', result)

    def test_docx_and_bad_input(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'synthetic.docx'
            with ZipFile(p, 'w') as z:
                z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>技能 Python</w:t></w:r></w:p></w:document>')
            self.assertIn('Python', extract_resume(p).text)
            p.write_bytes(b'bad')
            with self.assertRaises(ResumeError): extract_resume(p)

    def test_key_roundtrip_and_ciphertext(self):
        with tempfile.TemporaryDirectory() as folder:
            store = KeyStore(Path(folder)/'key.dpapi')
            self.assertEqual(store.load(), '')
            store.save('synthetic-secret')
            self.assertEqual(store.load(), 'synthetic-secret')
            self.assertNotIn(b'synthetic-secret', store.path.read_bytes())
            store.path.write_bytes(b'broken')
            with self.assertRaises(Exception): store.load()

    def test_context_payload_mock_only(self):
        captured=[]
        def opener(request, **kwargs):
            captured.append(json.loads(request.data))
            return io.BytesIO(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
        settings=DeepSeekSettings(api_key='synthetic', resume_context='Synthetic Python project')
        DeepSeekClient(opener=opener).stream(settings, 'question', threading.Event(), lambda _: None)
        self.assertIn('Synthetic Python project', captured[0]['messages'][1]['content'])
        self.assertEqual(captured[0]['messages'][-1]['content'], 'question')
        self.assertNotIn('Synthetic Python project', repr(settings))
