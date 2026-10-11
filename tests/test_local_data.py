import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from local_data import EncryptedLocalStore, LocalDataError, export_text


class LocalDataTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'DPAPI is Windows only')
    def test_roundtrip_is_encrypted_and_clear_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.dpapi'
            store = EncryptedLocalStore(path)
            self.assertIsNone(store.load())
            store.save({'text': 'synthetic-private-data'})
            self.assertNotIn(b'synthetic-private-data', path.read_bytes())
            self.assertEqual(store.load(), {'text': 'synthetic-private-data'})
            store.clear()
            self.assertFalse(path.exists())

    @unittest.skipUnless(os.name == 'nt', 'DPAPI is Windows only')
    def test_corrupt_and_oversize_data_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.dpapi'
            store = EncryptedLocalStore(path)
            with self.assertRaises(LocalDataError):
                store.save({'text': 'x' * (512 * 1024)})
            self.assertFalse(path.exists())
            path.write_bytes(b'corrupt')
            with self.assertRaises(LocalDataError):
                store.load()

    def test_export_requires_selected_path(self):
        with patch('local_data.filedialog.asksaveasfilename', return_value=''):
            self.assertFalse(export_text(None, 'synthetic'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.txt'
            with patch('local_data.filedialog.asksaveasfilename', return_value=str(path)):
                self.assertTrue(export_text(None, 'synthetic'))
            self.assertEqual(path.read_text(encoding='utf-8'), 'synthetic')
