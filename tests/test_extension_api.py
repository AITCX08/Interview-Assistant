import os
import threading
import unittest
from unittest.mock import Mock, patch
from gui import InterviewAssistantGUI
from extension_api import CORE_API_VERSION


@unittest.skipIf(os.name != 'nt' and not os.environ.get('DISPLAY'), 'GUI requires display')
class ExtensionTests(unittest.TestCase):
    def setUp(self):
        with patch('gui._default_compute_device', return_value='cpu'):
            self.gui = InterviewAssistantGUI(key_store=Mock(load=Mock(return_value='')))
        self.gui.root.withdraw()
        self.addCleanup(self.gui.root.destroy)

    def test_explicit_mount_and_queued_ui_callback(self):
        calls = []
        extension = Mock()
        def factory(parent, context):
            self.assertEqual(context.api_version, CORE_API_VERSION)
            self.context = context
            return extension
        self.assertEqual(len(self.gui._notebook.tabs()), 3)
        self.assertIs(self.gui.mount_extension('synthetic extension', factory), extension)
        self.assertEqual(len(self.gui._notebook.tabs()), 4)
        main_thread = threading.get_ident()
        thread = threading.Thread(target=lambda: self.context.post(lambda: calls.append(threading.get_ident())))
        thread.start()
        thread.join(1)
        self.assertEqual(calls, [])
        self.gui._poll_updates()
        self.assertEqual(calls, [main_thread])
        self.gui.close_extensions()
        self.gui.close_extensions()
        extension.close.assert_called_once()
        self.context.post(lambda: calls.append('stale'))
        self.gui._apply('extension', lambda: calls.append('stale'))
        self.assertNotIn('stale', calls)

    def test_context_resume_is_local_read_without_send_toggle(self):
        contexts = []
        self.gui.mount_extension('synthetic', lambda parent, context: contexts.append(context))
        self.gui._resume_box.insert('1.0', 'synthetic resume')
        self.assertFalse(self.gui._use_resume_var.get())
        self.assertEqual(contexts[0].get_resume_text(), 'synthetic resume')
        self.assertEqual(contexts[0].get_settings(False).resume_context, '')
