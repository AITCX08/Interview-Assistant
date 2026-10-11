import copy
import types
import unittest
from unittest.mock import patch
from funasr_runtime import clone_configuration, enable_resource_reuse

class RuntimeTests(unittest.TestCase):
    def test_readonly_frontend_reused_mutable_cache_isolated(self):
        frontend_type=type('WavFrontend', (), {'__module__':'funasr.frontends.wav_frontend', '__deepcopy__':lambda *_: (_ for _ in ()).throw(AssertionError('resource copied'))})
        frontend=frontend_type()
        base={'frontend':frontend,'cache':{'frames':[1,2]},'options':[{'threshold':.5}]}
        first=clone_configuration(base);second=clone_configuration(base)
        self.assertIs(first['frontend'],frontend)
        first['cache']['frames'].append(3)
        first['options'][0]['threshold']=9
        self.assertEqual(second['cache']['frames'],[1,2])
        self.assertEqual(base['options'][0]['threshold'],.5)

    def test_unknown_mutable_objects_are_copied(self):
        class Mutable:
            def __init__(self):self.items=[]
        original=Mutable();cloned=clone_configuration({'unknown':original})
        cloned['unknown'].items.append('x')
        self.assertEqual(original.items,[])

    def test_guard_and_per_request_reset(self):
        model=types.SimpleNamespace(_base_kwargs_map={'kwargs':{'ncpu':4,'cache':{'x':[]}}},_reset_runtime_configs=lambda:None)
        self.assertFalse(enable_resource_reuse(model,'future-version'))
        fake_torch=types.SimpleNamespace(get_num_threads=lambda:4,set_num_threads=lambda _:self.fail('unnecessary setting'))
        with patch.dict('sys.modules',{'torch':fake_torch}):
            self.assertTrue(enable_resource_reuse(model,'1.3.1'))
            model._reset_runtime_configs()
            model.kwargs['cache']['x'].append(2)
            model._reset_runtime_configs()
        self.assertEqual(model.kwargs['cache']['x'],[])
        self.assertEqual(model._base_kwargs_map['kwargs']['cache']['x'],[])

    def test_unsupported_model_stays_unchanged(self):
        original=lambda:None
        model=types.SimpleNamespace(_reset_runtime_configs=original)
        self.assertFalse(enable_resource_reuse(model,'1.3.1'))
        self.assertIs(model._reset_runtime_configs,original)

    def test_real_tokenizer_namespace_is_reused(self):
        tokenizer_type=type('CharTokenizer', (), {'__module__':'funasr.tokenizer.char_tokenizer', '__deepcopy__':lambda *_: (_ for _ in ()).throw(AssertionError('token table copied'))})
        tokenizer=tokenizer_type()
        self.assertIs(clone_configuration({'tokenizer':tokenizer})['tokenizer'],tokenizer)
