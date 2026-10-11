"""Avoid repeatedly copying read-only FunASR 1.3.1 resources.

Audited offline Paraformer/FSMN/CT-punc path: frontend cache is passed in a
fresh runtime dictionary; frontend CMVN and tokenizer tables are read-only.
Unknown objects retain deepcopy semantics. No upstream files are modified.
"""
import copy
import types

AUDITED_VERSION = '1.3.1'
READONLY_RESOURCES = {
    ('funasr.frontends.wav_frontend', 'WavFrontend'),
    ('funasr.frontends.wav_frontend', 'WavFrontendOnline'),
    ('funasr.tokenizer.char_tokenizer', 'CharTokenizer'),
}


def clone_configuration(value, memo=None):
    """Isolate mutable containers; retain only explicitly audited resources."""
    memo = {} if memo is None else memo
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if type(value) in (str, int, float, bool, bytes, type(None)):
        return value
    if (type(value).__module__, type(value).__name__) in READONLY_RESOURCES:
        memo[identity] = value
        return value
    if type(value) is dict:
        result = {}; memo[identity] = result
        for key, item in value.items():
            result[clone_configuration(key, memo)] = clone_configuration(item, memo)
        return result
    if type(value) is list:
        result = []; memo[identity] = result
        result.extend(clone_configuration(item, memo) for item in value)
        return result
    if type(value) is tuple:
        # Preserve ordinary tuple configuration values.
        result = tuple(clone_configuration(item, memo) for item in value)
        memo[identity] = result
        return result
    return copy.deepcopy(value, memo)


def enable_resource_reuse(model, version):
    """Version and capability guard: other releases use the original reset."""
    baseline = getattr(model, '_base_kwargs_map', None)
    if (version != AUDITED_VERSION or not isinstance(baseline, dict)
            or not isinstance(baseline.get('kwargs'), dict)
            or not callable(getattr(model, '_reset_runtime_configs', None))):
        return False

    def reset(this):
        import torch
        # One memo per reset preserves shared resources but never prior caches.
        configs = clone_configuration(this._base_kwargs_map)
        for name, config in configs.items():
            setattr(this, name, config)
        ncpu = int(this.kwargs.get('ncpu', 4))
        if ncpu < 1:
            ncpu = 4
        if torch.get_num_threads() != ncpu:
            torch.set_num_threads(ncpu)

    model._reset_runtime_configs = types.MethodType(reset, model)
    return True
