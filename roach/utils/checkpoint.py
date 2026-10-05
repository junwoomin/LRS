"""Restricted loading of tensors and the one supplied historical Roach checkpoint."""
import hashlib
from pathlib import Path
import codecs
import numpy as np
import torch


_LEGACY_PATH = Path(__file__).resolve().parents[1] / 'log' / 'ckpt_11833344.pth'
_LEGACY_SHA256 = '1fecec0c8a206a9ac07a6a9b0be77a5b7e95a073b20d9dfc2a7397354355094a'


def load_checkpoint(path, map_location='cpu'):
    path = Path(path)
    if path.resolve() == _LEGACY_PATH and hashlib.sha256(path.read_bytes()).hexdigest() == _LEGACY_SHA256:
        # These types were verified from this exact checkpoint's pickle bytecode.
        # Keep the allowlist scoped to this load; never fall back to unrestricted pickle.
        import gym
        from numpy.core.multiarray import _reconstruct
        from numpy.random._pickle import __randomstate_ctor
        allowed = [gym.spaces.Dict, gym.spaces.Box, np.dtype, np.ndarray,
                   _reconstruct, codecs.encode, __randomstate_ctor, np.random.RandomState]
        allowed += [type(np.dtype(t)) for t in ('float32', 'float64', 'uint32', 'uint8', 'int64', 'int32', 'bool')]
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location, weights_only=True)
