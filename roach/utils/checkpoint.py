"""Restricted loading of tensors and the one supplied historical Roach checkpoint."""
import hashlib
from pathlib import Path
import codecs
import numpy as np
import torch
from roach.utils.asset_manifest import ORIGINAL_CHECKPOINT_PATH, ORIGINAL_CHECKPOINT_SHA256


_LEGACY_PATH = Path(__file__).resolve().parents[2] / ORIGINAL_CHECKPOINT_PATH
_LEGACY_SHA256 = ORIGINAL_CHECKPOINT_SHA256


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
