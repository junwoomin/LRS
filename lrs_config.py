"""Local checkpoint configuration for the LRS prototype."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_CHECKPOINT = PROJECT_ROOT / "roach" / "checkpoints" / "ckpt_11833344.pth"


def policy_checkpoint_path(path=None):
    """Resolve an explicit checkpoint, LRS_CHECKPOINT, or the local default."""
    value = path if path is not None else os.environ.get("LRS_CHECKPOINT")
    checkpoint = Path(value).expanduser() if value else DEFAULT_POLICY_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Policy checkpoint not found: {checkpoint}. "
            "Place a compatible ROACH policy checkpoint at "
            f"{DEFAULT_POLICY_CHECKPOINT}, or set LRS_CHECKPOINT to its path."
        )
    return str(checkpoint)
