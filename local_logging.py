"""Local JSON/JSONL run records; no account or network service is needed."""
import json
import math
import os
from datetime import datetime
from pathlib import Path
import numpy as np


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class LocalRun:
    def __init__(self, directory, config=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if config is not None:
            with (self.directory / 'config.json').open('w') as stream:
                json.dump(_json_value(config), stream, ensure_ascii=False, indent=2)
        self.metrics_path = self.directory / 'metrics.jsonl'

    def log(self, metrics, step=None):
        record = dict(metrics)
        if step is not None:
            record['global_step'] = int(step)
        with self.metrics_path.open('a') as stream:
            stream.write(json.dumps(_json_value(record), ensure_ascii=False, allow_nan=False) + '\n')


def run_directory(town):
    configured = os.environ.get('OUTPUT_DIR')
    if configured:
        return Path(configured)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    return Path('roach_run') / town / stamp
