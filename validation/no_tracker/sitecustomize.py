import importlib.abc
import sys


class TrackerUnavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'wandb':
            raise ModuleNotFoundError('External tracking package is unavailable for this verification')


sys.meta_path.insert(0, TrackerUnavailable())
