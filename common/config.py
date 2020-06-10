from typing import Dict, Any

import yaml

from .checks import ConfigurationError


class Config(object):
    """
    Settings and hyper parameters.
    """

    def __init__(self, cfg: Dict[str, Dict], path: str, debug: bool = False):
        self.cfg = cfg
        self.path = path
        self.debug = debug
        return

    def __getitem__(self, key: str):
        if key in self.cfg:
            return self.cfg[key]
        else:
            raise KeyError("{} not found".format(key))

    def __setitem__(self, key, value):
        self.cfg[key] = value

    def __contains__(self, item):
        return item in self.cfg

    def get(self, key: str, default: Any = None):
        for k in self.cfg:
            if key in self.cfg[k]:
                return self.cfg[k][key]
        return default

    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        if file_path.endswith(('yml', 'yaml')):
            with open(file_path) as file:
                cfg = yaml.load(file, Loader=yaml.FullLoader)
                return cls(cfg, file_path)
        else:
            raise ConfigurationError(
                f"The config file at {file_path} has a wrong type!")

    @classmethod
    def from_args(cls, args):
        cfg = cls.from_file(args.yaml)
        cfg.debug = args.debug
        return cfg

    def to_file(self, file_path: str) -> None:
        with open(file_path, mode='w+') as file:
            yaml.dump(self.cfg, file)

    def save(self) -> None:
        """ save inplace. """
        self.to_file(self.path)

    def reload(self):
        return self.from_file(self.path)
