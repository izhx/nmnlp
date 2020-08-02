from typing import Dict, Any, Union
from argparse import Namespace

import yaml


def dict_to_namespace(dic: Dict) -> Namespace:
    if isinstance(dic, dict):
        for k, v in dic.items():
            dic[k] = dict_to_namespace(v)
        return Namespace(**dic)
    elif isinstance(dic, list):
        return [dict_to_namespace(i) for i in dic]
    return dic


def namespace_to_dict(obj: Namespace) -> Dict:
    if isinstance(obj, Namespace):
        obj = vars(obj)

    if isinstance(obj, dict):
        obj = {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj = [namespace_to_dict(i) for i in obj]
    return obj


def load_yaml(path: str) -> Dict[str, Any]:
    if path.endswith(('yml', 'yaml')):
        with open(path) as file:
            config = yaml.load(file)
            config['path'] = path
            return config
    else:
        raise ValueError(
            f"The config file at {path} has a wrong type!")


def save_yaml(obj: Union[Dict, Namespace], path: str = None) -> None:
    if isinstance(obj, Namespace):
        obj = namespace_to_dict(obj)
    if path is None:
        try:
            path = obj.pop('path')
        except KeyError:
            raise KeyError("存储的配置文件损坏!")
    with open(path, mode='w+') as file:
        yaml.dump(obj, file)
