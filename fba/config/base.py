import pathlib
import tempfile
import shutil
import sys
import json
from importlib import import_module
from addict import Dict
from fba import logger


def recursive_print_dict( d, indent = 0 ):
    result = ""
    for k, v in d.items():
        if isinstance(v, dict):
            result += "   " * indent + f"{k}\n"
            result += recursive_print_dict(v, indent+1)
            recursive_print_dict(v, indent+1)
        else:
            result += "   " * indent + f"{k}: {v}\n"
    return result


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            n = self.__class__.__name__
            ex = AttributeError(f"'{n}' object has no attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def recursive_update_B_into_A(A: dict, B: dict):
    for key, item in B.items():
        if key in A.keys():
            assert type(A[key] == B[key]),\
                f"Trying to overwrite a value with non-matchin data type. Key={key}"
            if isinstance(item, dict):
                recursive_update_B_into_A(A[key], B[key])
        else:
            A[key] = B[key]


class Config(object):
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    """

    @staticmethod
    def _py2dict(filepath: pathlib.Path):

        assert filepath.is_file(), filepath
        with tempfile.TemporaryDirectory() as temp_config_dir:
            shutil.copyfile(filepath, pathlib.Path(temp_config_dir, "_tempconfig.py"))
            sys.path.insert(0, temp_config_dir)
            mod = import_module("_tempconfig")
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__") and name not in ["os", "math"]
            }
            # delete imported module
            del sys.modules["_tempconfig"]

        cfg_text = f"{filepath}\n"
        with open(filepath, "r") as f:
            cfg_text += f.read()
        if "_base_config_" in cfg_dict:
            cfg_dir = filepath.parent
            base_filename = cfg_dict.pop("_base_config_")
            base_filename = (
                base_filename if isinstance(base_filename, list) else [base_filename]
            )

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(cfg_dir.joinpath(f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                overlapping_keys = base_cfg_dict.keys() & c.keys()
                if len(overlapping_keys) > 0:
                    logger.info(f"[WARN] base config has overlapping keys: {overlapping_keys}")
                recursive_update_B_into_A(base_cfg_dict, c)

            Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)
        return cfg_dict, cfg_text

    @staticmethod
    def _json2dict(filepath: pathlib.Path):
        with open(filepath, "r") as fp:
            return json.load(fp)

    @staticmethod
    def _file2dict(filepath):
        filepath = pathlib.Path(filepath)
        if filepath.suffix == ".py":
            return Config._py2dict(filepath)
        if filepath.suffix == ".json":
            return Config._json2dict(filepath), None
        raise ValueError("Expected json or python file:", filepath)

    @staticmethod
    def _merge_a_into_b(a, b):
        # merge dict a into dict b. values in a will overwrite b.
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                if b[k] is None:
                    b[k] = v
                    continue
                if not isinstance(b[k], dict):
                    raise TypeError(f"Cannot inherit key {k} from base!")
                Config._merge_a_into_b(v, b[k])
            else:
                if k in b:
                    if b[k] is not None and b[k] != v:
                        print("Overwriting", k, "Old:", b[k], "New:", v)
                b[k] = v

    @staticmethod
    def fromfile(filepath):
        cfg_dict, cfg_text = Config._file2dict(filepath)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filepath)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f"cfg_dict must be a dict, but got {type(cfg_dict)}")

        super(Config, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(Config, self).__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, "r") as f:
                text = f.read()
        else:
            text = ""
        super(Config, self).__setattr__("_text", text)

    @property
    def filename(self):
        return self._filename

    def model_name(self, start=0):
        return "_".join(pathlib.Path(self._filename).parts[start:])[:-3]

    @property
    def output_dir(self):
        parts = pathlib.Path(self.filename).parts
        parts = [pathlib.Path(p).stem for p in parts if "configs" not in p]
        output_dir = pathlib.Path(self._output_dir, *parts)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @property
    def checkpoint_dir(self):
        return self.output_dir.joinpath(self._checkpoint_dir)

    @property
    def cache_dir(self):
        return pathlib.Path(self._cache_dir)

    @property
    def commit(self):
        assert self._commit is not None, "Commit not set."
        return self._commit

    def __repr__(self):
        cfg_dict = super(Config, self).__getattribute__("_cfg_dict")
        return recursive_print_dict(cfg_dict)

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def dump(self):
        filepath = self.output_dir.joinpath("config_dump.json")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as fp:
            json.dump(dict(self), fp, indent=4)

    def merge_from_dict(self, options):
        """Merge list into cfg_dict

        Merge the dict parsed by MultipleKVAction into this cfg.
        Example,
            >>> options = {'model.backbone.depth': 50}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)

        Args:
            options (dict): dict of configs to merge from.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d[subkey] = ConfigDict()
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__("_cfg_dict")
        Config._merge_a_into_b(option_cfg_dict, cfg_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    cfg = Config.fromfile(parser.parse_args().filepath)
    print("Ouput directory", cfg.output_dir)
    cfg.dump()
