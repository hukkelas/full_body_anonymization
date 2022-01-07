import inspect
from functools import partial
from fba import logger


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        name = self.__class__.__name__
        format_str = f"{name} (name={self._name}, items={self._module_dict.keys()}"
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        obj = self._module_dict.get(key, None)
        if obj is None:
            raise KeyError(f"{key} is not in the {self._name} registry.")
        return obj

    def _register_module(self, module_class, force=False):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not isinstance(module_class, type) and not inspect.isfunction(module_class):
            raise TypeError(f"module must be a class, but got {type(module_class)}")
        module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self.name}")
        self._module_dict[module_name] = module_class

    def register_module(self, cls=None, force=False):
        if cls is None:
            return partial(self.register_module, force=force)
        self._register_module(cls, force=force)
        return cls


def build_from_cfg(_cfg, registry, **kwargs):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(_cfg, dict) and "type" in _cfg, _cfg
    args = _cfg.copy()
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif isinstance(obj_type, type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {obj_type}")
    try:
        return obj_cls(**args, **kwargs)
    except TypeError as e:
        logger.warn("Could not build: ", str(_cfg))
        raise e
