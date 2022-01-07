import inspect
import functools
import copy
from fba.utils import Registry, build_from_cfg


CRITERION_REGISTRY = Registry("CRITERIONS")


def build_criterion(type, cfg, **kwargs):
    if cfg["weight"] == 0 or type is None:
        return None
    criterion_ = CRITERION_REGISTRY.get(type)
    if inspect.isclass(criterion_):
        return build_from_cfg(cfg, CRITERION_REGISTRY, **kwargs)
    cfg = copy.deepcopy(cfg)
    cfg.pop("type")
    return functools.partial(CRITERION_REGISTRY.get(type), **cfg, **kwargs)
