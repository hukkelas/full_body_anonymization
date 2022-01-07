import copy
from .loss_handler import LossHandler


def build_losss_fnc(loss_cfg, **kwargs):
    cfg = copy.deepcopy(loss_cfg)
    cfg.pop("type")
    return LossHandler(**cfg, **kwargs)
