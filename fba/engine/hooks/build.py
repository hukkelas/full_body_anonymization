from fba.utils import Registry, build_from_cfg

HOOK_REGISTRY = Registry("HOOKS")


def build_hooks(cfg, trainer):
    hooks = cfg.hooks
    for _hook in hooks.values():
        if _hook is None:
            continue
        hook = build_from_cfg(
            _hook, HOOK_REGISTRY, semantic_labels=cfg.semantic_labels, fid_real_directory=cfg.fid_real_directory)
        trainer.register_hook(_hook.type, hook)


class HookBase:

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_extend(self):
        """
            Will be called after we increase resolution / model size
        """
        pass

    def before_extend(self):
        """
            Will be called before we increase resolution / model size
        """
        pass

    def load_state_dict(self, state_dict: dict):
        pass

    def state_dict(self):
        return None
