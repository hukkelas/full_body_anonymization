import pathlib
import weakref
import collections
from .hooks import HookBase, build_hooks
from . import checkpointer
from fba import logger, utils


class BaseTrainer:

    def __init__(self, checkpoint_dir: pathlib.Path, cfg, batch_size: int, max_images_to_train: int, **kwargs):
        self.hooks: collections.OrderedDict[str, HookBase] = {}
        self.sigterm_received = False
        self.checkpoint_dir = checkpoint_dir
        self._best_model_step = 0
        self._batch_size = batch_size
        self._max_images_to_train = max_images_to_train
        build_hooks(cfg, self)

    def register_hook(self, key: str, hook: HookBase):
        assert key not in self.hooks
        self.hooks[key] = hook
        # To avoid circular reference, hooks and trainer cannot own each other.
        # This normally does not matter, but will cause memory leak if the
        # involved objects contain __del__:
        # See
        # http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
        assert isinstance(hook, HookBase)
        hook.trainer = weakref.proxy(self)

    def before_train(self):
        for hook in self.hooks.values():
            hook.before_train()

    def before_step(self):
        for hook in self.hooks.values():
            hook.before_step()

    def after_step(self):
        for hook in self.hooks.values():
            hook.after_step()
        logger.update_global_step(logger.global_step + self._batch_size)

    def state_dict(self) -> dict:
        state_dict = {
            "best_model_step": self._best_model_step
        }
        for key, hook in self.hooks.items():
            hsd = hook.state_dict()
            if hsd is not None:
                state_dict[key] = hook.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self._best_model_step = state_dict["best_model_step"]
        for key, hook in self.hooks.items():
            if hook.state_dict() is None:
                continue
            if key == "FIDHook": # Intermediate fix to load previous training
                hook.load_state_dict({"next_check": logger.global_step})
            else:
                hook.load_state_dict(state_dict[key])

    def save_checkpoint(self, test_checkpoint=False, is_best: bool = False):
        if utils.rank() != 0:
            return
        if is_best:
            self._best_model_step = logger.global_step
        state_dict = self.state_dict()
        if test_checkpoint:
            ckpt_dir = self.checkpoint_dir.parent.joinpath("test_checkpoints")
            checkpointer.save_checkpoint(ckpt_dir, state_dict, self._best_model_step)
        else:
            checkpointer.save_checkpoint(
                self.checkpoint_dir, state_dict, self._best_model_step)

    def load_checkpoint(self):
        if not checkpointer.checkpoint_exists(self.checkpoint_dir):
            return
        sd = checkpointer.load_checkpoint(self.checkpoint_dir)
        self.load_state_dict(sd)

    def train_loop(self):
        self.before_train()
        logger.info("Starting train.")
        while utils.GracefulKiller() and logger.global_step < self._max_images_to_train:
            self.before_step()
            self.train_step()
            self.after_step()
        logger.info("Stopping train")
        self.save_checkpoint()
