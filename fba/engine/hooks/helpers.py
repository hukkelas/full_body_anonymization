import typing
from fba import logger
from .build import HookBase, HOOK_REGISTRY



@HOOK_REGISTRY.register_module
class CheckpointHook(HookBase):

    def __init__(
            self,
            ims_per_checkpoint: int,
            test_checkpoints: typing.List[int], # List of test checkpoints that aren't deleted. 
            **kwargs
            ):
        self.ims_per_checkpoint = ims_per_checkpoint
        self.next_validation_checkpoint = ims_per_checkpoint
        self._test_checkpoints = test_checkpoints

    def after_step(self):
        if logger.global_step >= self.next_validation_checkpoint:
            self.next_validation_checkpoint += self.ims_per_checkpoint
            self.trainer.save_checkpoint()

        self.save_test_checkpoint()

    def state_dict(self):
        return {"next_val_checkpoint": self.next_validation_checkpoint}

    def load_state_dict(self, state_dict: dict):
        self.next_validation_checkpoint = state_dict["next_val_checkpoint"]

    def save_test_checkpoint(self):
        checkpoints = self._test_checkpoints
        for ckpt_step in checkpoints:
            ckpt_step = ckpt_step
            prev_gstep = logger.global_step - self.trainer._batch_size
            if logger.global_step >= ckpt_step and prev_gstep < ckpt_step:
                logger.info("Saving global checkpoint for validation")
                self.trainer.save_checkpoint(test_checkpoint=True)
