import typing
import torch
from .base_trainer import BaseTrainer
from fba import logger, utils


class Trainer(BaseTrainer):

    def __init__(
            self, 
            generator: torch.nn.Module,
            discriminator: torch.nn.Module,
            EMA_generator: torch.nn.Module,
            D_optimizer: torch.optim.Optimizer,
            G_optimizer: torch.optim.Optimizer,
            data_train: typing.Iterator,
            data_val: typing.Iterable,
            scaler: torch.cuda.amp.GradScaler,
            ims_per_log: int,
            loss_handler,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator
        self.EMA_generator = EMA_generator
        self.D_optimizer = D_optimizer
        self.G_optimizer = G_optimizer
        self.data_train = data_train
        self.data_val = data_val
        self.scaler = scaler

        self._ims_per_log = ims_per_log
        self._next_log_point = logger.global_step

        logger.log_model([self.generator, self.discriminator])
        logger.log_dictionary({
            "stats/discriminator_parameters": utils.num_parameters(self.discriminator),
            "stats/generator_parameters": utils.num_parameters(self.generator),
        }, commit=False)

        self.load_checkpoint()
        self.to_log = {}
        self.loss_handler = loss_handler

    def state_dict(self):
        G_sd = self.generator.state_dict()
        D_sd = self.discriminator.state_dict()
        EMA_sd = self.EMA_generator.state_dict()
        if isinstance(self.generator, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            G_sd = self.generator.module.state_dict()
            D_sd = self.discriminator.module.state_dict()
        state_dict = {
            "D": D_sd,
            "G": G_sd,
            "EMA_generator": EMA_sd,
            "D_optimizer": self.D_optimizer.state_dict(),
            "G_optimizer": self.G_optimizer.state_dict(),
            "global_step": logger.global_step,
        }
        state_dict.update(super().state_dict())
        return state_dict

    def load_state_dict(self, state_dict: dict):
        logger.update_global_step(state_dict["global_step"])
        self.EMA_generator.load_state_dict(state_dict["EMA_generator"])
        if isinstance(self.generator, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            self.discriminator.module.load_state_dict(state_dict["D"])
            self.generator.module.load_state_dict(state_dict["G"])
        else:
            self.discriminator.load_state_dict(state_dict["D"])
            self.generator.load_state_dict(state_dict["G"])

        self.D_optimizer.load_state_dict(state_dict["D_optimizer"])
        self.G_optimizer.load_state_dict(state_dict["G_optimizer"])
        super().load_state_dict(state_dict)

    def train_step(self):
        with torch.autograd.profiler.record_function("data_fetch"):
            batch = next(self.data_train)
        self.to_log.update(self.step_D(batch))
        self.to_log.update(self.step_G(batch))
        self.EMA_generator.update(self.generator)
        if logger.global_step >= self._next_log_point:
            log = {f"loss/{key}": item.item() for key, item in self.to_log.items()}
            logger.log_variable("amp/grad_scale", self.scaler.get_scale())
            logger.log_dictionary(log, commit=True)
            self._next_log_point += self._ims_per_log
            self.to_log = {}

    def step_D(self, batch):
        utils.set_requires_grad(self.discriminator, True)
        utils.set_requires_grad(self.generator, False)
        utils.zero_grad(self.discriminator)
        loss, to_log = self.loss_handler.D_loss(batch)
        with torch.autograd.profiler.record_function("D_step"):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.D_optimizer)
            self.scaler.update()
        utils.set_requires_grad(self.discriminator, False)
        utils.set_requires_grad(self.generator, False)
        return to_log

    def step_G(self, batch):
        utils.set_requires_grad(self.discriminator, False)
        utils.set_requires_grad(self.generator, True)
        utils.zero_grad(self.generator)
        loss, to_log = self.loss_handler.G_loss(batch)
        with torch.autograd.profiler.record_function("G_step"):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.G_optimizer)
            self.scaler.update()
        utils.set_requires_grad(self.discriminator, False)
        utils.set_requires_grad(self.generator, False)
        return to_log

    def before_step(self):
        super().before_step()
        self.EMA_generator.update_beta()
