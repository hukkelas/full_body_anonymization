import time
import torch
import typing
import datetime
from fba import logger, metrics, utils
from fba.utils import vis_utils
from .build import HOOK_REGISTRY, HookBase


@HOOK_REGISTRY.register_module
class TimeLoggerHook(HookBase):

    def __init__(
            self,
            num_ims_per_log: int,
            **kwargs):
        self.num_ims_per_log = num_ims_per_log
        self.last_log_step = - num_ims_per_log + 1e3
        self.start_time = time.time()
        self.to_log = {}

    def state_dict(self):
        return {
            "total_time": (time.time() - self.start_time),
        }

    def load_state_dict(self, state_dict: dict):
        self.start_time = time.time() - state_dict["total_time"]
        self.last_log_step = logger.global_step

    def before_train(self):
        self.batch_start_time = time.time()

    def after_step(self):
        if logger.global_step < self.last_log_step + self.num_ims_per_log:
            return
        time_spent = time.time() - self.batch_start_time
        num_steps = logger.global_step - self.last_log_step
        num_steps = max(num_steps, 1)
        nsec_per_img = time_spent / num_steps
        total_time = (time.time() - self.start_time) / 60
        img_per_sec = logger.global_step / (total_time*60 + 1e-8)
        remaining_images = self.trainer._max_images_to_train - logger.global_step
        remaining_seconds = remaining_images / (img_per_sec+1e-8)
        remaining_seconds = min(remaining_seconds, 60*60*24*365) # Cut off to 1 year max
        estimated_time_done = time.time() + remaining_seconds
        date = datetime.datetime.fromtimestamp(estimated_time_done)
        logger.log_variable("stats/estimated_remaining_hours", remaining_seconds/60/60)
        logger.info(
            f"Estimated finish time:" + date.strftime("%A %d. %B %H:%M:%S.") +
            f"Estimated remaining time: {int(remaining_seconds//3600)}:{int(remaining_seconds//60%60)}:{int(remaining_seconds%60)}")
        to_log = {
            "stats/nsec_per_img": nsec_per_img,
            "stats/training_time_minutes": total_time,
        }
        self.batch_start_time = time.time()
        self.last_log_step = logger.global_step
        logger.log_dictionary(to_log, commit=True)


@HOOK_REGISTRY.register_module
class MetricHook(HookBase):

    def __init__(
            self,
            ims_per_log: int,
            cache_directory: str,
            **kwargs
            ):
        self.next_check = ims_per_log
        self.num_ims_per_fid = ims_per_log
        self._min_lpips = 9999
        self.cache_directory = cache_directory

    def state_dict(self):
        return {
            "next_check": self.next_check,
            "min_lpips": self._min_lpips}

    def load_state_dict(self, state_dict: dict):
        self.next_check = state_dict["next_check"]
        self._min_lpips = state_dict["min_lpips"] if "min_lpips" in state_dict else 9999

    def after_step(self):
        if logger.global_step >= self.next_check:
            self.next_check += self.num_ims_per_fid
            self.calculate_metrics()

    def calculate_metrics(self):
        logger.info("Starting calculation of metrics.")
        generator = self.trainer.EMA_generator
        metrics_ = metrics.compute_metrics_iteratively(
            self.trainer.data_val, generator, self.cache_directory
        )
        if "lpips" in metrics_:
            val = metrics_["lpips"]
            if val < self._min_lpips:
                self._min_lpips = val
                logger.info("Found new best checkpoint: LPIPS:" + str(val))
                self.trainer.save_checkpoint(is_best=True)

        metrics_ = {f"metrics/{key}": val for key, val in metrics_.items()}
        logger.log_dictionary(metrics_, log_level=logger.INFO)


@HOOK_REGISTRY.register_module
class ImageSaveHook(HookBase):

    def __init__(
            self,
            ims_per_save: int, n_diverse_samples: int, nims2log: int,
            n_diverse_images: int,
            save_train_G: bool, # saving images from the training generator (not EMA) as well
            semantic_labels: typing.Optional[list], 
            **kwargs
        ):
        self.ims_per_save = ims_per_save
        self.next_save_point = self.ims_per_save
        self._n_diverse_samples = n_diverse_samples
        self.n_diverse_images = n_diverse_images
        self.nims2log = nims2log
        self.save_train_G = save_train_G
        if semantic_labels is not None and "color" in semantic_labels[0]:
            self.semantic_colors = [c["color"] for c in semantic_labels]
        else:
            self.semantic_colors = None
        self.zs = None

    def state_dict(self):
        return {"next_save_point": self.next_save_point, "zs": self.zs}

    def load_state_dict(self, state_dict: dict):
        self.next_save_point = state_dict["next_save_point"]
        if "zs" in state_dict and state_dict["zs"] is not None:
            self.zs = state_dict["zs"].cpu()

    def after_step(self):
        if logger.global_step >= self.next_save_point:
            self.next_save_point += self.ims_per_save
            self.save_fake_images(self.trainer.EMA_generator.generator, "validation")
            if self.save_train_G:
                self.save_fake_images(self.trainer.generator, "train")
            self.save_images_diverse()

    @torch.no_grad()
    def save_fake_images(self, generator, tag: str):
        generator.eval()
        batch = next(iter(self.trainer.data_val))
        g = generator.module if isinstance(generator, torch.nn.parallel.DistributedDataParallel) else generator
        z = torch.zeros_like(g.get_z(batch["img"]))
        fake_data_sample = utils.denormalize_img(generator(**batch, z=z)["img"]).mul(255).byte()
        real = vis_utils.visualize_batch(**batch)
        condition = utils.gather_tensors(real)[:self.nims2log]
        fake_data_sample = utils.gather_tensors(fake_data_sample)[:self.nims2log]

        to_save = torch.cat([condition, fake_data_sample])
        logger.save_images(
            tag, to_save, nrow=len(condition))
        generator.train()

    @torch.no_grad()
    def save_images_diverse(self):
        """
            Generates images with several latent variables
        """
        g = self.trainer.EMA_generator.generator
        g.eval()
        batch = next(iter(self.trainer.data_val))
        assert self.n_diverse_images % utils.world_size() == 0
        batch = {k: v[:self.n_diverse_images//utils.world_size()] for k, v in batch.items()}
        reals = vis_utils.visualize_batch(**batch)
        to_vis = [utils.gather_tensors(reals)]
        if self.zs is None:
            g_ = g.module if isinstance(g, torch.nn.parallel.DistributedDataParallel) else g
            z = [g_.get_z(batch["img"]) for _ in range(self._n_diverse_samples)]
            self.zs = torch.stack(z)[:, :self.n_diverse_images//utils.world_size()]
        self.zs = self.zs.to(batch["img"].device)
        for z in self.zs:
            fake = utils.denormalize_img(g(**batch, z=z)["img"]).mul(255).byte()
            
            to_vis.append(utils.gather_tensors(fake))
        to_vis = torch.cat(to_vis)
        logger.save_images("diverse", to_vis, nrow=self.n_diverse_images)
        g.train()
