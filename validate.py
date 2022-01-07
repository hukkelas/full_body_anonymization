import click
import os
from fba.metrics.torch_metrics import compute_metrics_iteratively
from fba.infer import build_trained_generator
from fba.data import build_dataloader_val
from fba import logger, config


@click.command()
@click.argument("config_path")
@click.option("--n_diversity_samples", default=3, type=int)
@click.option("--truncation_value", default=None, type=float)
@click.option("--batch_size", default=16, type=int)
@click.option("--metrics", default=["l1", "mse", "psnr", "lpips"], type=click.Choice(["l1", "mse", "psnr", "lpips"]), multiple=True)
def validate(
        config_path,
        metrics,
        n_diversity_samples: int,
        truncation_value: float,
        batch_size: int,
         **kwargs
        ):
    cfg = config.Config.fromfile(config_path)
    cfg.data_val.dataset.percentage = 1
    if batch_size is not None:
        cfg.data_val.loader.batch_size = batch_size
    dataloader = build_dataloader_val(cfg)

    generator, global_step = build_trained_generator(cfg)
    # Center mask
    metric_values = compute_metrics_iteratively(
        dataloader, generator,
        metrics, n_diversity_samples,
        truncation_value=truncation_value,
        fid_real_directory=cfg.fid_real_directory)
    print(metric_values)
    identity = f"/top{n_diversity_samples}/"
    if truncation_value is not None:
        identity = identity + f"trunc{truncation_value}/"
    metric_values_center = {"metrics_end" + identity + k: v for k,v in metric_values.items()}

    logger.init(cfg, resume=True)
    logger.update_global_step(global_step)
    logger.log_dictionary(metric_values_center)
    logger.finish()


if __name__ == "__main__":
    if os.environ.get("AMP_ENABLED") is None:
        print("AMP not set. setting to True")
        os.environ["AMP_ENABLED"] = "1"
    else:
        assert os.environ["AMP_ENABLED"] in ["0", "1"]
    validate()