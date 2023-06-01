import click
import torch
import os
import tempfile
from fba.metrics.torch_metrics import compute_metrics_iteratively
from fba.metrics import calculate_ppl
from fba.infer import build_trained_generator
from fba.data import build_dataloader_val
from fba import logger, config, utils


def validate(
        rank,
        config_path,
        truncation_value: float,
        batch_size: int,
        world_size,
        temp_dir,
        ):
    utils.set_seed(0)
    if world_size > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        init_method = f'file://{init_file}'
        torch.distributed.init_process_group(
            "nccl", rank=rank, world_size=world_size, init_method=init_method)
        torch.cuda.set_device(utils.get_device()) # pin memory in dataloader would allocate memory on device:0 for distributed training.
        utils.set_world_size_and_rank(rank, world_size)


    cfg = config.Config.fromfile(config_path)
    cfg.data_val.dataset.percentage = 1

    if batch_size is not None:
        cfg.data_val.loader.batch_size = batch_size
    if world_size > 1:
        assert cfg.data_train.loader.batch_size > utils.world_size()
        assert cfg.data_val.loader.batch_size > utils.world_size()
        assert cfg.data_train.loader.batch_size % utils.world_size() == 0
        assert cfg.data_val.loader.batch_size % utils.world_size() == 0
        cfg.data_train.loader.batch_size //= world_size
        cfg.data_val.loader.batch_size //= world_size
    dataloader = build_dataloader_val(cfg)
    
#    generator = build_generator(cfg)
    generator, global_step = build_trained_generator(cfg)
    # Center mask
    ppl_metrics = calculate_ppl(dataloader, generator)
    print(ppl_metrics)
    metric_values = compute_metrics_iteratively(
        dataloader, generator,
        "test",
        truncation_value=truncation_value,
        autocast_fp16=False,
        include_two_fake=True)

    metric_values.update(ppl_metrics)
    print(metric_values)
    identity = ""
    if truncation_value is not None:
        identity = identity + f"trunc{truncation_value}/"
    metric_values_center = {"metrics_end/" + identity + k: v for k,v in metric_values.items()}
    if rank == 0:
        logger.init(cfg, resume=True)
        logger.update_global_step(global_step)
        logger.log_dictionary(metric_values_center)
        logger.finish()


@click.command()
@click.argument("config_path")
@click.option("--truncation_value", default=None, type=float)
@click.option("--batch_size", default=16, type=int)
def main(config_path, truncation_value: float, batch_size: int,):
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.set_start_method("spawn", force=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.multiprocessing.spawn(validate,
                args=(config_path, truncation_value, batch_size, world_size, temp_dir),
                nprocs=world_size)
    else:
        validate(
            0, config_path, truncation_value, batch_size,
            world_size=1, temp_dir=None)


if __name__ == "__main__":
    if os.environ.get("AMP_ENABLED") is None:
        print("AMP not set. setting to False")
        os.environ["AMP_ENABLED"] = "0"
    else:
        assert os.environ["AMP_ENABLED"] in ["0", "1"]
    main()