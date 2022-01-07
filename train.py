
import logging
import tempfile
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
import warnings
import traceback
import torch
import os
# We enable AMP through environment variable before importing.
# Start train with AMP_ENABLED=1 python train.py ... 
# to enable.
if os.environ.get("AMP_ENABLED") is None:
    print("AMP not set. setting to True")
    os.environ["AMP_ENABLED"] = "1"
else:
    assert os.environ["AMP_ENABLED"] in ["0", "1"]
from fba import config, engine, data, logger, utils, build_generator, build_discriminator
from fba.engine.checkpointer import checkpoint_exists
from fba.loss import build_losss_fnc

torch.backends.cudnn.benchmark = True


def start_train(rank, world_size, debug, cfg_path, temp_dir):
    if debug:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        conv2d_gradfix.enabled = False
        grid_sample_gradfix.enabled = False
    if world_size > 1:
#        os.environ['MASTER_ADDR'] = 'localhost'
#        os.environ['MASTER_PORT'] = '12355'
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        init_method = f'file://{init_file}'
        torch.distributed.init_process_group(
            "nccl", rank=rank, world_size=world_size, init_method=init_method)
        torch.cuda.set_device(utils.get_device()) # pin memory in dataloader would allocate memory on device:0 for distributed training.

    utils.set_world_size_and_rank(rank, world_size)
    cfg = config.Config.fromfile(cfg_path)
    resume_train = checkpoint_exists(cfg.checkpoint_dir)
    if utils.rank() == 0:
        logger.init(cfg, resume=resume_train)
        cfg.dump()
    if world_size > 1:
        assert cfg.data_train.loader.batch_size > utils.world_size()
        assert cfg.data_val.loader.batch_size > utils.world_size()
        assert cfg.data_train.loader.batch_size % utils.world_size() == 0
        assert cfg.data_val.loader.batch_size % utils.world_size() == 0
        cfg.data_train.loader.batch_size //= world_size
        cfg.data_val.loader.batch_size //= world_size
    if rank != 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    utils.set_seed(cfg.random_seed + rank)

    data_val = data.build_dataloader_val(cfg)
    data_train = iter(data.build_dataloader_train(cfg))
    generator = build_generator(cfg)
    discriminator = build_discriminator(cfg)

    EMA_generator = utils.EMA(generator, cfg.data_train.loader.batch_size*world_size, **cfg.EMA)
    if world_size > 1:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[rank], broadcast_buffers=False)
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank], broadcast_buffers=False)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=utils.AMP(), init_scale=4096)

    G_optimizer, D_optimizer = engine.build_optimizers(
        generator, discriminator, **cfg.optimizer
    )


    loss_fnc = build_losss_fnc(
        cfg.loss, discriminator=discriminator, generator=generator,
        scaler=grad_scaler, lazy_reg_interval=cfg.optimizer.lazy_reg_interval)
    logger.log_dictionary(
        {
            "stats/gpu_batch_size": cfg.data_train.loader.batch_size,
            "stats/gpu_batch_size_val": cfg.data_val.loader.batch_size,
            "stats/ngpus": world_size})
    trainer = engine.Trainer(
        generator=generator,
        discriminator=discriminator,
        EMA_generator=EMA_generator,
        D_optimizer=D_optimizer,
        G_optimizer=G_optimizer,
        data_train=data_train,
        data_val=data_val,
        scaler=grad_scaler,
        checkpoint_dir=cfg.checkpoint_dir,
        ims_per_log=cfg.ims_per_log,
        max_images_to_train=cfg.max_images_to_train,
        loss_handler=loss_fnc,
        cfg=cfg,
        batch_size=cfg.data_train.loader.batch_size*world_size
    )
    try:
        
        trainer.train_loop()
    except Exception as e:
        traceback.print_exc()
        exit()
    if world_size > 1:
        torch.distributed.barrier()
    logger.finish()

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = config.default_parser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true")
    args = parser.parse_args()
    if args.profile:
        import nvidia_dlprof_pytorch_nvtx
        nvidia_dlprof_pytorch_nvtx.init()

    def start():
        world_size = torch.cuda.device_count() # Manually overriding this does not work. have to set CUDA_VISIBLE_DEVICES environment variable
        if world_size > 1:
            torch.multiprocessing.set_start_method("spawn", force=True)
            with tempfile.TemporaryDirectory() as temp_dir:
                torch.multiprocessing.spawn(start_train, args=(world_size, args.debug, args.config_path, temp_dir), nprocs=torch.cuda.device_count())
        else:
            start_train(0, 1, args.debug, args.config_path, None)

    if args.profile:
        cfg = config.Config.fromfile(args.config_path)
        tb_dir = cfg.output_dir.joinpath("tensorboard_profile")
        tb_dir.mkdir(exist_ok=True, parents=True)
        with torch.autograd.profiler.emit_nvtx():
            start()
    else:
        start()
