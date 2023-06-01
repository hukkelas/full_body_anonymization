import signal
import torch
import os
from time import time
from torch.hub import get_dir
from contextlib import contextmanager
from pathlib import Path



def iterate_resolutions(start_res, end_res, mul=2, reverse=False):
    if reverse:
        while end_res >= start_res:
            yield end_res
            end_res //= mul
    else:
        while start_res <= end_res:
            yield start_res
            start_res = start_res * mul


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args, **kwargs):
        self.kill_now = True

    def __bool__(self):
        return not self.kill_now


def cache_embed_stats(embed_map: torch.Tensor):
    mean = embed_map.mean(dim=0, keepdim=True)
    rstd = ((embed_map - mean).square().mean(dim=0, keepdim=True)+1e-8).rsqrt()

    cache = dict(mean=mean, rstd=rstd, embed_map=embed_map)
    path = Path(get_dir(), f"embed_map_stats.torch")
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(cache, path)


def get_embed_stats():
    fp = os.path.join(get_dir(), f"embed_map_stats.torch")
    if os.path.isfile(fp):
        cache = torch.load(fp, map_location="cpu")
        return cache["mean"], cache["rstd"], cache["embed_map"]
    return None, None, None

@contextmanager
def timeit(desc):
    try:
        torch.cuda.synchronize()
        t0 = time()
        yield
    finally:
        torch.cuda.synchronize()
        print(f"({desc}) total time: {time() - t0:.1f}")

