import torch
import numpy as np
from fba.utils import to_cuda, get_device, get_seed
from contextlib import contextmanager


def cuda_stream_wrap(stream):
    if torch.cuda.is_available():
        return torch.cuda.stream(stream)

    @contextmanager
    def placeholder(stream):
        try:
            yield
        finally:
            pass
    return placeholder(stream)


class DataPrefetcher:

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 image_gpu_transforms: torch.nn.Module,
                 embed_map=None):
        self.original_loader = loader
        self.stream = None
        self.embed_map = to_cuda(embed_map)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream(device=get_device())
        self.loader = iter(self.original_loader)
        self.image_gpu_transforms = image_gpu_transforms
        self.it = 0

    @torch.no_grad()
    def _preload(self):
        try:
            self.container = next(self.loader)
            self.stop_iteration = False
        except StopIteration:
            self.stop_iteration = True
            return
        with cuda_stream_wrap(self.stream):
            for key, item in self.container.items():
                self.container[key] = to_cuda(item).float()
            if "E_mask" in self.container:
                self.container["border"] = 1 - self.container["mask"] - self.container["E_mask"]
            if "mask" in self.container:
                mask = self.container["mask"]
                mask = mask.view(mask.shape[0], 1, mask.shape[-2], mask.shape[-1])
                self.container["mask"] = mask
            self.container["img"] = self.container["img"] / 255
            self.container["embed_map"] = self.embed_map.clone()
            self.container = self.image_gpu_transforms(self.container)

    def __len__(self):
        return len(self.original_loader)

    def __next__(self):
        return self.next()

    def next(self):
        if torch.cuda.is_available():
            torch.cuda.current_stream(device=get_device()).wait_stream(self.stream)
        if self.stop_iteration:
            raise StopIteration
        container = self.container
        self._preload()
        return container

    def __iter__(self):
        self.it += 1
        self.loader = iter(self.original_loader)
        self._preload()
        return self

    @property
    def batch_size(self):
        return self.original_loader.batch_size

    @property
    def drop_last(self):
        return self.original_loader.drop_last or self.original_loader.sampler.drop_last

    @property
    def shuffle(self):
        return self.original_loader.shuffle

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.
# From https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/torch_utils/misc.py

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, window_size=0.5, **kwargs):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = get_seed()
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1