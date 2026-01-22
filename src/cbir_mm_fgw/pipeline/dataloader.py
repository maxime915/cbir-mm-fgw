import os
import queue
import threading
from itertools import batched
from typing import Any, Callable, Generator, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ._profiler import _profiler
from ..mrvips import ImgInfo, VIPSImage


class VIPSImageDataset(Dataset):
    def __init__(
        self, images: list[ImgInfo] | list[VIPSImage], transform: Callable[[np.ndarray], torch.Tensor]
    ):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.images[idx]
        if isinstance(img, ImgInfo):
            img = img.read()
        crop = img.numpy()
        return self.transform(crop)  # type: ignore


def _pt_dataloader(
    transform: Callable[[np.ndarray], torch.Tensor], images: list[ImgInfo] | list[VIPSImage], batch_size: int,
) -> Generator[torch.Tensor, Any, None]:
    "pytorch based dataloader"
    dataset = VIPSImageDataset(images, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,  # preserve order (this is the default)
    )

    yield from dataloader


def _preprocess_worker(
    images: list[ImgInfo] | list[VIPSImage],
    q: queue.Queue[torch.Tensor | None],
    transform: Callable[[np.ndarray], torch.Tensor],
    batch_size: int,
):
    for i in range(0, len(images), batch_size):
        with _profiler.profile("preprocess-q"):
            batch_imgs = images[i : i + batch_size]
            tensors: list[torch.Tensor] = []
            for img in batch_imgs:
                with _profiler.profile("preprocess-q/crop-img"):
                    if isinstance(img, ImgInfo):
                        img = img.read()
                with _profiler.profile("preprocess-q/read-crop"):
                    arr = img.numpy()
                with _profiler.profile("preprocess-q/transform"):
                    tensors.append(transform(arr))  # type: ignore
            tensors_s = torch.stack(tensors)
        q.put(tensors_s)
    q.put(None)


def _q_dataloader(
    transform: Callable[[np.ndarray], torch.Tensor], images: list[ImgInfo] | list[VIPSImage], batch_size: int,
):
    q: queue.Queue[torch.Tensor | None] = queue.Queue(
        maxsize=4
    )  # Tune size based on memory constraints

    # Start preprocessing in background
    thread = threading.Thread(target=_preprocess_worker, args=(images, q, transform, batch_size))
    thread.start()

    while (batch := q.get()) is not None:
        yield batch


def _s_dataloader(
    transform: Callable[[np.ndarray], torch.Tensor], images: list[ImgInfo] | list[VIPSImage], batch_size: int
):
    "simple data loader"
    for batch in batched(images, batch_size):
        batch_v = [img.read() if isinstance(img, ImgInfo) else img for img in batch]
        batch_t = [transform(img.numpy()) for img in batch_v]
        yield torch.stack(batch_t)  # type: ignore


def get_dataloader(
    dataloader: Literal["pytorch", "naive", "queue"],
    transform: Callable[[np.ndarray], torch.Tensor],
    images: list[ImgInfo] | list[VIPSImage],
    batch_size: int | None,
):
    if batch_size is None:
        batch_size = int(os.environ.get("BATCH_SIZE", "512"))

    if dataloader == "pytorch":
        return _pt_dataloader(transform, images, batch_size)
    elif dataloader == "naive":
        return _s_dataloader(transform, images, batch_size)
    elif dataloader == "queue":
        return _q_dataloader(transform, images, batch_size)
    else:
        raise ValueError(f"{dataloader=} is invalid")
