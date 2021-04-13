import os
import torch
import ptlearn

import numpy as np
import torch.distributed as dist

from typing import Tuple
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def get_xy(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.random([n, 2]) * 2.0 - 1.0
    y = (np.prod(x, axis=1, keepdims=True) > 0).astype(np.int64)
    return x, y


def run(
    rank: int,
    world_size: int,
    port: str = "12355",
    backend: str = "gloo",
) -> None:
    # setup ddp
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # regular run
    train_xy, valid_xy = map(get_xy, [10000, 100])
    train_loader = ptlearn.MLLoader(ptlearn.MLData(*train_xy), True)
    valid_loader = ptlearn.MLLoader(ptlearn.MLData(*valid_xy), False)
    torch.cuda.set_device(rank)
    loss = ptlearn.loss_dict["cross_entropy"]()
    fcnn = ptlearn.FCNN(2, 2, [16, 16]).to(rank)
    ddp_fcnn = DDP(fcnn, device_ids=[rank])
    inference = ptlearn.MLInference(fcnn)
    trainer = ptlearn.Trainer(num_epoch=40, rank=rank, metrics=ptlearn.Accuracy())
    trainer.fit(loss, ddp_fcnn, inference, train_loader, valid_loader, cuda=str(rank))  # type: ignore
    # clean up ddp
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    size = 2
    mp.spawn(run, args=(size,), nprocs=size, join=True)
