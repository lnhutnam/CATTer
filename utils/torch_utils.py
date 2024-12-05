import os
import random
from contextlib import contextmanager

import torch
import torch.backends.cudnn as cudnn
import numpy as np

# from einops import rearrange

# from typing import TypeVar

# T = TypeVar("T")
# D = TypeVar("D")


# def exists(var: T | None) -> bool:
#     return var is not None


# def default(var: T | None, val: D) -> T | D:
#     return var if exists(var) else val


# def enlarge_as(src: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
#     """
#     Add sufficient number of singleton dimensions
#     to tensor a **to the right** so to match the
#     shape of tensor b. NOTE that simple broadcasting
#     works in the opposite direction.
#     """
#     return rearrange(src, f'... -> ...{" 1" * (other.dim() - src.dim())}').contiguous()


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    torch.backends.cudnn.benchmark = True
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def com_mult(a, b):
    return torch.mul(a, b) + a + b


def com_add(a, b):
    return torch.add(a, b)


def com_sub(a, b):
    return torch.subtract(a, b)


def com_corr(a, b):
    return (
        torch.fft.irfftn(
            torch.conj(torch.fft.rfftn(a, (-1))) *
            torch.fft.rfftn(b, (-1)), (-1)
        )
        + a
        + b
    )


def poly_ker2(x, d=2):
    return (1 + x * x) ** d


def poly_ker3(x, d=3):
    return (1 + x * x) ** d


def poly_ker5(x, d=5):
    return (1 + x * x) ** d


def gauss_ker(x, z=0, sigma=1):
    expo = -abs(x - z) ** 2 / (2 * sigma**2)
    return torch.exp(expo)
