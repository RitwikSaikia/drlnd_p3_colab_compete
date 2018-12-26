import random

import numpy as np
import torch


def set_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=dtype)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    soft_update(target, source, tau=1.0)
