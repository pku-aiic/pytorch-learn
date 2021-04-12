from typing import *
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler_dict = {}


def register_scheduler(name: str) -> Callable[[Type], Type]:
    def _register(cls_: Type) -> Type:
        global scheduler_dict
        scheduler_dict[name] = cls_
        return cls_

    return _register


register_scheduler("cyclic")(CyclicLR)
register_scheduler("cosine")(CosineAnnealingLR)
register_scheduler("cosine_restarts")(CosineAnnealingWarmRestarts)


__all__ = ["scheduler_dict", "register_scheduler"]
