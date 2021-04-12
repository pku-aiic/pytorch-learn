import torch
import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional

from ..types import tensor_dict_type
from ..protocol import TrainerState
from ..protocol import ModelProtocol
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY
from ..modules.blocks import BN


class Mapping(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        activation: Optional[str] = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        nets: List[nn.Module] = [nn.Linear(in_dim, out_dim, bias)]
        if activation is not None:
            nets.append(getattr(nn, activation)())
        if batch_norm:
            nets.append(BN(out_dim))
        if dropout > 0.0:
            nets.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*nets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCNN(ModelProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_units: List[int],
        *,
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        nets: List[nn.Module] = []
        for num_unit in num_units:
            mapping = Mapping(in_dim, num_unit, bias, activation, batch_norm, dropout)
            nets.append(mapping)
            in_dim = num_unit
        nets.append(nn.Linear(in_dim, out_dim, bias))
        self.net = nn.Sequential(*nets)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.net(batch[INPUT_KEY])}


__all__ = ["FCNN"]
