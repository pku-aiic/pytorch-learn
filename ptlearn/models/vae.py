import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from typing import List
from typing import Optional

from ..types import losses_type
from ..types import tensor_dict_type
from ..protocol import TrainerState
from ..protocol import LossProtocol
from ..protocol import ModelProtocol
from ..constants import LOSS_KEY
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY
from ..modules.blocks import get_conv_blocks
from ..modules.blocks import Conv2d
from ..modules.blocks import Lambda
from ..modules.blocks import UpsampleConv2d


def f_map_dim(img_size: int, num_layer: int) -> int:
    return int(round(img_size / 2 ** num_layer))


def num_downsample(img_size: int, *, min_size: int = 2) -> int:
    return max(2, int(round(math.log2(img_size / min_size))))


@LossProtocol.register("vae")
class VAELoss(LossProtocol):
    def _init_config(self) -> None:
        self.kld_ratio = self.config.setdefault("kld_ratio", 0.1)

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        **kwargs: Any,
    ) -> losses_type:
        # reconstruction loss
        original = batch[INPUT_KEY]
        reconstruction = forward_results[PREDICTIONS_KEY]
        mse = F.mse_loss(reconstruction, original)
        # kld loss
        mu, log_var = map(forward_results.get, ["mu", "log_var"])
        assert mu is not None and log_var is not None
        var = log_var.exp()
        kld_losses = -0.5 * torch.sum(1 + log_var - mu ** 2 - var, dim=1)
        kld_loss = torch.mean(kld_losses, dim=0)
        # gather
        loss = mse + self.kld_ratio * kld_loss
        return {"mse": mse, "kld": kld_loss, LOSS_KEY: loss}


class VAE(ModelProtocol):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        latent_dim: int = 128,
        latent_channels: int = 128,
        first_kernel_size: int = 7,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_downsample = num_downsample(img_size)
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        # encoder
        self.latent_channels = latent_channels
        start_channels = int(round(latent_channels / (2 ** self.num_downsample)))
        if start_channels <= 0:
            raise ValueError(
                f"latent_channels ({latent_channels}) is too small "
                f"for num_downsample ({self.num_downsample})"
            )
        blocks = get_conv_blocks(
            in_channels,
            start_channels,
            first_kernel_size,
            1,
            activation=nn.LeakyReLU(0.2),
        )
        in_nc = start_channels
        for i in range(self.num_downsample):
            is_last = i == self.num_downsample - 1
            if is_last:
                out_nc = latent_channels
            else:
                out_nc = min(in_nc * 2, latent_channels)
            new_blocks: List[nn.Module]
            if is_last:
                new_blocks = [Conv2d(in_nc, out_nc, kernel_size=3, stride=2)]
            else:
                new_blocks = get_conv_blocks(
                    in_nc,
                    out_nc,
                    3,
                    2,
                    activation=nn.LeakyReLU(0.2),
                )
            blocks.extend(new_blocks)
            in_nc = out_nc
        # vae latent
        map_dim = f_map_dim(img_size, self.num_downsample)
        out_flat_dim = latent_channels * map_dim ** 2
        blocks.append(
            nn.Sequential(
                Lambda(lambda tensor: tensor.view(tensor.shape[0], -1), "flatten"),
                nn.Linear(out_flat_dim, 2 * latent_dim),
            )
        )
        self.encoder = nn.Sequential(*blocks)
        # decoder
        shape = -1, latent_channels, map_dim, map_dim
        blocks = [
            nn.Sequential(
                nn.Linear(latent_dim, out_flat_dim),
                Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
            )
        ]
        in_nc = latent_channels
        for i in range(self.num_downsample):
            out_nc = in_nc // 2
            blocks.append(UpsampleConv2d(in_nc, out_nc, factor=2, kernel_size=3))
            in_nc = out_nc
        blocks.extend(
            [
                Conv2d(
                    in_nc,
                    self.out_channels,
                    kernel_size=first_kernel_size,
                    stride=1,
                ),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*blocks)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: "TrainerState",
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        net = self.encoder(net)
        mu, log_var = net.chunk(2, dim=1)
        net = self.reparameterize(mu, log_var)
        net = self.decoder(net)
        net = F.interpolate(net, size=self.img_size)
        return {PREDICTIONS_KEY: net, "mu": mu, "log_var": log_var}


__all__ = ["VAE"]
