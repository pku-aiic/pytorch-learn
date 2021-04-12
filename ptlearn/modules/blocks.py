import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional

from ..misc.toolkit import update_dict


class Lambda(nn.Module):
    def __init__(self, fn: Callable, name: str = None):
        super().__init__()
        self.name = name
        self.fn = fn

    def extra_repr(self) -> str:
        return "" if self.name is None else self.name

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


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
        blocks: List[nn.Module] = [nn.Linear(in_dim, out_dim, bias)]
        if activation is not None:
            blocks.append(getattr(nn, activation)())
        if batch_norm:
            blocks.append(BN(out_dim))
        if dropout > 0.0:
            blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BN(nn.BatchNorm1d):
    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        net = super().forward(net)
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        return net


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: Any = "reflection",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        gain: float = math.sqrt(2.0),
    ):
        super().__init__()
        self.in_c, self.out_c = in_channels, out_channels
        self.kernel_size = kernel_size
        self.reflection_pad = None
        if padding == "reflection":
            padding = 0
            reflection_padding: Any = kernel_size // 2
            if transform_kernel:
                reflection_padding = [reflection_padding] * 4
                reflection_padding[0] += 1
                reflection_padding[2] += 1
            self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.groups, self.stride = groups, stride
        self.dilation, self.padding = dilation, padding
        self.transform_kernel = transform_kernel
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        if not bias:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.demodulate = demodulate
        # initialize
        with torch.no_grad():
            # nn.init.normal_(self.weight.data, 0.0, 0.02)
            nn.init.xavier_normal_(self.weight.data, gain / math.sqrt(2.0))
            if self.bias is not None:
                self.bias.zero_()

    def _same_padding(self, size: int) -> int:
        stride = self.stride
        dilation = self.dilation
        return ((size - 1) * (stride - 1) + dilation * (self.kernel_size - 1)) // 2

    def forward(self, x: Tensor, style: Optional[Tensor] = None) -> Tensor:
        b, c, *hw = x.shape
        # padding
        padding = self.padding
        if self.padding == "same":
            padding = tuple(map(self._same_padding, hw))
        if self.reflection_pad is not None:
            x = self.reflection_pad(x)
        # transform kernel
        w: Union[nn.Parameter, Tensor] = self.weight
        if self.transform_kernel:
            w = F.pad(w, [1, 1, 1, 1], mode="constant")
            w = (
                w[:, :, 1:, 1:]
                + w[:, :, :-1, 1:]
                + w[:, :, 1:, :-1]
                + w[:, :, :-1, :-1]
            ) * 0.25
        # ordinary convolution
        if style is None:
            bias = self.bias
            groups = self.groups
        # 'stylized' convolution, used in StyleGAN
        else:
            suffix = "when `style` is provided"
            if self.bias is not None:
                raise ValueError(f"`bias` should not be used {suffix}")
            if self.groups != 1:
                raise ValueError(f"`groups` should be 1 {suffix}")
            if self.reflection_pad is not None:
                raise ValueError(
                    f"`reflection_pad` should not be used {suffix}, "
                    "maybe you want to use `same` padding?"
                )
            w = w[None, ...] * (style[..., None, :, None, None] + 1.0)
            # prepare for group convolution
            bias = None
            groups = b
            x = x.view(1, -1, *hw)  # 1, b*in, h, w
            w = w.view(b * self.out_c, *w.shape[2:])  # b*out, in, wh, ww
        if self.demodulate:
            w = w * torch.rsqrt(w.pow(2).sum([-3, -2, -1], keepdim=True) + 1e-8)
        # if style is provided, we should view the results back
        x = F.conv2d(
            x,
            w,
            bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=groups,
        )
        if style is None:
            return x
        return x.view(-1, self.out_c, *hw)

    def extra_repr(self) -> str:
        return (
            f"{self.in_c}, {self.out_c}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"demodulate={self.demodulate}"
        )


def upscale(x: torch.Tensor, factor: float) -> torch.Tensor:
    return F.interpolate(x, scale_factor=factor, recompute_scale_factor=True)  # type: ignore


class UpsampleConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: Union[int, str] = "reflection",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        factor: float = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            dilation=dilation,
            padding=padding,
            transform_kernel=transform_kernel,
            bias=bias,
            demodulate=demodulate,
        )
        self.factor = factor

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if self.factor is not None:
            x = upscale(x, factor=self.factor)
        return super().forward(x, y)


class PixelNorm(nn.Module):
    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return F.normalize(net, dim=1)


class NormFactory:
    def __init__(self, norm_type: Optional[str]):
        self.norm_type = norm_type

    @property
    def use_bias(self) -> bool:
        return self.norm_type is None or not self.norm_type.startswith("batch")

    @property
    def norm_base(self) -> Callable:
        norm_type = self.norm_type
        norm_layer: Union[Type[nn.Module], Any]
        if norm_type == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "batch1d":
            norm_layer = nn.BatchNorm1d
        elif norm_type == "instance":
            norm_layer = nn.InstanceNorm2d
        elif norm_type == "spectral":
            norm_layer = torch.nn.utils.spectral_norm
        elif norm_type == "pixel":
            norm_layer = PixelNorm
        elif norm_type is None:

            def norm_layer(_: Any, *__: Any, **___: Any) -> nn.Identity:
                return nn.Identity()

        else:
            msg = f"normalization layer '{norm_type}' is not found"
            raise NotImplementedError(msg)
        return norm_layer

    @property
    def default_config(self) -> Dict[str, Any]:
        norm_type = self.norm_type
        config = {}
        if norm_type == "batch":
            config = {"affine": True, "track_running_stats": True}
        elif norm_type == "instance":
            config = {"affine": False, "track_running_stats": False}
        return config

    def make(self, *args: Any, **kwargs: Any) -> nn.Module:
        kwargs = update_dict(kwargs, self.default_config)
        return self.norm_base(*args, **kwargs)

    def inject_to(
        self,
        dim: int,
        norm_kwargs: Dict[str, Any],
        current_blocks: List[nn.Module],
        *subsequent_blocks: nn.Module,
    ) -> None:
        if self.norm_type != "spectral":
            new_block = self.make(dim, **norm_kwargs)
            current_blocks.append(new_block)
        else:
            last_block = current_blocks[-1]
            last_block = self.make(last_block, **norm_kwargs)
            current_blocks[-1] = last_block
        current_blocks.extend(subsequent_blocks)


def get_conv_blocks(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    *,
    bias: bool = True,
    demodulate: bool = False,
    norm_type: Optional[str] = None,
    norm_kwargs: Optional[Dict[str, Any]] = None,
    activation: Optional[nn.Module] = None,
    **conv2d_kwargs: Any,
) -> List[nn.Module]:
    blocks: List[nn.Module] = [
        Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            demodulate=demodulate,
            **conv2d_kwargs,
        )
    ]
    if not demodulate:
        factory = NormFactory(norm_type)
        factory.inject_to(out_channels, norm_kwargs or {}, blocks)
    if activation is not None:
        blocks.append(activation)
    return blocks
