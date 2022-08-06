from __init__ import *


def Depthwise_Conv2d(
        channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = False
):
    if not padding and padding != 0:
        padding = dilation * (kernel_size - 1) // 2

    dw_conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation,
                        groups=channels, bias=bias)
    return dw_conv



def Pointwise_Conv2d(
        in_channels: int,
        out_channels: int,
        bias: bool = False
):
    pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
    return pw_conv



def Seperable_Conv2d(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = False
):
    conv = nn.Sequential(Depthwise_Conv2d(in_channels, kernel_size, stride, padding, dilation, False),
                         Pointwise_Conv2d(in_channels, out_channels, bias))
    return conv


class Static_ConvLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = False,
                 batch_norm: bool = True,
                 Act: Optional[nn.Module] = nn.ReLU(),
                 **kwargs):

        batch_eps = kwargs.get('eps', 1e-05)
        batch_momentum = kwargs.get('momentum', 0.1)

        padding = (kernel_size - 1) // 2

        super().__init__()

        layer = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
        if batch_norm:
            layer.append(nn.BatchNorm2d(out_channels, eps=batch_eps, momentum=batch_momentum))
        if Act:
            layer.append(Act)

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)
