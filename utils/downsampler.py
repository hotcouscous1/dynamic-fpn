from __init__ import *


class Downsampler_Conv(nn.Module):

    __doc__ = r"""
        This module adjusts padding to get a desired feature size from the given size,
        and downsample a feature by nn.Conv2d.
        
        Args:
            in_size: 2d size of input feature map, assumed that the height and width are same
            out_size: 2d size of output feature map, assumed that the height and width are same
            in_channels: in_channels of nn.Conv2d
            out_channels: out_channels of nn.Conv2d
            kernel_size: kernel_size of nn.Conv2d
            stride: stride of nn.Conv2d
            dilation: dilation of nn.Conv2d
            groups: groups of nn.Conv2d
            bias: bias of nn.Conv2d
        
        Output:
            a downsampled tensor
        
        The error condition is same as for nn.Conv2d.
    """

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 **kwargs):

        super().__init__()
        padding = math.ceil((stride * (out_size - 1) - in_size + dilation * (kernel_size - 1) + 1) / 2)

        if padding < 0:
            raise ValueError('negative padding is not supported for Conv2d')
        if stride < 2:
            raise ValueError('downsampling stride must be greater than 1')

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, **kwargs)

    def forward(self, x):
        return self.conv(x)



class Downsampler_Pool(nn.Module):

    __doc__ = r"""
        This module adjusts padding to get a desired feature size from the given size,
        and downsample a feature by nn.Maxpool2d or nn.AvgPool2d.

        Args:
            in_size: 2d size of input feature map, assumed that the height and width are same
            out_size: 2d size of output feature map, assumed that the height and width are same
            mode: decide whether to apply nn.MaxPool2d or nn.AvgPool2d
            kernel_size: kernel_size of nn.MaxPool2d and nn.AvgPool2d
            stride: stride of nn.MaxPool2d and nn.AvgPool2d
            dilation: dilation of nn.MaxPool2d

        Output:
            a downsampled tensor

        The error condition is same as for nn.MaxPool2d or nn.AvgPool2d.
    """

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 mode: str = 'maxpool',
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 **kwargs):

        super().__init__()

        if mode == 'maxpool':
            padding = math.ceil((stride * (out_size - 1) - in_size + dilation * (kernel_size - 1) + 1) / 2)

            if padding > kernel_size / 2:
                raise ValueError('pad should be smaller than half of kernel size in Pool2d')
            if stride < 2:
                raise ValueError('downsampling stride must be greater than 1')

            self.pool = nn.MaxPool2d(kernel_size, stride, padding, dilation, **kwargs)


        elif mode == 'avgpool':
            padding = math.ceil((stride * (out_size - 1) - in_size + (kernel_size - 1) + 1) / 2)

            if padding > kernel_size / 2:
                raise ValueError('pad should be smaller than half of kernel size in Pool2d')
            if stride < 2:
                raise ValueError('downsampling stride must be greater than 1')

            self.pool = nn.AvgPool2d(kernel_size, stride, padding, **kwargs)

        else:
            raise ValueError('please select the mode between maxpool and avgpool')


    def forward(self, x):
        return self.pool(x)

