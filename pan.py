from __init__ import *
from utils.downsampler import Downsampler_Conv
from fpn import FPN


class BU_FPN(nn.Module):

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: Optional[List] = None,
                 strides: list = None):

        self.num_levels = num_levels

        assert len(in_channels) == num_levels, \
            'make len(in_channels) = num_levels'
        if sizes:
            assert len(sizes) == num_levels and len(strides) == num_levels - 1, \
                'make len(sizes) = num_levels, and len(strides) = num_levels - 1'

        super().__init__()

        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

        if sizes and strides:
            self.downsamples = nn.ModuleList([Downsampler_Conv(sizes[i], sizes[i + 1], out_channels, out_channels, 1, strides[i], bias=True)
                                              for i in range(len(sizes) - 1)])
        else:
            self.downsamples = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 1, 2, padding=0, bias=True)
                                              for _ in range(num_levels - 1)])

        self.fuses = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) for _ in range(num_levels)])


    def forward(self, features: List[Tensor]) -> List[Tensor]:
        p_features = []

        for i in range(self.num_levels):
            p = self.laterals[i](features[i])

            if p_features:
                d = self.downsamples[i - 1](p_features[-1])
                p += d

            p = self.fuses[i](p)
            p_features.append(p)

        return p_features



class PAN(nn.Module):

    __doc__ = r"""
        paper: https://arxiv.org/abs/1803.01534

        * All list arguments and input, output feature maps are given in bottom-to-top.  

        Args:
            num_levels: the number of feature maps
            in_channels: channels of each input feature maps in list
            out_channels: channels of output feature maps 
            sizes: 2d size of each feature maps in list
            strides: list of strides between two feature maps, of nn.Conv2d for downsampling
            up_mode: nn.Upsample mode

        Output:
            list of feature maps in the same number of channels

        If 'sizes' and 'strides' are not given, 'scale_factor' of every upsampling 
        and 'stride' of every downsampling are set to 2.
        """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: Optional[List] = None,
                 strides: Optional[List] = None,
                 up_mode: str = 'nearest'):

        super().__init__()

        self.top_down = FPN(num_levels, in_channels, out_channels, sizes, up_mode)
        self.bottom_up = BU_FPN(num_levels, len(in_channels) * [out_channels], out_channels, sizes, strides)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        features = self.top_down(features)
        features = self.bottom_up(features)

        return features

