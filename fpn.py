from __init__ import *


class FPN(nn.Module):

    __doc__ = r"""
        paper: https://arxiv.org/abs/1612.03144
        
        * All list arguments and input, output feature maps are given in bottom-to-top.
          
        Args:
            num_levels: the number of feature maps
            in_channels: channels of each input feature maps in list
            out_channels: channels of output feature maps 
            sizes: 2d size of each feature maps in list
            up_mode: nn.Upsample mode
        
        Output:
            list of feature maps in the same number of channels
            
        If 'sizes' is not given, 'scale_factor' of every upsampling are set to 2.
        """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: Optional[List] = None,
                 up_mode: str = 'nearest'):

        self.num_levels = num_levels

        assert len(in_channels) == num_levels, \
            'make len(in_channels) = num_levels'
        if sizes:
            assert len(sizes) == num_levels, \
                'make len(sizes) = num_levels'

        super().__init__()

        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

        if sizes:
            self.upsamples = nn.ModuleList([nn.Upsample(size=size, mode=up_mode) for size in sizes[:-1]])
        else:
            self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2, mode=up_mode) for _ in range(num_levels - 1)])

        self.fuses = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) for _ in range(num_levels)])


    def forward(self, features: List[Tensor]) -> List[Tensor]:
        p_features = []

        for i in range(self.num_levels - 1, -1, -1):
            p = self.laterals[i](features[i])

            if p_features:
                u = self.upsamples[i](p_features[-1])
                p += u

            p_features.append(p)

        p_features = p_features[::-1]
        p_features = [f(p) for f, p in zip(self.fuses, p_features)]

        return p_features

