from __init__ import *


class FeatureFusion(nn.Module):

    __doc__ = r"""
        This module is to fuse features by addition, multiplication or concatenation.

        Args:
            num: number of feature maps to be fused
            mode: how to combine features, it should be one of 'sum', 'mul', 'concat'
            weight: assign trainable weights to each feature map 
            normalize: normalize feature maps by the weight
            nonlinear: apply non-linear function to the weight 
            softmax: apply softmax function to the weight

        Output:
            a fused feature
            
        If 'softmax = True', 'normalize' should be False because it is redundant.
    """

    def __init__(self,
                 num: int,
                 mode: str = 'sum',
                 weight: bool = True,
                 normalize: bool = True,
                 nonlinear: Optional[nn.Module] = None,
                 softmax: bool = False):

        super().__init__()

        self.weight = torch.ones(num, dtype=torch.float32, device=device)
        if weight:
            self.weight = nn.Parameter(self.weight)

        self.mode = mode
        self.normalize = normalize
        self.nonlinear = nonlinear
        self.softmax = softmax


    def forward(self, features):
        weight = self.weight
        fusion = 0

        if self.nonlinear:
            weight = self.nonlinear(weight)

        if self.softmax:
            weight = weight.softmax(dim=0)

        if self.mode == 'sum':
            for w, f in zip(weight, features):
                fusion += w * f

        elif self.mode == 'mul':
            for w, f in zip(weight, features):
                fusion *= w * f

        elif self.mode == 'concat':
            features = [w * f for w, f in zip(weight, features)]
            fusion = torch.cat(features, dim=1)

        else:
            raise RuntimeError('select mode in sum, mul and concat')

        if self.normalize and not self.softmax:
            fusion /= (weight.sum() + 1e-4)

        return fusion

