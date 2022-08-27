# Dynamic FPNs: FPN, PAN, BiFPN

<p align="center">
  <img src="https://github.com/hotcouscous1/Logo/blob/main/TensorBricks_Logo.png" width="500" height="120">
</p>

The parameters of fpn must be given independently from the backbone.  
  
Dynamic FPNs are free from
- the number and channels of pyramid-levels
- the size of input image, to decide each level's feature map size 
- the strides between feature maps  

```python
# example1
fpn = FPN(num_levels=4,
          in_channels=[64, 128, 128, 256],
          out_channels=128,
          sizes=[20, 10, 10, 5])

# example2
bifpn = BiFPN(num_levels=5,
              num_repeat=3,
              in_channels=[64, 128, 256],
              out_channels=128,
              sizes=[40, 20, 10, 4, 1],
              strides=[2, 2, 2, 4])
```

## License
BSD 3-Clause License Copyright (c) 2022, hotcouscous1
