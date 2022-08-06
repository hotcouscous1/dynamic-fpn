import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Optional

Numpy = np.array
Tensor = torch.Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')