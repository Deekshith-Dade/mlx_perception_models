import mlx.core as mx
import mlx.nn as nn

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.dim = dim
        self.init_values = init_values
    
    def __call__(self, x):
        return x * self.gamma
    
    def init_arrays(self):
        self.gamma = self.init_values * mx.ones((self.dim))

        