from abc import ABC, abstractmethod

import mlx.nn as nn

class BaseProjector(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.adaptive_avg_pool = None
    
    @abstractmethod
    def setup_projector(self):
        pass

    def __call__(self, x):
        x = x.transpose(1, 0, 2) # NLD -> LND
        x = self.projector(x)
        x = x.transpose(1, 0, 2)
        if self.adaptive_avg_pool is not None:
            x = self.adaptive_avg_pool(x)
        return x