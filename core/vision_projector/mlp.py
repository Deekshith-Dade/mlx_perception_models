import math
import mlx.core as mx
import mlx.nn as nn

from core.utils import get_init_fn
from core.vision_projector.base import BaseProjector

class AdaptiveAvgPooling(nn.Module):
    def __init__(self, pooling_ratio=2):
        super(AdaptiveAvgPooling, self).__init__()
        self.pooling_ratio = pooling_ratio
    
    def __call__(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        
        x = x.reshape(b, h, h, c)
        
        new_h = h // self.pooling_ratio
        new_w = h // self.pooling_ratio
        
        x = x.reshape(b, new_h, self.pooling_ratio, new_w, self.pooling_ratio, c)
        
        x = mx.mean(x, axis=(2, 4))
        x = x.reshape(b, new_h * new_w, c)
        
        return x
        
        
class MLPProjector(BaseProjector):
    def __init__(self, args):
        super().__init__()
        self.setup_projector(args)
        self.pooling_ratio = args.pooling_ratio
        self.adaptive_avg_pool = AdaptiveAvgPooling(pooling_ratio=args.pooling_ratio)
        self.remove_vision_class_token = args.remove_vision_class_token

    def init_arrays(self):
        self.projector.layers[0].weight = self.init_method(self.projector.layers[0].weight)
        self.projector.layers[0].bias = self.init_method(self.projector.layers[0].bias)
        self.projector.layers[2].weight = self.init_method(self.projector.layers[2].weight)
        self.projector.layers[2].bias = self.init_method(self.projector.layers[2].bias)
    
    def sanitize_weights(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "vision_projector.projector" in key:
                n_key = key.replace("vision_projector.projector", "vision_projector.projector.layers")
                new_state_dict[n_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    def setup_projector(self, args):
        self.init_method = get_init_fn(args.mlp_init, args.dim, init_depth=None)
        input_size = args.vision_model["width"]
        output_size = args.dim
        self.projector = nn.Sequential(
            nn.Linear(
                input_dims=input_size,
                output_dims=output_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(
                input_dims=input_size,
                output_dims=output_size,
                bias=True,
            ),
        )


