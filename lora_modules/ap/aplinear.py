import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class APLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda",
        is_A: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if is_A:
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features, device=device, dtype=dtype)
                * math.sqrt(2. / in_features)
            )
        else:
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features, device=device, dtype=dtype)
                * 0.01
            )
        
        self.scale = nn.Parameter(
            torch.ones(out_features, device=device, dtype=dtype),
            requires_grad=True
        )
        
        with torch.no_grad():
            self.scale.copy_(self.weight.abs().mean(dim=1) + 1e-6)

    def forward(self, x: torch.Tensor):
        binary_weights = torch.sign(self.weight)
        scaled_weights = binary_weights * self.scale.unsqueeze(1)
        
        quantized_weights = self.weight + (scaled_weights - self.weight).detach()
        return F.linear(x, quantized_weights)
