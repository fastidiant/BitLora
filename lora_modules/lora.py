import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(
        self, 
        base_layer: nn.Module,
        rank: int = 32,
        alpha: float = 32,
        dropout_p: float = 0.0,
        dtype: str = torch.bfloat16,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        
        weight_shape = base_layer.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(weight_shape[1], rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, weight_shape[0], device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for param in self.base_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = (x @ self.lora_A) @ self.lora_B
        return base_output + (lora_output * self.scaling)


    
