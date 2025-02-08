import torch
import torch.nn as nn
from lora_modules.ap.aplinear import APLinear

class APLoraLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 32,
        alpha: float = 32,
        dropout_p: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        lora_init_scale: float = 0.01,  
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        out_features, in_features = base_layer.weight.shape
        
        self.lora_A = APLinear(
            in_features=in_features,
            out_features=rank,
            dtype=dtype,
            device=base_layer.weight.device,
            is_A=True
        )
        
        self.lora_B = APLinear(
            in_features=rank,
            out_features=out_features,
            dtype=dtype,
            device=base_layer.weight.device,
            is_A=False 
        )

        for param in base_layer.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor):
        base_output = self.base_layer(x)
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        return base_output + (lora_output * self.scaling)

