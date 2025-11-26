
import torch
import torch.nn as nn
import torch.nn.functional as F
class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.orig = orig_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.in_features = orig_linear.in_features 
        self.out_features = orig_linear.out_features 

        # 保存原始权重和偏置的引用，主要是open_clip需要这个接口
        self.weight = orig_linear.weight
        self.bias = orig_linear.bias
        
        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))  # (r, in_dim)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))  # (out_dim, r)
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = self.orig(x)
        lora = (x @ self.lora_A.T) @ self.lora_B.T
        return out + self.scaling * lora