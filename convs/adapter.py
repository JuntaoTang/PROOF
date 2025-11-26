import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterLayer(nn.Module):
    """Adapter 层"""
    
    def __init__(self, input_dim, adapter_dim=64, activation='relu',alpha=0.8):
        super().__init__()
        
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, input_dim)
        self.alpha=alpha

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        # 初始化适配器
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        return (1-self.alpha)*self.up_proj(self.activation(self.down_proj(x)))+self.alpha*x
