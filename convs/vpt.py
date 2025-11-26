import torch
import torch.nn as nn
import torch.nn.functional as F
class VisualPrompt(nn.Module):
    def __init__(self, prompt_length=10, embed_dim=768):  # prompt_length 控制参数量
        super().__init__()
        # 可学习的 Prompt 向量（shape: [prompt_length, embed_dim]）
        self.prompts = nn.Parameter(torch.randn(prompt_length, embed_dim))
        # 初始化（用正态分布初始化，均值 0，方差 0.02）
        nn.init.normal_(self.prompts, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]（ViT 输入序列，含 CLS token + 图像 patch token）
        batch_size = x.shape[0]
        # 扩展 Prompt 到批次维度：[batch_size, prompt_length, embed_dim]
        prompts = self.prompts.unsqueeze(0).repeat(batch_size, 1, 1)
        # 插入到 CLS token 之后、图像 patch 之前
        return torch.cat([x[:, :1, :], prompts, x[:, 1:, :]], dim=1)  # [batch_size, 1 + prompt_length + seq_len -1, embed_dim]