"""
multi_head_trans.py
Function: Multi-heads Attention Module
Author: aihonfeng
History:
20241218    aihongfeng  v1
"""
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=2, head_dim=160, channel_dim=320):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.linear_q = torch.nn.Linear(channel_dim, channel_dim)
        self.linear_k = torch.nn.Linear(channel_dim, channel_dim)
        self.linear_v = torch.nn.Linear(channel_dim, channel_dim)

    def split_heads(self, tensor, num_heads):
        batch_size, seq_len, feature_dim = tensor.size()
        head_dim = feature_dim // num_heads
        output = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        # B x H x O x Ch
        return  output 

    def forward(self, x):
        # B x Ch x O -> B x O x Ch
        x = x.transpose(1,2)

        assert x.size(-1) == self.num_heads * self.head_dim

        Q = self.linear_q(x)  # B x O x Ch
        K = self.linear_k(x)  # B x O x Ch
        V = self.linear_v(x)  # B x O x Ch

        Q = self.split_heads(Q, self.num_heads)  # B x H x O x Ch
        K = self.split_heads(K, self.num_heads)  # B x H x O x Ch
        V = self.split_heads(V, self.num_heads)  # B x H x O x Ch
        
        raw_weights = torch.matmul(Q, K.transpose(-2, -1)) # B x H x O x O

        # scale weights
        scale_factor = K.size(-1) ** 0.5
        scaled_weights = raw_weights / scale_factor  # B x H x O x O

        # softmax the scale weights and get attention wgt
        attn_weights = F.softmax(scaled_weights, dim=-1)  # B x H x O x O

        # attentive value
        attn_outputs = torch.matmul(attn_weights, V)  # B x H x O x head_dim

        return attn_outputs


if __name__=="__main__":
   x = torch.randn(64, 320, 256)  # B 64 x Ch 320 x O 256, O is time steps
   MHA = MultiHeadAttention(num_heads=2, head_dim=160, channel_dim=320)
   attn_outputs = MHA(x)
   print(attn_outputs.shape) # B x H x O x head_dim torch.Size([64, 2, 256, 160])