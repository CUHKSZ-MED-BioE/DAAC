"""
encoder.py
Function: encoder
History:
20241219    aihongfeng  add MultiHeadTSEncoder
20250101    aihongfeng  add view mask for self-patient-view loss but not view mask for patient-view loss
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from .multi_head_trans import MultiHeadAttention as MHA

def generate_continuous_mask(B, T, C=None, n=5, l=0.1):
    if C:
        res = torch.full((B, T, C), True, dtype=torch.bool)
    else:
        res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            if C:
                # For a continuous timestamps, mask random half channels
                index = np.random.choice(C, int(C/2), replace=False)
                res[i, t:t + l, index] = False
            else:
                # For a continuous timestamps, mask all channels
                res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, C=None, p=0.5):
    if C:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T, C))).to(torch.bool)
    else:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class ProjectionHead(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        # projection head for finetune
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.repr_dropout(self.proj_head(x))
        if self.output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x
    
class FTClassifier(nn.Module):
    def __init__(self, input_dims, output_dims, depth, p_output_dims, 
                 pat_multihead=False, tra_multihead=False, hidden_dims=64, num_heads=2, head_dim=160, channel_dim=320,
                 p_hidden_dims=128, device='cuda', flag_use_multi_gpu=True):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.p_hidden_dims = p_hidden_dims  # Cph
        self.p_output_dims = p_output_dims  # Cp
        self.pat_multihead = pat_multihead
        self.tra_multihead = tra_multihead
        self.dualmultihead = pat_multihead and tra_multihead
        self.multihead = pat_multihead or tra_multihead
        # @ahf
        if self.pat_multihead and self.tra_multihead:
            self._net = DaulMultiHeadTSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, num_heads=num_heads, head_dim=head_dim, channel_dim=channel_dim)
        elif self.pat_multihead or self.tra_multihead:
            print('Using MultiHeadTSEncoder')
            self._net = MultiHeadTSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, num_heads=num_heads, head_dim=head_dim, channel_dim=channel_dim)
        else:
            self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        # projection head for finetune
        self.proj_head = ProjectionHead(output_dims, p_output_dims, p_hidden_dims)
        device = torch.device(device)
        if device == torch.device('cuda') and flag_use_multi_gpu:
            self._net = nn.DataParallel(self._net)
            self.proj_head = nn.DataParallel(self.proj_head)
        self._net.to(device)
        self.proj_head.to(device)

        # stochastic weight averaging, see link:
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

    def forward(self, x):
        # @ahf
        if self.dualmultihead:
            out, pat_views_out, tri_views_out = self.net(x)
            # concat out and views_out
            # B x H x O x head_dim -> B x O x H*head_dim 
            B, Od = pat_views_out.size(0), pat_views_out.size(2)
            pat_views_out = torch.einsum('bhmo->bmho', pat_views_out)
            # pat_views_out = pat_views_out.contiguous().view(B, Od, -1)
            out = torch.cat([out, pat_views_out.contiguous().view(B, Od, -1)], dim=2)

            B, Od = tri_views_out.size(0), tri_views_out.size(2)
            tri_views_out = torch.einsum('bhmo->bmho', tri_views_out)
            # tri_views_out = tri_views_out.contiguous().view(B, Od, -1)
            out = torch.cat([out, tri_views_out.contiguous().view(B, Od, -1)], dim=2)
        elif self.multihead:
            out, views_out, _ = self.net(x)
            # concat out and views_out
            # B x H x O x head_dim -> B x O x H*head_dim 
            B, Od = views_out.size(0), views_out.size(2)
            views_out = torch.einsum('bhmo->bmho', views_out)
            views_out = views_out.contiguous().view(B, Od, -1)             

            out = torch.cat([out, views_out], dim=1)
        else:
            out = self.net(x)  # B x O x Co

        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=out.size(1),
        ).transpose(1, 2)  # B x 1 x Co
        out = out.squeeze(1)  # B x Co
        x = self.proj_head(out)  # B x Cp
        if self.p_output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],  # a list here
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # input dimension : B x O x Ci
        x = self.input_fc(x)  # B x O x Ch (hidden_dims)
        
        # generate & apply mask, default is binomial
        if mask is None:
            # mask should only use in training phase
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')

        # mask &= nan_masK
        # ~ works as operator.invert
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x O
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x O
        x = x.transpose(1, 2)  # B x O x Co
        
        return x

# @ahf add multi-heads self-attention for TSEncoder
class MultiHeadTSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial',
                 num_heads=2, head_dim=160, channel_dim=320):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],  # a list here
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.MHA = MHA(num_heads, head_dim, channel_dim)

    def forward(self, x, mask=None, view_mask=None):  # input dimension : B x O x Ci
        x = self.input_fc(x)  # B x O x Ch (hidden_dims)
        
        # generate & apply mask, default is binomial
        if mask is None:
            # mask should only use in training phase
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        # @ahf add view masking
        if view_mask is None:
            if self.training:
                view_mask = "continuous"
                view_mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
            else:
                view_mask = 'all_true'
                view_mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')


        x2 = x.clone()

        # mask &= nan_masK
        # ~ works as operator.invert
        x[~mask] = 0
        x2[~view_mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x O
        x = self.feature_extractor(x) # B 64 x Ch 320 x O 256, O is time steps

        # conv encoder
        x2 = x2.transpose(1, 2)  # B x Ch x O
        x2 = self.feature_extractor(x2) # B 64 x Ch 320 x O 256, O is time steps

        # @ahf add multi-heads attention
        x_views = self.MHA(x) # B x H x O x head_dim 
        x_mask_views = self.MHA(x2) # B x H x O x head_dim 

        x = self.repr_dropout(x)  # B x Co x O
        x_views = self.repr_dropout(x_views)
        x_mask_views = self.repr_dropout(x_mask_views)

        x = x.transpose(1, 2)  # B x O x Co
        
        return x, x_views, x_mask_views

# @ahf add Daul multi-heads self-attention for TSEncoder
class DaulMultiHeadTSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial',
                 num_heads=2, head_dim=160, channel_dim=320):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],  # a list here
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.MHA = MHA(num_heads, head_dim, channel_dim)

    def forward(self, x, mask=None, view_mask=None):  # input dimension : B x O x Ci
        x = self.input_fc(x)  # B x O x Ch (hidden_dims)
        
        
        # generate & apply mask, default is binomial
        if mask is None:
            # mask should only use in training phase
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        # @ahf add view masking
        if view_mask is None:
            if self.training:
                view_mask = "continuous"
            else:
                view_mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')

        # x2 = copy.deepcopy(x)
        x2 = x.clone()
        print('--------------encoder2 is used, and deepcopy is used------')
        # x2 = x.detach()

        # mask &= nan_masK
        # ~ works as operator.invert
        x[~mask] = 0
        x2[~view_mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x O
        x = self.feature_extractor(x) # B 64 x Ch 320 x O 256, O is time steps

        # conv encoder
        x2 = x2.transpose(1, 2)  # B x Ch x O
        x2 = self.feature_extractor(x2) # B 64 x Ch 320 x O 256, O is time steps

        # @ahf add multi-heads attention
        x_views = self.MHA(x) # B x H x O x head_dim 
        x_mask_views = self.MHA(x2) # B x H x O x head_dim 

        x = self.repr_dropout(x)  # B x Co x O
        x_views = self.repr_dropout(x_views)
        x_mask_views = self.repr_dropout(x_mask_views)

        x = x.transpose(1, 2)  # B x O x Co
        
        return x, x_views, x_mask_views