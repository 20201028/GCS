import torch
import torch.nn as nn
import torch.nn.functional as F
class RealNVPLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask_type='even'):
        super().__init__()
        self.mask_type = mask_type
        # 针对 128 维，输入一半维度 64
        in_dim = dim // 2
        self.s_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim), nn.Tanh()
        )
        self.t_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, z):
        # 简单的维度拆分掩码
        if self.mask_type == 'even':
            z1, z2 = z[::2], z[1::2]
        else:
            z2, z1 = z[::2], z[1::2]

        s = self.s_net(z1)
        t = self.t_net(z1)
        
        # 仿射变换
        z2_new = z2 * torch.exp(s) + t
        log_det_j = torch.sum(s, dim=-1) #标量

        # 合并回原始维度
        z_new = torch.empty_like(z)
        if self.mask_type == 'even':
            z_new[::2], z_new[1::2] = z1, z2_new
        else:
            z_new[::2], z_new[1::2] = z2_new, z1
            
        return z_new, log_det_j, s