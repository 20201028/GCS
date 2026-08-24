"""

条件变分自编码器实现

"""

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

        # z2_new = z2 * torch.exp(s) + t
        z2_new = (z2-t) * torch.exp(-s)

        log_det_j = torch.sum(-s, dim=-1) #标量



        # 合并回原始维度

        z_new = torch.empty_like(z)

        if self.mask_type == 'even':

            z_new[::2], z_new[1::2] = z1, z2_new

        else:

            z_new[::2], z_new[1::2] = z2_new, z1

           

        return z_new, log_det_j, s


class GMMPosterior(nn.Module):

    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim, n_flows):

        super().__init__()

        self.latent_dim = latent_dim

        # 编码器

        self.encoder = nn.Sequential(

            nn.Linear(4*hidden_dim, hidden_dim),

            nn.ReLU()

        )

        # 潜变量分布参数

        self.mu_head = nn.Linear(1*hidden_dim, latent_dim)

        self.logvar_head = nn.Linear(1*hidden_dim, latent_dim)

        # 关键：添加正态流层

        self.flows = nn.ModuleList([

            RealNVPLayer(latent_dim, hidden_dim, mask_type='even' if i%2==0 else 'odd')

            for i in range(n_flows)

        ])

   

    def posterior(self, x, condition):

        """编码输入和条件到潜变量分布"""

        h_mean = torch.mean(x, dim=0)  # 全局池化

        h_max = torch.max(x, dim=0)[0]   # 形状 []

        h_min = torch.min(x, dim=0)[0]

        # 拼接后得到更丰富的社区表示

        h = torch.cat([h_mean, h_max, h_min, condition], dim=-1)

       

        h = self.encoder(h)

        mu = self.mu_head(h)

        logvar = self.logvar_head(h)

        return mu, logvar
    def flow_forward(self, z0):
        all_s = []

        # 3. 通过正态流 (z0 -> zk)

        zk = z0

        log_det_sum = 0

        for flow in self.flows:

            zk, log_det_j, s = flow(zk)

            log_det_sum += log_det_j

            all_s.append(s)

       

        return zk, log_det_sum, all_s
    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        
        return z
    
class GMMPrior(nn.Module):

    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim, n_flows=1):

        super().__init__()

        self.latent_dim = latent_dim

        self.prior_net = nn.Sequential(

        nn.Linear(3*input_dim + condition_dim, hidden_dim),

        nn.ReLU())

        # 潜变量分布参数

        self.mu_head = nn.Linear(hidden_dim, latent_dim)

        self.logvar_head = nn.Linear(hidden_dim, latent_dim)



    def prior(self, condition, x):

        h_mean = torch.mean(x, dim=0)  # 全局池化

        h_max = torch.max(x, dim=0)[0]   # 形状 []

        h_min = torch.min(x, dim=0)[0]

        # 拼接后得到更丰富的社区表示

        h = torch.cat([h_mean, h_max, h_min, condition], dim=-1)

        h = self.prior_net(h)

        mu = self.mu_head(h)

        logvar = self.logvar_head(h)

        return mu, logvar

class Decoder(nn.Module):

    def __init__(self, condition_dim, latent_dim, hidden_dim, dropout):

        super().__init__()

       

        # # 解码器

        self.community_head = nn.Sequential(

            nn.Linear(latent_dim + condition_dim + hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.LayerNorm(hidden_dim),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.LayerNorm(hidden_dim),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),

        )

    def decode_community(self, z, condition, node_emb):

        """解码社区概率"""

        # 2. 拼接 Z 和 Condition

        z_conditioned = torch.cat([z, condition], dim=-1) # [latent+cond]

        z_conditioned = z_conditioned.unsqueeze(0).expand(node_emb.size(0), -1)

        x = torch.cat([node_emb, z_conditioned], dim=-1)

        pred = self.community_head(x)

        # print(pred)

        pred = pred.squeeze()

        # print(pred)

        return pred 

