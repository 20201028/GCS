"""
条件变分自编码器实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Posterior(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim):
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
    
    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        
        return z

class Prior(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim):
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
            # nn.Linear(latent_dim + condition_dim + hidden_dim + 1, hidden_dim),
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
        z_conditioned = torch.cat([z, condition], dim=-1) # [latent+cond]
        z_conditioned = z_conditioned.unsqueeze(0).expand(node_emb.size(0), -1)
        x = torch.cat([node_emb, z_conditioned], dim=-1)
        pred = self.community_head(x)
        # print(pred)
        pred = pred.squeeze()
        # print(pred)
        return pred

