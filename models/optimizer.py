"""
潜空间优化器实现
"""
import torch
import torch.nn.functional as F

class LatentSpaceOptimizer:
    """潜空间优化器"""
    def __init__(self, model, config):
        self.model = model
        self.lr = config.latent_lr
        self.latent_reg  = config.latent_reg
        self.steps = config.latent_steps
        self.latent_condu = config.latent_condu
        self.latent_volume = config.latent_volume
        self.prob_threshold = config.prob_threshold

    def optimize(self, initial_z, condition, node_emb, query_nodes, edge_index):
        """优化潜变量"""
        with torch.enable_grad():
            z = initial_z.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([z], lr=self.lr)

            for step in range(self.steps):

                optimizer.zero_grad()
                # 前向
                query_indicator = torch.zeros(node_emb.size(0), 1, device=node_emb.device)
                query_indicator[query_nodes] = 1.0
                y_hat = self.model.decoder.decode_community(z, condition, node_emb)
                probs = torch.sigmoid(y_hat)
                # 查询节点包含度评分

                query_node_score = torch.log(probs[query_nodes] + 1e-8).mean()


                # 正则项防止 z 跑得离先验太远 (KL 正则)
                reg_loss = F.mse_loss(z, initial_z)

                # threshold = 0.1
                high_mask = (probs >= self.prob_threshold).float().detach()
                low_mask = (probs < self.prob_threshold).float().detach()
                
                num_high = high_mask.sum()
                num_low = low_mask.sum()
                ratio_penalty = num_high / (num_low + 1e-6)
                # loss = -query_node_score + 1 * reg_loss + 0.5 * ratio_penalty
                loss = -query_node_score + self.latent_reg * reg_loss + self.latent_volume * ratio_penalty
            
                # 检查计算图
                loss.backward()

                optimizer.step()


        return z.detach()
    