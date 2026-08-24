"""
完整模型定义：集成所有组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import StructureEncoder, AttributeEncoder, CrossModalFusion, QueryEncoder
from .vae import Posterior, Prior, Decoder
from .optimizer import LatentSpaceOptimizer
import numpy as np

class LSOCommunitySearch(nn.Module):
    """完整的LSO-CS模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # === 编码器组件 ===
        self.struct_encoder = StructureEncoder(
            in_dim=config.node_feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            dropout=config.dropout_rate
        )
        
        self.attr_encoder = AttributeEncoder(
            in_dim=config.node_feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            dropout=config.dropout_rate
        )
        
        self.fusion = CrossModalFusion(
            struct_dim=config.hidden_dim,
            attr_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout_rate
        )
        
        self.query_encoder = QueryEncoder(
            struct_dim=config.hidden_dim,
            attr_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout_rate
        )
        
        self.posterior = Posterior(
            input_dim=config.hidden_dim,
            condition_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim
        )
        self.prior = Prior(
            input_dim=config.hidden_dim,
            condition_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim
        )
        self.decoder = Decoder(
            condition_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout_rate
        )
        
        # === 优化器和后处理器 ===
        self.latent_optimizer = LatentSpaceOptimizer(self, config)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    
    def forward(self, batch, mode='train', attr = True):
        graph = batch['graph']
        query = batch['query']
        edge_index = graph['edge_index']

        # ===== 1. Encode graph & query =====
        #学习社区内节点在图中的结构是什么样的
        

        # struct_emb = self.struct_encoder(
        #     node_ench_emb, graph['edge_index']
        # )
        struct_emb = self.struct_encoder(
            graph['attr_embeddings'], graph['attributes'], graph['edge_index']
        )
        # struct_emb = self.struct_encoder(
        #     graph['node_embeddings'], graph['edge_index']
        # )
        #进行属性编码，一些属性是同义的
        attr_emb, node_att_emb = self.attr_encoder(
            graph['attr_embeddings'], graph['hyperedge_index']
        )
        #学习大多数社区是什么主导的
        node_emb = self.fusion(struct_emb, node_att_emb)
        

        if attr:
            query_encoding = self.query_encoder(
                struct_emb,
                query['nodes'],
                attr_emb,
                query['attrs']
            )
        else:
            query_encoding = self.query_encoder(
                struct_emb,
                query['nodes']
            )
        # query_encoding = self.query_encoder(
        #     node_emb,
        #     query['nodes'],
        #     attr_emb,
        #     query['attrs']
        # )

        # 测量全局编码器
        # g_emb = torch.mean(node_emb, dim=0)

        # ===== 2. Posterior & prior (always computable) =====
        # mu_p, logvar_p = self.prior.prior(query_encoding, g_emb)
        mu_p, logvar_p = self.prior.prior(query_encoding, node_emb)
        if mode == 'train':
            # A修改的核心：只取正样本计算后验
            target_mask = batch['target']['community_mask']
            pos_node_emb = node_emb[target_mask == 1]
            
            #给定查询所在社区的分布
            mu, logvar = self.posterior.posterior(pos_node_emb, query_encoding)
            
            z = self.posterior.reparameterize(mu, logvar)
        else:
            # 验证和测试：直接跟随先验走
            query_node_idx = batch['query']['nodes']
            z = mu_p
            # query_node_emb = node_emb[query_node_idx]
            
            # # 计算查询点的“局部后验”
            # mu_local, _ = self.posterior.posterior(query_node_emb, query_encoding)
            
            # # 融合全局先验和局部后验，权重可调 (例如 0.2:0.8)
            # z = self.config.alpha * mu_p + self.config.beta * mu_local
            
            # z = mu_local
            mu, logvar = mu_p, logvar_p # 仅用于 Loss 占位
            z = self.latent_optimizer.optimize(z, query_encoding, node_emb, query_node_idx, edge_index)

        # ===== 3. Training path (no latent optimization) =====
        query_node_idx = batch['query']['nodes']
        query_indicator = torch.zeros(node_emb.size(0), 1, device=node_emb.device)
        query_indicator[query_node_idx] = 1.0
        community_pred = self.decoder.decode_community(
                z, query_encoding, node_emb
            )
        
        
        # print(f'mu : {mu}, logvar : {logvar}')
        outputs = {
            'mu': mu,
            'logvar': logvar,
            'mu_p': mu_p,
            'logvar_p': logvar_p,
            'z': z,
            'query_encoding': query_encoding,
            'community_pred': community_pred,
            'struct_emb' : struct_emb, 
            'node_att_emb' : node_att_emb,
            'node_emb' : node_emb
        }

        return outputs
    
    def compute_loss(self, outputs, batch, current_epoch):
        # 1. 二值交叉熵
        target_mask = batch['target']['community_mask']
        query_emb = outputs['query_encoding']
        node_emb = outputs['node_emb']
        edge_index = batch['graph']['edge_index']
        community_pred = outputs['community_pred']
        community_indices = batch['target']['community_indices']
        struct_emb = outputs['struct_emb']
        attr_emb = outputs['node_att_emb']
        query_node_idx = batch['query']['nodes']

    
        
        # 1. 创建权重矩阵
        # 基础权重：负样本 1.0, 正样本 2.0 (解决正负样本不平衡)
        weight = torch.ones_like(target_mask)
        weight[target_mask == 1] = (target_mask == 0).sum() / (target_mask == 1).sum()
        
        # 2. 【核心改进】：给查询点极高的权重
        # 哪怕查询点只有一个，它的错误也要顶得上几十个普通节点
        weight[query_node_idx] = (target_mask == 0).sum() / len(query_node_idx)
        # print(weight.shape)
        # 3. 使用带权重的 BCE Loss
        comm_loss = F.binary_cross_entropy_with_logits(community_pred, target_mask, weight=weight)

        


        mu_q, logvar_q = outputs['mu'], outputs['logvar']
        mu_p, logvar_p = outputs['mu_p'], outputs['logvar_p']
        
        # 正确计算KL散度
        var_q = torch.exp(logvar_q) + 1e-8  # 避免数值不稳定
        var_p = torch.exp(logvar_p) + 1e-8
        # print(f'var_p: {var_p},logvar_p: {logvar_p},var_q: {var_q},logvar_q: {logvar_q}')
        kl_loss = 0.5 * torch.mean(
            logvar_p - logvar_q +
            (var_q + (mu_q - mu_p).pow(2)) / var_p -
            1
        )
        # 强制 KL 损失至少为 0.1 ~ 0.2
        # 这样即便后期权重加大，模型也会保持 logvar_p 和 logvar_q 的距离
        target_kl = self.config.kl_threshold 
        kl_loss = torch.max(kl_loss, torch.tensor(target_kl).to(kl_loss.device))
       

        N, d = struct_emb.shape
    
        # L2归一化
        struct_emb_norm = F.normalize(struct_emb, dim=1)
        attr_emb_norm = F.normalize(attr_emb, dim=1)
        
        # 计算所有节点对之间的相似度矩阵 [N, N]
        # sim[i,j] = 节点i的结构与节点j的属性的相似度
        
        sim_matrix = torch.matmul(struct_emb_norm, attr_emb_norm.T) / self.config.temperature
        
        # InfoNCE损失（结构到属性方向）
        # 正对是 sim_matrix[i,i]
        labels = torch.arange(N, device=struct_emb.device)
        loss_s2a = F.cross_entropy(sim_matrix, labels)
        
        
        # # 属性到结构方向
        # sim_matrix_t = sim_matrix.T
        # loss_a2s = F.cross_entropy(sim_matrix_t, labels)
        # loss_s2a = loss_a2s
        # loss_a2s = loss_s2a
        # 平均损失
        # constra_loss = (loss_s2a + loss_a2s) / 2
        constra_loss = loss_s2a

        # 1. 分离正负样本嵌入
        pos_mask = (target_mask == 1)
        neg_mask = (target_mask == 0)
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0).to(node_emb.device)

        pos_embs = node_emb[pos_mask]  # [N_pos, D]
        neg_embs = node_emb[neg_mask]  # [N_neg, D]
        query_emb = query_emb.unsqueeze(0)
        # 2. 计算节点到查询锚点(Anchor)的欧式距离
        # 使用 torch.cdist 计算成对距离，结果形状 [N_pos, 1] 和 [N_neg, 1]
        dist_pos = torch.cdist(pos_embs, query_emb, p=2) 
        dist_neg = torch.cdist(neg_embs, query_emb, p=2)

        # 3. 采样策略：为了 Precision，我们必须关注“难负样本”
        # 即：那些离查询非常近（dist_neg 小）的背景节点
        # 取 dist_neg 最小的前 K 个样本进行惩罚
        num_hard_negs = min(len(dist_neg), len(dist_pos) * self.config.K) # 动态采样
        hard_dist_neg, _ = torch.topk(dist_neg, k=num_hard_negs, dim=0, largest=False)

        # 4. Triplet Loss 公式: L = max(0, dist_pos - dist_neg + margin)
        # 我们希望 dist_pos 越小越好，dist_neg 越大越好，且两者差距至少为 margin
        # margin = 2.0
        query_loss = torch.relu(dist_pos.mean() - hard_dist_neg.mean() + self.config.margin)  

        # 只取社区内的节点和部分负样本，防止计算量爆炸
        pos_idx = torch.where(target_mask == 1)[0]
        neg_idx = torch.where(target_mask == 0)[0]
        
        # 采样一部分负样本以平衡计算
        if len(neg_idx) > len(pos_idx) * 2:
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:len(pos_idx) * 2]]
            
        combined_idx = torch.cat([pos_idx, neg_idx])
        features = F.normalize(node_emb[combined_idx], dim=1)
        labels = target_mask[combined_idx]
        
        # 计算相似度矩阵 [M, M]
        logits = torch.matmul(features, features.T) / self.config.temperature
        
        # 构造 Mask：同一社区的节点互为正样本
        # labels.unsqueeze(0) == labels.unsqueeze(1) 得到 [M, M] 的布尔矩阵
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(node_emb.device)
        # 移除自身对比
        mask = mask * (1 - torch.eye(len(combined_idx)).to(node_emb.device))
        
        # 对每行进行归一化并计算对数似然
        exp_logits = torch.exp(logits) * (1 - torch.eye(len(combined_idx)).to(node_emb.device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # 计算均值损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        SupCon_loss = -mean_log_prob_pos.mean()

        # 总损失
        total_loss = (
            comm_loss
            # + min(1.0, current_epoch / self.config.kl_annealing) * self.config.beta_kl * kl_loss 
            + kl_loss
            # + self.config.beta_constra * constra_loss
            # + self.config.beta_query * query_loss
            # + self.config.beta_supcon * SupCon_loss
        )
        # print(f'comm_loss:{comm_loss.item()}, kl_loss:{kl_loss.item()}')
        loss_dict = {
            'total_loss': total_loss.item(),
            'comm_loss': comm_loss.item(),
            'kl_loss': kl_loss.item(),
            # 'balanced_loss': balanced_loss.item()
        }
        
        return total_loss, loss_dict
    
