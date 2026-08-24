"""
完整模型定义：集成所有组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import StructureEncoder, AttributeEncoder, CrossModalFusion, QueryEncoder
from .FlowVae import GMMPosterior, GMMPrior, Decoder
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
        
        self.posterior = GMMPosterior(
            input_dim=config.hidden_dim,
            condition_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim, 
            n_flows=config.n_flows
        )
        self.prior = GMMPrior(
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
        

        
        struct_emb = self.struct_encoder(
            graph['attr_embeddings'], graph['attributes'], graph['edge_index']
        )
        
        #进行属性编码，一些属性是同义的
        attr_emb, node_att_emb = self.attr_encoder(
            graph['attr_embeddings'], graph['hyperedge_index']
        )
        #学习大多数社区是什么主导的
        node_emb = self.fusion(struct_emb, node_att_emb)
        # node_emb = struct_emb
        

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
            
            z0 = self.posterior.reparameterize(mu, logvar)
            zk, log_det_sum, all_s = self.posterior.flow_forward(z0)
        else:
            # 验证和测试：直接跟随先验走
            query_node_idx = batch['query']['nodes']
            mu, logvar = mu_p, logvar_p # 仅用于 Loss 占位
            z0 = mu_p
            zk, log_det_sum, all_s = self.posterior.flow_forward(z0)
            zk = self.latent_optimizer.optimize(zk, query_encoding, node_emb, query_node_idx, edge_index)

        # ===== 3. Training path (no latent optimization) =====
        query_node_idx = batch['query']['nodes']
        query_indicator = torch.zeros(node_emb.size(0), 1, device=node_emb.device)
        query_indicator[query_node_idx] = 1.0
        community_pred = self.decoder.decode_community(
                zk, query_encoding, node_emb
            )
        
        
        # print(f'mu : {mu}, logvar : {logvar}')
        outputs = {
            'mu': mu,
            'logvar': logvar,
            'mu_p': mu_p,
            'logvar_p': logvar_p,
            'z0': z0,
            'zk': zk,
            'log_det_sum': log_det_sum,
            'all_s' : all_s,
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
        
        # 3. 使用带权重的 BCE Loss
        comm_loss = F.binary_cross_entropy_with_logits(community_pred, target_mask, weight=weight)

        


        mu_q, logvar_q = outputs['mu'], outputs['logvar']
        mu_p, logvar_p = outputs['mu_p'], outputs['logvar_p']
        zk, z0 = outputs['zk'], outputs['z0']
        log_det_sum = outputs['log_det_sum']
        # # 正确计算KL散度
        var_q = torch.exp(logvar_q) + 1e-8  # 避免数值不稳定
        var_p = torch.exp(logvar_p) + 1e-8
        # print(f'var_p: {var_p},logvar_p: {logvar_p},var_q: {var_q},logvar_q: {logvar_q}')
        kl_loss = 0.5 * torch.mean(
            logvar_p - logvar_q +
            (var_q + (mu_q - mu_p).pow(2)) / var_p -
            1
        )
        target_kl = self.config.kl_threshold
        kl_loss1 = torch.max(kl_loss, torch.tensor(target_kl).to(kl_loss.device))
        # # 强制 KL 损失至少为 0.1 ~ 0.2
        # # 这样即便后期权重加大，模型也会保持 logvar_p 和 logvar_q 的距离   
        kl_loss = self.compute_flow_kl(zk, z0, mu_q, logvar_q, mu_p, logvar_p, log_det_sum, (self.config.latent_dim//2) * self.config.n_flows)
        # log_q0 = -0.5 * torch.sum(logvar_q + torch.log(torch.tensor(2 * np.pi).to(z0.device)) + ((z0 - mu_q) ** 2) / var_q)
        # nvp_loss = log_q0 + log_det_sum/(self.config.latent_dim//2) * self.config.n_flows

        # 核心惩罚项：对所有流层的缩放强度进行 L2 正则
        all_s = outputs['all_s']
        s_penalty = torch.sum(torch.stack([torch.mean(s**2) for s in all_s]))

        # zk_align = F.mse_loss(zk_prior, anchor, reduction='mean')


        # 总损失
        total_loss = (
            comm_loss
            # + min(1.0, current_epoch / self.config.kl_annealing) * self.config.beta_kl * kl_loss 
            +  kl_loss1
            +  kl_loss
            # + 0.4 * s_penalty
            # + s_penalty
            # + nvp_loss
            # + self.config.beta_constra * constra_loss
            # + self.config.beta_query * query_loss
            # + self.config.beta_supcon * SupCon_loss
        )
        loss_dict = {
            'total_loss': total_loss.item(),
            'comm_loss': comm_loss.item(),
            'kl_loss': kl_loss.item(),
            # 'balanced_loss': balanced_loss.item()
        }
        
        return total_loss, loss_dict
    # def compute_flow_kl(self, zk, z0, mu_q, logvar_q, mu_p, logvar_p, log_det_sum):
    #     """
    #     正确的 Flow-enhanced CVAE KL 损失计算
    #     直接在 z0 (高斯基底空间) 计算相对散度，并结合后验流的 log_det 修正
    #     """
    #     var_q = torch.exp(logvar_q) + 1e-8
    #     var_p = torch.exp(logvar_p) + 1e-8
        
    #     # 1. 计算解析形式的条件高斯 KL 散度 (z0 空间的相对距离)
    #     kl_raw = 0.5 * torch.mean(
    #         logvar_p - logvar_q +
    #         (var_q + (mu_q - mu_p).pow(2)) / var_p -
    #         1.0
    #     )
        
    #     # 2. 减去后验流的熵减修正项 (log_det_sum 表示后验从高斯球展开为复杂流形时的体积放大率)
    #     # 因为 log q(zk) = log q0(z0) - log_det_sum，所以对应的散度需要减去平均行列式值
    #     total_kl = kl_raw - torch.mean(log_det_sum)
        
    #     # 3. 施加你之前被验证行之有效的 Free Bits 保护盾
    #     target_kl = self.config.kl_threshold 
    #     total_kl = torch.max(total_kl, torch.tensor(target_kl).to(total_kl.device))
        
    #     return total_kl
    def compute_flow_kl(self, zk, z0, mu_q, logvar_q, mu_p, logvar_p, log_det_sum, dim):
        """
        正确的 Flow-enhanced CVAE KL 损失计算
        直接在 z0 (高斯基底空间) 计算相对散度，并结合后验流的 log_det 修正
        """
        var_q = torch.exp(logvar_q) + 1e-8
        var_p = torch.exp(logvar_p) + 1e-8
        
        # 1. 计算解析形式的条件高斯 KL 散度 (z0 空间的相对距离)
        kl_raw = 0.5 * torch.mean(
            logvar_p - logvar_q +
            (var_q + (mu_q - mu_p).pow(2)) / var_p -
            1.0
        )
        # print(f'kl_raw:{kl_raw.item()}')
        # print(f'log_det_sum mean: {log_det_sum /dim}')
        # print(f'log_det_sum sum: {log_det_sum}')
        # target_kl = self.config.kl_threshold 
        # kl_raw = torch.max(kl_raw, torch.tensor(target_kl).to(kl_raw.device))
        # 2. 减去后验流的熵减修正项 (log_det_sum 表示后验从高斯球展开为复杂流形时的体积放大率)
        # 因为 log q(zk) = log q0(z0) - log_det_sum，所以对应的散度需要减去平均行列式值
        # total_kl = kl_raw - torch.mean(log_det_sum)
        total_kl = kl_raw + log_det_sum/dim
        # total_kl = log_det_sum/dim
        # total_kl = kl_raw + log_det_sum
        
        # 3. 施加你之前被验证行之有效的 Free Bits 保护盾
        
        
        return total_kl
    
