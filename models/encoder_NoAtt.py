"""
多模态图编码器实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv
import torch_scatter

class StructureEncoder(nn.Module):
    """结构编码器：使用GCN提取结构信息"""
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.selfLoop = nn.ModuleList()
        # 输入层
        self.layers.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.selfLoop.append(nn.Linear(in_dim, hidden_dim))
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(
                hidden_dim, hidden_dim
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.selfLoop.append(nn.Linear(in_dim, hidden_dim))
        
        # 输出层
        self.layers.append(GCNConv(
            hidden_dim, hidden_dim
        ))
        self.selfLoop.append(nn.Linear(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
    def forward(self, attr_emb, nodes_feats, edge_index):
        if isinstance(nodes_feats, np.ndarray):
            nodes_feats = torch.from_numpy(nodes_feats).float()
            nodes_feats = nodes_feats.to(edge_index.device)
        
        # 1. 计算每个属性的出现频率（在整个图中出现的次数）
        # nodes_feats 是 [N, M]，每列是一个属性在所有节点上的出现情况
        attr_freq = nodes_feats.sum(dim=0)  # [M]
        
        # 2. 计算逆频率权重 (Inverse Frequency Weighting)
        # 出现越多的属性，权重越小
        # 使用 log 变换平滑，类似 IDF
        importance_weights = torch.log(len(nodes_feats) / (attr_freq + 1.0))  # [M]
        
        # 3. 将重要性权重应用到属性嵌入上
        # importance_weights: [M] -> [M, 1] 以便广播
        weighted_attr_emb = attr_emb * importance_weights.unsqueeze(1)  # [M, D]
        
        # 4. 聚合每个节点的属性嵌入（加权平均）
        # nodes_feats: [N, M] 作为权重，与加权属性嵌入相乘
        # 每个节点的嵌入 = sum(属性权重 * 属性嵌入) / sum(属性权重)
        node_emb = torch.mm(nodes_feats, weighted_attr_emb)  # [N, D]
        
        # 归一化：除以每个节点的属性数量（避免节点间因属性数量不同而尺度不一）
        attr_counts = nodes_feats.sum(dim=1, keepdim=True)  # [N, 1]
        x = node_emb / attr_counts # [N, D]
        
        x_0 = x.clone() # 保持初始特征
        i = 0
        for conv in self.layers[:-1]:
            x = F.relu(conv(x, edge_index)) + self.selfLoop[i](x_0)
            x = self.norms[i](x)
            x = self.dropout(x)
            i += 1
        x = F.relu(self.layers[-1](x, edge_index)) + self.selfLoop[-1](x_0)
        x= self.norms[-1](x)
        x = self.dropout(x)
        return x
    
class AttributeEncoder(nn.Module):
    """
    封装超图操作，直接处理组嵌入
    """
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        
        # 超图卷积层
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.selfLoop = nn.ModuleList()
        # 输入层
        self.layers.append(HypergraphConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.selfLoop.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(HypergraphConv(
                hidden_dim, hidden_dim
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.selfLoop.append(nn.Linear(in_dim, hidden_dim))

        self.layers.append(HypergraphConv(
            hidden_dim, hidden_dim
        ))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.selfLoop.append(nn.Linear(in_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        
    def compute_hyperedge_embeddings(self, x, hyperedge_index):
        """
        x: [num_attributes, dim] - 经过 HyperConv 更新后的属性嵌入
        hyperedge_index: [2, num_connections] 
            - row: 属性索引 (Source Nodes in Hypergraph)
            - col: 原图节点索引 (Hyperedges in Hypergraph)
        """
        # 1. 统计每个属性（超图节点）的度 (Degree)
        # 这里的度表示：该属性被多少个原图节点所拥有
        attr_index = hyperedge_index[0]
        node_index = hyperedge_index[1]
        
        # 计算每个属性的出现频率
        attr_degrees = torch.zeros(x.size(0), device=x.device)
        ones = torch.ones_like(attr_index, dtype=torch.float)
        attr_degrees.scatter_add_(0, attr_index, ones) # [num_attributes]
        
        # 2. 计算属性的“重要性权重” (类似 IDF)
        # 度越大的属性，分配的权重越小，防止其信息淹没其他特征
        # 增加 epsilon 防止除零，使用 log 或倒数平滑
        num_hyperedges = node_index.max() + 1
        # 权重公式：log(N / (1 + degree))
        attr_weights = torch.log(num_hyperedges / (1.0 + attr_degrees)) 
        attr_weights = torch.clamp(attr_weights, min=0.1) # 保证基础权重
        
        # 3. 将权重应用到属性嵌入上
        weighted_x = x * attr_weights.unsqueeze(-1) # [num_attributes, dim]

        
        # 使用scatter_mean一次性计算
        hyperedge_emb = torch_scatter.scatter_mean(
            weighted_x[attr_index],  # 源数据
            node_index,                   # 索引
            dim=0,                           # 聚合维度
            dim_size=node_index.max().item() + 1  # 输出大小
        )
        return hyperedge_emb    
    
    def forward(self, x, hyperedge_index):
        x_0 = x.clone() # 保持初始特征
        # 超图卷积更新节点嵌入
        i = 0
        for conv in self.layers[:-1]:
            x = F.relu(conv(x, hyperedge_index)) + self.selfLoop[i](x_0)
            x = self.norms[i](x)
            x = self.dropout(x)
            i += 1
        # x = self.layers[-1](x, hyperedge_index) + self.selfLoop[-1](x_0)
        x = F.relu(self.layers[-1](x, hyperedge_index)) + self.selfLoop[-1](x_0)
        x = self.norms[-1](x)
        x = self.dropout(x)
        # 计算超边嵌入
        hyperedge_emb = self.compute_hyperedge_embeddings(x, hyperedge_index)
        
        # 如果指定了特定的组，返回对应的超边嵌入
        # return x
        return x, hyperedge_emb

class CrossModalFusion(nn.Module):
    def __init__(self, struct_dim, attr_dim, hidden_dim, dropout):
        super().__init__()
        # 结构专家和属性专家
        self.struct_gate = nn.Linear(struct_dim, 1)
        self.attr_gate = nn.Linear(attr_dim, 1)
        
    def forward(self, s_emb, a_emb):
        # 计算当前节点/社区对两种模态的“敏感度”
        s_score = self.struct_gate(s_emb)
        a_score = self.attr_gate(a_emb)
        # print(s_score)
        # 归一化得到驱动力分布
        weights = torch.softmax(torch.cat([s_score, a_score], dim=-1), dim=-1)
        # print(weights)
        # 记录这个 weights，它就是你想要的“驱动力指标”
        # weights[:, 0] 越高，代表结构驱动；weights[:, 1] 越高，代表属性驱动
        fused = weights[:, 0:1] * s_emb + weights[:, 1:2] * a_emb
        # fused = s_emb + a_emb
        return fused
# class CrossModalFusion(nn.Module):
#     def __init__(self, struct_dim, attr_dim, hidden_dim, dropout):
#         super().__init__()
#         # 结构专家和属性专家
#         self.alpha_net = nn.Sequential(
#             nn.Linear(2*hidden_dim, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, s_emb, a_emb):
#         # 计算当前节点/社区对两种模态的“敏感度”
#         # print(s_score)
#         # 归一化得到驱动力分布
#         weights = self.alpha_net(torch.cat([s_emb, a_emb], dim=-1))
#         # print(weights)
#         # 记录这个 weights，它就是你想要的“驱动力指标”
#         # weights[:, 0] 越高，代表结构驱动；weights[:, 1] 越高，代表属性驱动
#         fused = weights * s_emb + (1 - weights) * a_emb
#         return fused
# class CrossModalFusion(nn.Module):
#     def __init__(self, struct_dim, attr_dim, hidden_dim, dropout):
#         super().__init__()
#         # 结构专家和属性专家
#         self.alpha_net = nn.Sequential(
#             nn.Linear(2*hidden_dim, hidden_dim),
#             nn.Sigmoid()
#         )
#         self.struct_net = nn.Linear(hidden_dim, hidden_dim)
#         self.attr_net = nn.Linear(hidden_dim, hidden_dim)
#     def forward(self, s_emb, a_emb):
#         # 计算当前节点/社区对两种模态的“敏感度”
#         query_node_encoding = self.struct_net(s_emb)
#         query_attr_encoding = self.attr_net(a_emb)
#         alpha = self.alpha_net(torch.cat([query_node_encoding, query_attr_encoding], dim=-1))
#         fused = alpha * s_emb + (1 - alpha) * a_emb
#         return fused
# class CrossModalFusion(nn.Module):
#     def __init__(self, struct_dim, attr_dim, hidden_dim, dropout):
#         super().__init__()
#         # self.alpha_net = nn.Sequential(
#         #     nn.Linear(struct_dim + attr_dim, hidden_dim),
#         #     nn.Sigmoid()
#         # )
#         self.alpha_net = nn.Sequential(
#             nn.Linear(struct_dim + attr_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Sigmoid() 
#         )
#     def forward(self, struct_emb, attr_emb):
#         print(struct_emb.shape, attr_emb.shape)
#         alpha = self.alpha_net(torch.cat([struct_emb, attr_emb], dim=-1))
#         # print(alpha)
#         fused = alpha * struct_emb + (1 - alpha) * attr_emb
#         return fused
#         # return alpha
    
# class QueryEncoder(nn.Module):
#     """查询编码器：编码查询节点和属性"""
#     def __init__(self, struct_dim, attr_dim, hidden_dim, dropout):
#         super().__init__()
#         self.alpha_net = nn.Sequential(
#             nn.Linear(struct_dim + attr_dim, hidden_dim),
#             nn.Sigmoid()
#         )
#         # self.alpha_net = nn.Sequential(
#         #     nn.Linear(struct_dim + attr_dim, hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.Sigmoid() 
#         # )
#         self.struct_net = nn.Linear(struct_dim, hidden_dim)
#         self.attr_net = nn.Linear(attr_dim, hidden_dim)
#     def forward(self, node_embeddings, query_nodes, attr_embeddings, query_attrs):
#         # 编码查询节点
#         query_node_embs = node_embeddings[query_nodes]
#         query_node_encoding = torch.mean(query_node_embs, dim=0)
#         # query_node_encoding = self.struct_net(query_node_encoding)
#         # 编码查询属性
#         query_attr_embs = attr_embeddings[query_attrs]
#         query_attr_encoding = torch.mean(query_attr_embs, dim=0)
#         # query_attr_encoding = self.attr_net(query_attr_encoding)
#         # 合并
#         alpha = self.alpha_net(torch.cat([query_node_encoding, query_attr_encoding], dim=-1))
#         fused = alpha * query_node_encoding + (1 - alpha) * query_attr_encoding
#         return fused
class QueryEncoder(nn.Module):
    """查询编码器：编码查询节点和属性"""
    def __init__(self, struct_dim, hidden_dim, dropout):
        super().__init__()
        self.struct_net = nn.Linear(3*struct_dim, hidden_dim)
    def forward(self, node_embeddings, query_nodes):
        # 编码查询节点
        query_node_embs = node_embeddings[query_nodes]
        if query_node_embs.dim() == 1:
            query_node_embs=query_node_embs.reshape(1, -1)
        query_node_mean = torch.mean(query_node_embs, dim=0)
        query_node_max = torch.max(query_node_embs, dim=0)[0]   # 形状 []
        query_node_min = torch.min(query_node_embs, dim=0)[0]

        # 拼接后得到更丰富的表示
        query_node_encoding = torch.cat([query_node_mean, query_node_max, query_node_min], dim=-1)
        query_node_encoding = self.struct_net(query_node_encoding)
        fused = query_node_encoding
        return fused
    
# class QueryEncoder(nn.Module):
#     """查询编码器：使用双专家 + Softmax（维度级向量权重）进行融合"""
#     def __init__(self, struct_dim, attr_dim, hidden_dim, dropout):
#         super().__init__()
#         self.struct_net = nn.Linear(3*struct_dim, hidden_dim)
#         self.attr_net = nn.Linear(3*attr_dim, hidden_dim)
        
#         # 【双专家设计 - 输出维度为 hidden_dim】
#         self.struct_gate = nn.Linear(hidden_dim, hidden_dim)
#         self.attr_gate = nn.Linear(hidden_dim, hidden_dim)
        
#     def forward(self, node_embeddings, query_nodes, attr_embeddings, query_attrs):
#         # 1. 编码查询节点与属性
#         query_node_embs = node_embeddings[query_nodes]
#         query_node_encoding = torch.cat([torch.mean(query_node_embs, dim=0), torch.max(query_node_embs, dim=0)[0], torch.min(query_node_embs, dim=0)[0]], dim=-1)
#         query_node_encoding = self.struct_net(query_node_encoding)
        
#         query_attr_embs = attr_embeddings[query_attrs]
#         query_attr_encoding = torch.cat([torch.mean(query_attr_embs, dim=0), torch.max(query_attr_embs, dim=0)[0], torch.min(query_attr_embs, dim=0)[0]], dim=-1)
#         query_attr_encoding = self.attr_net(query_attr_encoding)
        
#         # 2. 维度级评分
#         s_scores = self.struct_gate(query_node_encoding).unsqueeze(0) # [1, hidden_dim]
#         a_scores = self.attr_gate(query_attr_encoding).unsqueeze(0)   # [1, hidden_dim]
        
#         # 3. 沿模态维度（dim=0）做 Softmax 竞争
#         # 保证在每一个特征维度上，结构权重 + 属性权重 = 1
#         scores_cat = torch.cat([s_scores, a_scores], dim=0) # [2, hidden_dim]
#         weights = torch.softmax(scores_cat, dim=0)          # [2, hidden_dim]
        
#         # 4. 维度级加权融合
#         fused = weights[0] * query_node_encoding + weights[1] * query_attr_encoding
#         return fused
# class QueryEncoder(nn.Module):
#     """查询编码器：编码查询节点和属性"""
#     def __init__(self, struct_dim, attr_dim, hidden_dim, dropout):
#         super().__init__()
#         self.struct_gate = nn.Linear(hidden_dim, 1)
#         self.attr_gate = nn.Linear(hidden_dim, 1)
#         self.alpha_net = nn.Sequential(
#             nn.Linear(2*hidden_dim, hidden_dim),
#             nn.Sigmoid()
#         )
#         # self.alpha_net = nn.Sequential(
#         #     nn.Linear(struct_dim + attr_dim, hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.Sigmoid() 
#         # )
#         self.struct_net = nn.Linear(3*struct_dim, hidden_dim)
#         self.attr_net = nn.Linear(3*attr_dim, hidden_dim)
#     def forward(self, node_embeddings, query_nodes, attr_embeddings, query_attrs):
#         # 编码查询节点
#         query_node_embs = node_embeddings[query_nodes]
#         query_node_mean = torch.mean(query_node_embs, dim=0)
#         query_node_max = torch.max(query_node_embs, dim=0)[0]   # 形状 []
#         query_node_min = torch.min(query_node_embs, dim=0)[0]

#         # 拼接后得到更丰富的表示
#         query_node_encoding = torch.cat([query_node_mean, query_node_max, query_node_min], dim=-1)
#         query_node_encoding = self.struct_net(query_node_encoding)
#         # 编码查询属性
#         query_attr_embs = attr_embeddings[query_attrs]
#         query_attr_mean = torch.mean(query_attr_embs, dim=0)
#         query_attr_max = torch.max(query_attr_embs, dim=0)[0]   # 形状 []
#         query_attr_min = torch.min(query_attr_embs, dim=0)[0]

#         # 拼接后得到更丰富的表示
#         query_attr_encoding = torch.cat([query_attr_mean, query_attr_max, query_attr_min], dim=-1)
#         query_attr_encoding = self.attr_net(query_attr_encoding)
#         # 计算当前节点/社区对两种模态的“敏感度”
#         s_score = self.struct_gate(query_node_encoding)
#         a_score = self.attr_gate(query_attr_encoding)
#         # print(s_score)
#         # 归一化得到驱动力分布
#         weights = torch.softmax(torch.cat([s_score, a_score], dim=-1), dim=-1)
#         # print(weights)
#         # 记录这个 weights，它就是你想要的“驱动力指标”
#         # weights[:, 0] 越高，代表结构驱动；weights[:, 1] 越高，代表属性驱动
#         fused = weights[0:1] * query_node_encoding + weights[1:2] * query_attr_encoding
#         return fused
        
