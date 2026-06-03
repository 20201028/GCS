import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import os

class AttributeAugmentedGraph:
    """
    属性增强图构建和预训练类
    对应论文中的 GA = (V∪VA, E∪EA)
    """
    
    def __init__(self, data_dir, dataset):
        """
        初始化属性增强图
        
        参数:
            graph: 原始图 G = (V, E, A)
            attributes: 节点属性字典，{节点ID: [属性列表]}
        """
        self.data_dir = data_dir
        self.dataset = dataset
        self.original_graph = None
        self.node_num = 0
        self.id_mapping = None
        self.reverse_mapping = None
        self.attributes = {}
        self.feat_num = 0
        # self.attribute_nodes = {}  # 属性节点映射：属性ID -> 属性节点ID
        # self.reverse_attribute_nodes = {}  # 反向映射
        self.attribute_augmented_graph = None
        self.node_embeddings = None
        # self.attribute_embeddings = {}  # 预训练的属性嵌入
    def _build_graph_from_file(self, remap_ids=True):
        """
        从文件构建图
        
        Args:
            file_path: 边文件路径
            remap_ids: 是否重新映射节点ID（解决非连续ID问题）
            
        Returns:
            graph: NetworkX图对象
            nodes_adj: 邻接表字典
            n_nodes: 节点数量
            id_mapping: 原始ID到新ID的映射（如果remap_ids=True）
        """
        edges = []
        unique_nodes = set()
        
        # 第一遍：收集所有节点
        file_path = os.path.join(self.data_dir, self.dataset, f"{self.dataset}.txt")
        for line in open(file_path, encoding='utf-8'):
            node1, node2 = line.strip().split()
            node1_ = int(node1)
            node2_ = int(node2)
            
            # 跳过自环
            if node1_ == node2_:
                continue
                
            edges.append([node1_, node2_])
            unique_nodes.add(node1_)
            unique_nodes.add(node2_)
        
        n_nodes = len(unique_nodes)
        
        if remap_ids:
            # 创建ID映射：原始ID -> 连续ID (0, 1, 2, ...)
            original_nodes = sorted(list(unique_nodes))
            id_mapping = {original_id: new_id for new_id, original_id in enumerate(original_nodes)}
            reverse_mapping = {new_id: original_id for original_id, new_id in id_mapping.items()}
            
            # 映射边
            mapped_edges = []
            for node1, node2 in edges:
                mapped_edges.append([id_mapping[node1], id_mapping[node2]])
            
            edges = mapped_edges
        else:
            id_mapping = None
            reverse_mapping = None
        
        # 构建图
        nodes_lists = list(range(n_nodes))
        graph = nx.Graph()
        graph.add_nodes_from(nodes_lists)
        graph.add_edges_from(edges)
        self.original_graph = graph
        self.node_num = n_nodes
        self.id_mapping = id_mapping
        self.reverse_mapping = reverse_mapping    
    def _load_features(self):
        """
        加载稠密格式的特征文件
        
        文件格式：节点ID + 480个0/1值
        例如：1 0 0 0 1 0 1 ... (480个值)
        
        需要：
        1. nodes_feats: 直接存储0/1向量矩阵
        2. node_feats_nid: 从0/1向量提取特征索引
        3. nodes_adj_bi: 从特征索引构建
        
        Args:
            file_path: 特征文件路径
            n_nodes: 节点数量
        """
        file_path = os.path.join(self.data_dir, self.dataset, f"{self.dataset}_feat.txt")
        if not os.path.isfile(file_path):
            raise Exception(f"No such file: {file_path}")
        
        feats_node = {}          # 临时存储向量，用于构建nodes_feats
        
        with open(file_path, encoding='utf-8') as f:
            # 第一行：节点数 特征维度
            # first_line = f.readline().strip()
            # if not first_line:
            #     raise ValueError("Feature file is empty")
            
            # node_n_, node_in_dim = map(int, first_line.split())
            
            # # 更新节点数量
            # if n_nodes < node_n_:
            #     n_nodes = node_n_
            
            # 读取每个节点的特征
            for line in f:
                if not line.strip():
                    continue
                
                values = list(map(int, line.strip().split()))
                if len(values) < 2:
                    continue
                
                if values[0] in self.id_mapping:
                    node_id = self.id_mapping[values[0]] 
                else:
                    continue  # 跳过不在映射中的节点
                feature_vector = values[1:]  # 480个0/1值
                self.attributes[node_id] = feature_vector 
                self.feat_num = len(feature_vector)
    def construct_attribute_augmented_graph(self) -> nx.Graph:
        """
        构建属性增强图 GA = (V∪VA, E∪EA)
        
        返回:
            attribute_augmented_graph: 属性增强图
        """
        print("开始构建属性增强图...")
        
        # 创建原始图的深拷贝
        self.attribute_augmented_graph = self.original_graph.copy()
        
        # 创建属性节点

        
        # 为每个属性创建节点，使用负ID表示属性节点
        attribute_node_id = self.node_num 
        
        for attr in range(self.feat_num):
            # self.attribute_nodes[attr] = attribute_node_id
            # self.reverse_attribute_nodes[attribute_node_id] = attr
            self.attribute_augmented_graph.add_node(attribute_node_id, 
                                                   type='attribute', 
                                                   attribute_id=attr)
            attribute_node_id += 1
        
        print(f"创建了 {self.feat_num} 个属性节点")
        
        # 添加连接节点和属性节点的边
        edges_added = 0
        for node_id, attrs in self.attributes.items():
            if node_id in self.attribute_augmented_graph:
                count = 0
                for attr in attrs:
                    if attr == 1:
                        attr_node_id = self.node_num + count
                        count += 1
                        self.attribute_augmented_graph.add_edge(node_id, attr_node_id)
                        edges_added += 1
        
        print(f"添加了 {edges_added} 条节点-属性边")
        print(f"增强图总节点数: {self.attribute_augmented_graph.number_of_nodes()}")
        print(f"增强图总边数: {self.attribute_augmented_graph.number_of_edges()}")
        
        return self.attribute_augmented_graph
    
    def preprocess_for_embedding(self) -> Tuple[sp.csr_matrix, List[int]]:
        """
        为ProNE算法准备数据
        
        返回:
            adjacency_matrix: 邻接矩阵
            node_list: 节点ID列表
        """
        if self.attribute_augmented_graph is None:
            self.construct_attribute_augmented_graph()
        
        # 获取所有节点（包括属性节点）
        nodes = list(self.attribute_augmented_graph.nodes())
        node_index = {node: idx for idx, node in enumerate(nodes)}
        
        # 构建邻接矩阵
        n_nodes = len(nodes)
        adjacency_matrix = sp.lil_matrix((n_nodes, n_nodes))
        
        for u, v in self.attribute_augmented_graph.edges():
            i, j = node_index[u], node_index[v]
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
        
        return adjacency_matrix.tocsr(), nodes
    
    def prone_embedding(self, dim: int = 128, step: int = 10, 
                       mu: float = 0.2, theta: float = 0.5) -> np.ndarray:
        """
        实现ProNE算法进行图嵌入
        
        参数:
            dim: 嵌入维度
            step: 迭代步数
            mu: 正则化参数
            theta: 谱传播参数
            
        返回:
            embeddings: 所有节点的嵌入
        """
        print("开始ProNE嵌入预训练...")
        
        # 获取邻接矩阵和节点列表
        adjacency_matrix, node_list = self.preprocess_for_embedding()
        n_nodes = adjacency_matrix.shape[0]
        
        # 归一化邻接矩阵
        print("步骤1: 归一化邻接矩阵...")
        degree = np.array(adjacency_matrix.sum(axis=1)).flatten()
        degree_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1))
        degree_inv_sqrt_mat = sp.diags(degree_inv_sqrt)
        normalized_adj = degree_inv_sqrt_mat @ adjacency_matrix @ degree_inv_sqrt_mat
        
        # 谱传播
        print("步骤2: 谱传播...")
        I = sp.eye(n_nodes)
        if theta > 0:
            normalized_adj = theta * normalized_adj
            for _ in range(step):
                normalized_adj = normalized_adj @ normalized_adj
        
        # 稀疏矩阵分解
        print("步骤3: 稀疏矩阵分解...")
        svd = TruncatedSVD(n_components=dim, random_state=42)
        embeddings = svd.fit_transform(normalized_adj)
        
        # 增强嵌入
        print("步骤4: 嵌入增强...")
        embeddings = normalize(embeddings, axis=1, norm='l2')
        
        # 存储所有节点的嵌入
        self.node_embeddings = {}
        for idx, node_id in enumerate(node_list):
            self.node_embeddings[node_id] = embeddings[idx]
        embed_feats_path = os.path.join(self.data_dir, self.dataset, f"{self.dataset}_emb.npy")
        # 提取属性节点的嵌入
        # count = 0
        
        attr_embeddings = []
        
        for attr_node_id in range(self.node_num, self.node_num + self.feat_num):
            if attr_node_id in self.node_embeddings:
                #把self.node_embeddings[attr_node_id]输出到文件，并且可以通过embed_feats_path=data_dir+"/emb/egotwitter_enhanced{}.emb.npy".format(ego_node_id);embed_feats = torch.from_numpy(np.load(embed_feats_path,allow_pickle=True))读取
                attr_embedding = self.node_embeddings[attr_node_id]
                attr_embeddings.append(attr_embedding)
                # np.save(embed_feats_path, attr_embedding)
                # count += 1

        # print(f"预训练完成，获得 {count} 个属性嵌入")
        if attr_embeddings:
            attr_embeddings_array = np.array(attr_embeddings)  # shape: (feat_num, embedding_dim)
            np.save(embed_feats_path, attr_embeddings_array)
            print(f"保存了 {len(attr_embeddings)} 个属性嵌入到 {embed_feats_path}")

        embed_node_path = os.path.join(self.data_dir, self.dataset, f"{self.dataset}_node_emb.txt")
        org_node_embeddings = {self.reverse_mapping[node_id]: self.node_embeddings[node_id] for node_id in range(self.node_num)}
        if org_node_embeddings:
            with open(embed_node_path, "w") as f:
                for node_id, embedding in org_node_embeddings.items():
                    f.write(f"{node_id} {embedding}\n")
            print(f"保存了 {len(org_node_embeddings)} 个节点嵌入到 {embed_node_path}")
        return embeddings
    
if __name__ == "__main__":
    data_dir = "./data"
    dataset = "cora"
    
    graph_builder = AttributeAugmentedGraph(data_dir, dataset)
    graph_builder._build_graph_from_file()
    graph_builder._load_features()
    graph_builder.construct_attribute_augmented_graph()
    graph_builder.prone_embedding(dim=128, step=10, mu=0.2, theta=0.5)