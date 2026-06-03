import os
import random
import networkx as nx
import numpy as np

class CommunityQueryLoader:
    def __init__(self, root, dataset):
        """
        初始化数据加载器
        
        Args:
            root: 数据根目录
            dataset: 数据集名称
            feats_path: 特征文件路径
        """
        self.root = root
        self.dataset = dataset
        
        # 存储数据的变量
        self.id_mapping = None
        self.reverse_mapping = None
        self.graph = None
        self.nodes_adj = None
        self.nodes_adj_bi = None
        self.nodes_feats = None
        self.node_feats_nid = None
        self.node_in_dim = None
        self.n_nodes = None
        
        # 查询数据直接存储在内存中
        self.train_queries = None
        self.val_queries = None
        self.test_queries = None
        
    # def _build_graph_from_file(self, file_path):
    #     """
    #     从文件构建图
        
    #     Args:
    #         file_path: 边文件路径
            
    #     Returns:
    #         graph: NetworkX图对象
    #         nodes_adj: 邻接表字典
    #         n_nodes: 节点数量
    #     """
    #     max_node = 0
    #     edges = []
        
    #     for line in open(file_path, encoding='utf-8'):
    #         node1, node2 = line.strip().split()
    #         node1_ = int(node1)
    #         node2_ = int(node2)
            
    #         # 跳过自环
    #         if node1_ == node2_:
    #             continue
                
    #         # 更新最大节点ID
    #         max_node = max(max_node, node1_, node2_)
    #         edges.append([node1_, node2_])
        
    #     n_nodes = max_node + 1
    #     nodes_lists = list(range(n_nodes))
        
    #     # 构建图
    #     graph = nx.Graph()
    #     graph.add_nodes_from(nodes_lists)
    #     graph.add_edges_from(edges)
        
    #     # 构建邻接表
    #     nodes_adj = {}
    #     for id1, id2 in graph.edges:
    #         nodes_adj.setdefault(id1, []).append(id2)
    #         nodes_adj.setdefault(id2, []).append(id1)
            
    #     return graph, nodes_adj, n_nodes
    
    # def _load_features(self, file_path, n_nodes):
    #     """
    #     加载节点特征
        
    #     Args:
    #         file_path: 特征文件路径
    #         n_nodes: 节点数量
            
    #     Returns:
    #         nodes_feats: 节点特征矩阵
    #         node_feats_nid: 节点特征字典
    #         nodes_adj_bi: 双向邻接表
    #         node_in_dim: 特征维度
    #     """
    #     if not os.path.isfile(file_path):
    #         raise Exception(f"No such file: {file_path}")
        
    #     feats_node = {}
    #     node_feats_nid = {}
    #     nodes_adj_bi = {}
        
    #     with open(file_path, encoding='utf-8') as f:
    #         # 第一行包含节点数量和特征维度
    #         first_line = f.readline().strip()
    #         if not first_line:
    #             raise ValueError("Feature file is empty")
                
    #         node_n_, node_in_dim = first_line.split()
    #         node_n_ = int(node_n_)
    #         node_in_dim = int(node_in_dim)
            
    #         # 更新节点数量
    #         if n_nodes < node_n_:
    #             n_nodes = node_n_
            
    #         # 读取特征
    #         for line in f:
    #             if not line.strip():
    #                 continue
                    
    #             emb = [int(x) for x in line.strip().split()]
    #             if len(emb) < 2:
    #                 continue
                    
    #             node_id = emb[0]
    #             features = emb[1:]
                
    #             # 存储原始特征
    #             node_feats_nid[node_id] = features
                
    #             # 创建one-hot特征向量
    #             feat_vector = np.zeros(node_in_dim)
    #             for f in features:
    #                 feat_vector[f] = 1.0
    #                 # 构建双向邻接表
    #                 nodes_adj_bi.setdefault(f, []).append(node_id)
                
    #             feats_node[node_id] = feat_vector
        
    #     # 创建完整的特征矩阵
    #     features_matrix = []
    #     for i in range(n_nodes):
    #         if i in feats_node:
    #             features_matrix.append(feats_node[i])
    #         else:
    #             features_matrix.append([0.0] * node_in_dim)
        
    #     nodes_feats = np.array(features_matrix)
        
    #     return nodes_feats, node_feats_nid, nodes_adj_bi, node_in_dim, n_nodes
    def _build_graph_from_file(self, file_path, remap_ids=True):
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
        
        # 构建邻接表
        nodes_adj = {}
        for id1, id2 in graph.edges:
            nodes_adj.setdefault(id1, []).append(id2)
            nodes_adj.setdefault(id2, []).append(id1)
        
        return graph, nodes_adj, n_nodes, id_mapping, reverse_mapping    
    def _load_features(self, file_path, n_nodes, id_mapping=None):
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
        if not os.path.isfile(file_path):
            raise Exception(f"No such file: {file_path}")
        
        feats_node = {}          # 临时存储向量，用于构建nodes_feats
        node_feats_nid = {}      # 存储特征索引列表
        nodes_adj_bi = {}        # {特征索引: [节点列表]}
        
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
                
                if values[0] in id_mapping:
                    node_id = id_mapping[values[0]] 
                else:
                    continue  # 跳过不在映射中的节点
                feature_vector = values[1:]  # 480个0/1值
                node_in_dim = len(feature_vector)
                # 检查维度
                if len(feature_vector) != node_in_dim:
                    print(f"Warning: Node {node_id} feature dimension mismatch. "
                        f"Expected {node_in_dim}, got {len(feature_vector)}")
                    # 调整维度
                    if len(feature_vector) < node_in_dim:
                        feature_vector.extend([0] * (node_in_dim - len(feature_vector)))
                    else:
                        feature_vector = feature_vector[:node_in_dim]
                
                # 1. 直接存储向量到feats_node（用于构建nodes_feats）
                feat_array = np.array(feature_vector, dtype=np.float32)
                feats_node[node_id] = feat_array
                
                # 2. 提取特征索引存储到node_feats_nid
                # 找出值为1的特征位置
                feature_indices = [idx for idx, val in enumerate(feature_vector) if val == 1]
                node_feats_nid[node_id] = feature_indices
                
                # 3. 构建nodes_adj_bi
                for feat_idx in feature_indices:
                    nodes_adj_bi.setdefault(feat_idx, []).append(node_id)
        
        # 构建完整的特征矩阵
        features_matrix = []
        zero_vector = np.zeros(node_in_dim, dtype=np.float32)
        
        for i in range(n_nodes):
            if i in feats_node:
                features_matrix.append(feats_node[i])
            else:
                features_matrix.append(zero_vector)
        
        nodes_feats = np.array(features_matrix)
        
        print(f"Loaded dense features: {len(node_feats_nid)} nodes, "
            f"feature dim: {node_in_dim}, matrix shape: {nodes_feats.shape}")
        
        return nodes_feats, node_feats_nid, nodes_adj_bi, node_in_dim, n_nodes    
    # def _parse_query_line(self, line):
    #     """
    #     解析查询数据行
        
    #     Args:
    #         line: 查询数据行
            
    #     Returns:
    #         tuple: (查询节点列表, 属性列表, 社区节点列表)
    #     """
    #     if not line.strip():
    #         return None
            
    #     try:
    #         qlists_str, attrlists_str, comm_str = line.strip().split(",")
            
    #         # 解析查询节点
    #         qlists = [int(x) for x in qlists_str.split()]
    #         qlists = qlists[:1]  # 只取第一个查询节点
            
    #         # 解析属性列表
    #         attrlists = [int(x) for x in attrlists_str.split()]
            
    #         # 解析社区节点
    #         comm = [int(x) for x in comm_str.split()]
            
    #         return (qlists, attrlists, comm)
    #     except ValueError:
    #         print(f"Warning: Could not parse line: {line.strip()}")
    #         return None
    
    # def load_data_with_queries(self, train_n=None, val_n=None, test_n=None, 
    #                            train_path=None, test_path=None, val_path=None):
    #     """
    #     加载所有数据（包括查询数据）
        
    #     Args:
    #         train_n: 训练样本数量
    #         val_n: 验证样本数量
    #         test_n: 测试样本数量
    #         train_path: 训练查询文件路径
    #         test_path: 测试查询文件路径
    #         val_path: 验证查询文件路径
            
    #     Returns:
    #         所有加载的数据
    #     """
    #     # 构建文件路径
    #     edge_file = os.path.join(self.root, self.dataset, f"{self.dataset}.txt")
        
    #     # 构建图
    #     self.graph, self.nodes_adj, self.n_nodes = self._build_graph_from_file(edge_file)
        
    #     # 加载特征
    #     feat_file = os.path.join(self.root, self.dataset, self.feats_path)
    #     self.nodes_feats, self.node_feats_nid, self.nodes_adj_bi, self.node_in_dim, self.n_nodes = \
    #         self._load_features(feat_file, self.n_nodes)
        
    #     # 加载查询数据（如果有文件路径提供）
    #     if train_path:
    #         train_file = os.path.join(self.root, self.dataset, f"{self.dataset}{train_path}")
    #         self.train_queries = self._load_queries_from_file(train_file, train_n)
        
    #     if test_path:
    #         test_file = os.path.join(self.root, self.dataset, f"{self.dataset}{test_path}")
    #         self.test_queries = self._load_queries_from_file(test_file, test_n)
        
    #     if val_path:
    #         val_file = os.path.join(self.root, self.dataset, f"{self.dataset}{val_path}")
    #         self.val_queries = self._load_queries_from_file(val_file, val_n)
        
    #     return self._get_all_data()
    
    # def _load_queries_from_file(self, file_path, max_samples=None):
    #     """
    #     从文件加载查询
        
    #     Args:
    #         file_path: 查询文件路径
    #         max_samples: 最大样本数
            
    #     Returns:
    #         query_lists: 查询列表
    #     """
    #     if not os.path.isfile(file_path):
    #         raise Exception(f"No such file: {file_path}")
        
    #     query_lists = []
        
    #     with open(file_path, encoding='utf-8') as f:
    #         for line in f:
    #             if max_samples and len(query_lists) >= max_samples:
    #                 break
                    
    #             query_data = self._parse_query_line(line)
    #             if query_data:
    #                 query_lists.append(query_data)
        
    #     return query_lists
    
    def _generate_queries(self, feats_node, map_comm, map_comm_attrs, 
                         map_comm_all, query_size):
        """
        查询生成规则：
        1.一个查询包括一组查询点、一组查询属性以及所在的社区
        2.保证查询覆盖极可能多的社区，要求总查询数满足query_size，分为训练集、验证集、测试集
        3.先从社区中随机选择1-3个节点作为查询节点
        4.选择社区内频率最高的3个属性作为查询属性
        5.如果社区没有属性，则使用查询节点的特征作为属性
        生成查询数据（不保存到文件）
        
        Args:
            feats_node: 节点特征字典
            map_comm: 社区映射字典
            map_comm_attrs: 社区属性映射字典
            map_comm_all: 全局属性映射字典
            start_idx: 起始社区索引
            end_idx: 结束社区索引
            query_size: 查询数量
            
        Returns:
            生成的查询列表
        """
        query_lists = []
        community_range = list(range(0, len(map_comm)))
        
        # 生成足够的社区ID
        truth_comm_ids = []
        while len(truth_comm_ids) < query_size:
            random.shuffle(community_range)
            truth_comm_ids.extend(community_range)
        truth_comm_ids = truth_comm_ids[:query_size]
        random.shuffle(truth_comm_ids)
        
        count = 0
        generated = 0
        
        while generated < query_size and count < len(truth_comm_ids):
            comm_id = truth_comm_ids[count]
            community = map_comm.get(comm_id)
            
            # 检查社区是否存在且足够大
            if not community or len(community) <= 2:
                count += 1
                continue
            
            # 随机选择查询节点
            query_size_nodes = random.randint(1, 3)
            random.shuffle(community)
            query_nodes = community[:query_size_nodes]
            
            # 选择属性
            comm_attrs = map_comm_attrs.get(comm_id, {})
            attr_scores = {}
            
            # 计算属性分数
            # for attr, freq in comm_attrs.items():
            #     global_freq = map_comm_all.get(attr, 0)
            #     score = freq - (global_freq - freq)
            #     attr_scores[attr] = score
            
            # 选择分数最高的3个属性
            # if attr_scores:
            #     sorted_attrs = sorted(attr_scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
            #     selected_attrs = [attr for attr, _ in sorted_attrs[:3]]
            if comm_attrs:
                sorted_attrs = sorted(comm_attrs.items(), key=lambda x: (x[1], x[0]), reverse=True)
                selected_attrs = [attr for attr, _ in sorted_attrs[:3]]
            else:
                # 如果没有属性，使用查询节点的第一个特征
                selected_attrs = []
                for node in query_nodes:
                    if node in feats_node and feats_node[node]:
                        selected_attrs = feats_node[node][:3]
                        break
            
            # 添加到查询列表
            query_lists.append((query_nodes, selected_attrs, community))
            generated += 1
            count += 1
        
        print(f"Generated {generated} queries")
        return query_lists
    
    def generate_and_load_queries(self, train_size, val_size, test_size):
        """
        生成查询数据并直接加载到内存
        
        Args:
            train_size: 训练查询数量
            val_size: 验证查询数量
            test_size: 测试查询数量
            
        Returns:
            所有加载的数据
        """
        # 首先加载基础数据（图结构和特征）
        edge_file = os.path.join(self.root, self.dataset, f"{self.dataset}.txt")
        self.graph, self.nodes_adj, self.n_nodes, self.id_mapping, self.reverse_mapping = self._build_graph_from_file(edge_file)
        
        feat_file = os.path.join(self.root, self.dataset, f"{self.dataset}_feat.txt")
        self.nodes_feats, self.node_feats_nid, self.nodes_adj_bi, self.node_in_dim, self.n_nodes = \
            self._load_features(feat_file, self.n_nodes, self.id_mapping)
        
        # 读取真实社区
        cmty_file = os.path.join(self.root, self.dataset, f"{self.dataset}_cmty.txt")
        
        if not os.path.isfile(cmty_file):
            raise Exception(f"No such file: {cmty_file}")
        
        map_comm = {}
        seen_comms = set()
        
        with open(cmty_file, encoding='utf-8') as f:
            comm_id = 0
            for line in f:
                comm = line.strip().split()
                comm = comm[1:] if len(comm) > 1 else []
                
                if not comm:
                    continue
                
                comm_str = " ".join(sorted(comm))
                
                # 去重
                if comm_str in seen_comms:
                    continue
                seen_comms.add(comm_str)
                
                comm_nodes = list(map(int, comm))
                
                # 跳过太小的社区
                if len(comm_nodes) <= 2:
                    continue
                comm_nodes = [self.id_mapping[node] for node in comm_nodes if node in self.id_mapping]
                map_comm[comm_id] = comm_nodes
                comm_id += 1
        
        print(f"Found {len(map_comm)} communities")
        # random.shuffle(truth_comms)
        
        # # 构建节点特征字典（只包含原始特征，不包含one-hot编码）
        # feats_node_raw = {}
        # map_cout_all = {}
        
        # # 从特征矩阵重建原始特征
        # for node_id in range(len(self.node_feats_nid)):
        #     if node_id in self.node_feats_nid:
        #         features = self.node_feats_nid[node_id]
        #         feats_node_raw[node_id] = features
                
        #         # 统计全局属性频率
        #         for attr in features:
        #             map_cout_all[attr] = map_cout_all.get(attr, 0) + 1
        
        # 构建社区映射
        
        map_comm_attrs = {}
        map_comm_all = {}
        
        for idx, community in map_comm.items():
            # if idx >= 100:  # 限制社区数量
            #     break
                
            # map_comm[idx] = community
            
            # 统计社区内属性频率
            attr_counts = {}
            for node in community:
                if node not in self.node_feats_nid:
                    continue
                    
                for attr in self.node_feats_nid[node]:
                    attr_counts[attr] = attr_counts.get(attr, 0) + 1
            
            map_comm_attrs[idx] = attr_counts
            
            # 更新全局属性最大频率
            # for attr, count in attr_counts.items():
            #     if attr not in map_comm_all or count > map_comm_all[attr]:
            #         map_comm_all[attr] = count
        
        # 生成查询数据（直接存储在内存中）
        print("Generating training queries...")
        self.train_queries = self._generate_queries(
            self.node_feats_nid, map_comm, map_comm_attrs, map_comm_all, train_size
        )
        
        print("Generating validation queries...")
        self.val_queries = self._generate_queries(
            self.node_feats_nid, map_comm, map_comm_attrs, map_comm_all, val_size
        )
        
        print("Generating test queries...")
        self.test_queries = self._generate_queries(
            self.node_feats_nid, map_comm, map_comm_attrs, map_comm_all, test_size
        )
        
        return self._get_all_data()
    
    def _get_all_data(self):
        """
        获取所有加载的数据
        
        Returns:
            所有数据组成的元组
        """
        return (self.nodes_feats, self.train_queries, self.val_queries, self.test_queries,
                self.node_in_dim, self.n_nodes, self.graph, self.nodes_adj, 
                self.nodes_adj_bi, self.node_feats_nid)
    
    # def save_queries_to_files(self, train_path="_train.txt", test_path="_test.txt", 
    #                          val_path="_val.txt"):
    #     """
    #     将内存中的查询数据保存到文件（可选）
        
    #     Args:
    #         train_path: 训练查询文件路径
    #         test_path: 测试查询文件路径
    #         val_path: 验证查询文件路径
    #     """
    #     if self.train_queries:
    #         self._save_queries_to_file(self.train_queries, train_path)
        
    #     if self.test_queries:
    #         self._save_queries_to_file(self.test_queries, test_path)
        
    #     if self.val_queries:
    #         self._save_queries_to_file(self.val_queries, val_path)
    
    # def _save_queries_to_file(self, queries, file_path):
    #     """
    #     将查询数据保存到文件
        
    #     Args:
    #         queries: 查询数据列表
    #         file_path: 文件路径
    #     """
    #     output_file = os.path.join(self.root, self.dataset, f"{self.dataset}{file_path}")
        
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         for query_nodes, selected_attrs, community in queries:
    #             query_str = " ".join(str(x) for x in query_nodes)
    #             attr_str = " ".join(str(x) for x in selected_attrs)
    #             comm_str = " ".join(str(x) for x in community)
                
    #             f.write(f"{query_str},{attr_str},{comm_str}\n")
        
    #     print(f"Saved {len(queries)} queries to {output_file}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    root = "./data/"
    dataset = "example_dataset"
    feats_path = "_feats.txt"
    
    # 创建数据加载器
    # loader = CommunityQueryLoader(root, dataset, feats_path)
    
    # # 方法1：直接从文件加载查询数据
    # print("=== 从文件加载数据 ===")
    # data1 = loader.load_data_with_queries(
    #     train_n=1000, val_n=200, test_n=200,
    #     train_path="_train.txt", test_path="_test.txt", val_path="_val.txt"
    # )
    
    # 方法2：生成查询数据并直接加载到内存（省去文件IO）
    print("\n=== 生成查询数据并加载到内存 ===")
    loader2 = CommunityQueryLoader(root, dataset)
    data2 = loader2.generate_and_load_queries(
        train_size=150, val_size=100, test_size=100
    )#4:3:3比例划分
    
    # 如果需要，可以将数据保存到文件
    # loader2.save_queries_to_files()
    
    # 访问数据
    nodes_feats, train_queries, val_queries, test_queries, node_in_dim, n_nodes, graph, nodes_adj, nodes_adj_bi, node_feats_nid = data2
    
    print(f"\n数据统计:")
    print(f"- 节点数量: {n_nodes}")
    print(f"- 特征维度: {node_in_dim}")
    print(f"- 训练查询数量: {len(train_queries)}")
    print(f"- 验证查询数量: {len(val_queries)}")
    print(f"- 测试查询数量: {len(test_queries)}")
    print(f"- 图边数量: {graph.number_of_edges()}")
    
    # 示例：访问第一个训练查询
    if train_queries:
        q_nodes, q_attrs, q_comm = train_queries[0]
        print(f"\n第一个训练查询:")
        print(f"  查询节点: {q_nodes}")
        print(f"  查询属性: {q_attrs}")
        print(f"  社区大小: {len(q_comm)}")