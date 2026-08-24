import os
import random
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp

class CommunityQueryLoader:
    def __init__(self, root, dataset, min_community_size, min_community_num):
        """
        初始化数据加载器
        
        Args:
            root: 数据根目录
            dataset: 数据集名称
            feats_path: 特征文件路径
        """
        # self.config = config
        self.root = root
        self.dataset = dataset
        self.min_community_size=min_community_size
        self.min_community_num = min_community_num
    def _build_graph_from_file(self, file_path, n_nodes, id_mapping, remap_ids=True):
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
        edge_str = set()
        
        # 第一遍：收集所有节点
        for line in open(file_path, encoding='utf-8'):
            node1, node2 = line.strip().split()
            node1_ = int(node1)
            node2_ = int(node2)
            
            # 跳过自环
            edge_s = str(min(node1_, node2_)) + "_" + str(max(node1_, node2_))
            if node1_ == node2_ or edge_s in edge_str or node1_ not in id_mapping or node2_ not in id_mapping:
                continue
            edge_str.add(edge_s)
            edges.append([id_mapping[node1_], id_mapping[node2_]])
            edges.append([id_mapping[node2_], id_mapping[node1_]])
        if not edges:
            return None, None
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        row, col = edge_index.cpu().numpy()
        data = np.ones(len(row))
        adj_matrix = sp.csr_matrix((data, (row, col)), 
                                  shape=(n_nodes, n_nodes))
        adj = torch.from_numpy(adj_matrix.toarray()).float()
        # print(features.shape)
        return edge_index, adj   
    def _load_features(self, file_path):
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
        # id_mapping = self.id_mapping
        # n_nodes = self.n_nodes
        feats_node = {}          # 临时存储向量，用于构建nodes_feats
        node_feats_nid = {}      # 存储特征索引列表
        unique_nodes = set()
        with open(file_path, encoding='utf-8') as f:
            # 读取每个节点的特征
            for line in f:
                if not line.strip():
                    continue
                
                values = list(map(int, line.strip().split()))
                if len(values) < 2:
                    continue
                
                
                feature_vector = values[1:]  # 480个0/1值
                node_in_dim = len(feature_vector)
                # 2. 提取特征索引存储到node_feats_nid
                # 找出值为1的特征位置
                feature_indices = [idx for idx, val in enumerate(feature_vector) if val == 1]
                if len(feature_indices) == 0:
                    # print(f"Warning: Node {values[0]} has no features")   
                    continue 
                node_id = values[0]
                if node_id not in unique_nodes:
                    unique_nodes.add(node_id)
                else:
                    continue  # 跳过不在映射中的节点            
                node_feats_nid[node_id] = feature_indices
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
                
                
        original_nodes = sorted(list(unique_nodes))
        id_mapping = {original_id: new_id for new_id, original_id in enumerate(original_nodes)}
        reverse_mapping = {new_id: original_id for original_id, new_id in id_mapping.items()}
        n_nodes = len(original_nodes)
        # 构建完整的特征矩阵
        features_matrix = []
        
        for i in range(n_nodes):
            features_matrix.append(feats_node[reverse_mapping[i]])
        
        nodes_feats = np.array(features_matrix)
        node_feature_indices = {i : node_feats_nid[reverse_mapping[i]] for i in range(n_nodes)}
        print(
            f"Loaded dense features: {len(node_feats_nid)} nodes, "
            f"feature dim: {node_in_dim}, matrix shape: {nodes_feats.shape}")
        return nodes_feats, node_feature_indices, node_in_dim, n_nodes, id_mapping   
    def _load_embeddings(self, file_path1,file_path2, n_nodes, id_mapping):
        """
        加载稠密格式的特征文件
        
        文件格式：节点ID + 嵌入
        
        需要：
        1. nodes_feats: 直接存储0/1向量矩阵
        2. node_feats_nid: 从0/1向量提取特征索引
        3. nodes_adj_bi: 从特征索引构建
        
        Args:
            file_path: 特征文件路径
            n_nodes: 节点数量
        """
        
        if not os.path.isfile(file_path1):
            raise Exception(f"No such file: {file_path1}")
        # id_mapping = self.id_mapping
        # n_nodes = self.n_nodes
        feats_node = {}          # 临时存储向量，用于构建nodes_feats
        
        with open(file_path1, encoding='utf-8') as f:
            # 读取每个节点的特征
            for line in f:
                if not line.strip():
                    continue
                
                values = line.strip().split()
                node_id = int(values[0])
                if node_id not in id_mapping:
                    continue
                node_id = id_mapping[node_id] 
                
                feature_vector = [float(x) for x in values[1:]]
                # print(feature_vector[0])
                node_in_dim = len(feature_vector)
                if node_in_dim < 128:
                    feature_vector.extend([0.0] * (128 - node_in_dim))
                
                # 1. 直接存储向量到feats_node（用于构建nodes_feats）
                feat_array = np.array(feature_vector, dtype=np.float32)
                feats_node[node_id] = feat_array
        
        # 构建完整的特征矩阵
        features_matrix = []
        zero_vector = np.zeros(128, dtype=np.float32)
        
        for i in range(n_nodes):
            if i in feats_node:
                features_matrix.append(feats_node[i])
            else:
                features_matrix.append(zero_vector)
        
        nodes_feats = torch.from_numpy(np.array(features_matrix))
        
        print(
            f"feature dim: {node_in_dim}, matrix shape: {nodes_feats.shape}")
        
        embed_feats_np = np.load(file_path2,allow_pickle=True)
        # print(embed_feats_np.dtype)
        # print(embed_nodes)
        embed_feats = torch.from_numpy(embed_feats_np.astype(np.float32))
        # 获取当前维度
        current_dim = embed_feats.size(1)  # 假设形状为 [num_nodes, dim]
        print(f"当前嵌入维度: {current_dim}")

        # 如果小于128，进行填充
        if current_dim < 128:
            # 计算需要填充的维度
            padding_size = 128 - current_dim
            
            # 用零填充（也可以选择其他初始化方式）
            padding = torch.zeros(embed_feats.size(0), padding_size, dtype=embed_feats.dtype)
            
            # 拼接原始嵌入和填充部分
            embed_feats_padded = torch.cat([embed_feats, padding], dim=1)
            
            print(f"从 {current_dim} 维填充到 {embed_feats_padded.size(1)} 维")
        else:
            embed_feats_padded = embed_feats
            print(f"嵌入维度已经是 {current_dim}，无需填充")
        return nodes_feats, embed_feats_padded
    def _load_community(self, file_path, id_mapping):
        """
        加载社区文件
        
        文件格式：每行一个社区，节点ID以空格分隔
        例如：1 2 3 4 5
        
        Args:
            file_path: 社区文件路径
            id_mapping: 原始ID到新ID的映射
            
        Returns:
            map_comm: 社区映射字典 {社区ID: [节点列表]}
        """
        if not os.path.isfile(file_path):
            raise Exception(f"No such file: {file_path}")
        
        map_comm = {}
        seen_comms = set()
        
        with open(file_path, encoding='utf-8') as f:
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
                
                comm_nodes = set(map(int, comm))
                comm_nodes = list(comm_nodes)
                comm_nodes = [id_mapping[node] for node in comm_nodes if node in id_mapping]
                
                # 跳过太小的社区
                if len(comm_nodes) < self.min_community_size:
                    continue
                
                map_comm[comm_id] = comm_nodes
                comm_id += 1
        
        print(f"Loaded {len(map_comm)} communities from {file_path}")
        return map_comm
    
    def _generate_queries(self, graph, query_size):
        feats_node = graph['attribute_index']
        map_comm = graph['communities']
        map_comm_attrs = graph['community_attributes']
        query_lists = []
        community_range = list(range(0, len(map_comm)))
        
        # 生成足够的社区ID，如果社区数量不足，则重复使用；如果社区数量大于等于query_size，则随机选择query_size个
        truth_comm_ids = []
        if len(community_range) < query_size:
            while len(truth_comm_ids) < query_size:
                truth_comm_ids.extend(community_range)  
        else:
            random.shuffle(community_range)
            truth_comm_ids.extend(community_range)
        # truth_comm_ids = []
        # while len(truth_comm_ids) < query_size:
        #     random.shuffle(community_range)
        #     truth_comm_ids.extend(community_range)
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
            # query_lists.append((query_nodes, selected_attrs, community))
            community_mask = torch.zeros(len(feats_node))
            community_mask[community] = 1
            
            # 查询节点索引
            query_nodes = torch.tensor(query_nodes, dtype=torch.long)
            # query_nodes_mask = torch.zeros(len(feats_node))
            # query_nodes_mask[query_nodes] = 1
            
            
            # 查询属性
            if selected_attrs is not None:
                query_attrs = torch.tensor(selected_attrs, dtype=torch.long)
                # query_attrs_mask = torch.zeros(num_attributes)
                # query_attrs_mask[selected_attrs] = 1
            else:
                query_attrs = torch.tensor([], dtype=torch.long)
                # query_attrs_mask = torch.tensor([], dtype=torch.long)
            # print(query_nodes)
            # print(query_attrs)
            # print(community)
            # print(len(feats_node))
            query_lists.append({
                # 'community': community,
                # 'query_nodes': query_nodes,
                # 'query_attrs': selected_attrs
                'graph': {
                'adj' : graph['adj'],
                'num_nodes': graph['num_nodes'],
                'node_embeddings': graph['node_embeddings'],
                'attr_embeddings': graph['attr_embeddings'],
                'edge_index': graph['edge_index'],
                'num_attributes': graph['num_attributes'],
                'attributes': graph['attributes'],
                'hyperedge_index': graph['hyperedge_index']
                },
                'query': {
                    'nodes': query_nodes,
                    'attrs': query_attrs
                    # 'nodes_mask': query_nodes_mask,
                    # 'attrs_mask': query_attrs_mask
                },
                'target': {
                    'community_indices': community,
                    'community_mask': community_mask
                }
            })
            generated += 1
            count += 1
        
        print(f"Generated {generated} queries")
        return query_lists

    def load_single_graph(self, dataname):
        """
        生成查询数据并直接加载到内存
        要求每个图的社区数量至少是9
        Args:
            train_size: 训练查询数量
            val_size: 验证查询数量
            test_size: 测试查询数量
            
        Returns:
            所有加载的数据
        """
        # 首先加载基础数据（图结构和特征）
        feat_file = os.path.join(self.root, self.dataset, f"{dataname}.feat")
        attributes, node_feats_nid, node_in_dim, n_nodes, id_mapping = \
            self._load_features(feat_file)
        if n_nodes < 2:
            return None
        edge_file = os.path.join(self.root, self.dataset, f"{dataname}.edges")
        edge_index, adj = \
            self._build_graph_from_file(edge_file, n_nodes, id_mapping)
        if edge_index is None:
            return None
        embed_nodes_path=os.path.join(self.root, self.dataset, f"{dataname}_node_emb.txt")
        embed_feats_path=os.path.join(self.root, self.dataset, f"{dataname}_emb.npy")
        embed_nodes, embed_feats =self._load_embeddings(embed_nodes_path,embed_feats_path, n_nodes, id_mapping)
        # print(embed_nodes)
        
        
        attr_index = []
        node_index = []
        for node_id, attrs in node_feats_nid.items():
            for attr in attrs:
                attr_index.append(attr)
                node_index.append(node_id)
        hyperedge_index = torch.tensor([attr_index, node_index], dtype=torch.long)
        # 读取真实社区
        cmty_file = os.path.join(self.root, self.dataset, f"{dataname}.circles")
        map_comm = self._load_community(cmty_file, id_mapping)
        
        
        # 构建社区映射
        
        map_comm_attrs = {}
        if len(map_comm) < self.min_community_num:
            return None
        for idx, community in map_comm.items():
            # if idx >= 100:  # 限制社区数量
            #     break
                
            # map_comm[idx] = community
            
            # 统计社区内属性频率
            attr_counts = {}
            for node in community:
                if node not in node_feats_nid:
                    continue
                    
                for attr in node_feats_nid[node]:
                    attr_counts[attr] = attr_counts.get(attr, 0) + 1
            
            map_comm_attrs[idx] = attr_counts
            
            # 更新全局属性最大频率
            # for attr, count in attr_counts.items():
            #     if attr not in map_comm_all or count > map_comm_all[attr]:
            #         map_comm_all[attr] = count
        graph = {
                # 'features': attributes,
                'adj' : adj,
                'node_embeddings' : embed_nodes,
                'attr_embeddings': embed_feats,
                'edge_index': edge_index,
                'attributes': attributes,
                'attribute_index': node_feats_nid,
                'hyperedge_index': hyperedge_index,
                'num_nodes': n_nodes,
                'num_attributes': node_in_dim,
                'communities': map_comm,
                'community_attributes': map_comm_attrs
            }
        return graph
    def load_multi_graph(self):
        graphs = list()
        data_dir = os.path.join(self.root, self.dataset)
        for file_name in os.listdir(data_dir):
            if os.path.splitext(file_name)[1] != '.edges':
                continue
            ego_node_id = os.path.splitext(file_name)[0]
            graph = self.load_single_graph(ego_node_id)
            if graph is not None:
                graphs.append(graph)
        return graphs
    def save_query_batches(self,query_batchs):
        """
        保存 support_batchs 和 query_batchs 到文件
        
        Args:
            support_batchs: support数据列表
            query_batchs: query数据列表
            save_dir: 保存目录
            dataset_name: 数据集名称
            num_shots: few-shot 数量
        """
        
        # 保存数据
        save_path = os.path.join(self.root, f"{self.dataset}_queries.pt")
        
        data_to_save = {
            'query_batchs': query_batchs,
            'metadata': {
                'dataset': self.dataset,
                'num_graphs': len(query_batchs),
                # 'support_shape': [len(batch) for batch in support_batchs[:5]],  # 前5个的形状
                # 'query_shape': [len(batch) for batch in query_batchs[:5]]
            }
        }
        
        torch.save(data_to_save, save_path)
        print(f"数据已保存到: {save_path}")
        print(f"Graphs数量: {len(query_batchs)}")
        print(f"每个Graph的query大小: {[len(b) for b in query_batchs[:3]]}...")
        
        return save_path
    
def save_query_batches(root, dataset, query_batchs):
        """
        保存 support_batchs 和 query_batchs 到文件
        
        Args:
            support_batchs: support数据列表
            query_batchs: query数据列表
            save_dir: 保存目录
            dataset_name: 数据集名称
            num_shots: few-shot 数量
        """
        
        # 保存数据
        save_path = os.path.join(root, f"{dataset}_queries.pt")
        
        data_to_save = {
            'query_batchs': query_batchs,
            'metadata': {
                'dataset': dataset,
                'num_graphs': len(query_batchs),
                # 'support_shape': [len(batch) for batch in support_batchs[:5]],  # 前5个的形状
                # 'query_shape': [len(batch) for batch in query_batchs[:5]]
            }
        }
        
        torch.save(data_to_save, save_path)
        print(f"数据已保存到: {save_path}")
        print(f"Graphs数量: {len(query_batchs)}")
        print(f"每个Graph的query大小: {[len(b) for b in query_batchs[:3]]}...")
        
        return save_path
# 使用示例
if __name__ == "__main__":
    # 配置参数
    root = "./DataSet/"
    dataset = "citeseer"  # 可选：'facebook', 'twitter', 'cora', 'citeseer'
    single_graph = True
    min_community_size = 16
    min_community_num = 2
    
    
    if single_graph:
        query_loader = CommunityQueryLoader(root, dataset, min_community_size, min_community_num)
        if dataset in ['facebook', 'twitter']:
            graphs = query_loader.load_multi_graph()
            gn=len(graphs)
            query_batchs = []
            print("len:{}".format(gn))
            for graph in graphs:
                queries = query_loader._generate_queries(graph, query_size=9)
                query_batchs.append(queries)
            random.shuffle(query_batchs)
            query_loader.save_query_batches(query_batchs)
        elif dataset in ['cora', 'citeseer']:
            graph = query_loader.load_single_graph(dataset)
            query_batchs = []
            queries = query_loader._generate_queries(graph, query_size=100)
            query_batchs.append(queries)
            query_loader.save_query_batches(query_batchs)
    else:
        query_batchs = []
        query_loader1 = CommunityQueryLoader(root, "twitter", min_community_size, min_community_num)
        graphs = query_loader1.load_multi_graph()
        gn=len(graphs)
        print("len:{}".format(gn))
        for graph in graphs:
            queries = query_loader1._generate_queries(graph, query_size=9)
            query_batchs.append(queries)
        query_loader2 = CommunityQueryLoader(root, "facebook", min_community_size, min_community_num)
        graphs = query_loader2.load_multi_graph()
        gn=len(graphs)
        print("len:{}".format(gn))
        for graph in graphs:
            queries = query_loader2._generate_queries(graph, query_size=9)
            query_batchs.append(queries)
        save_query_batches(root, "twitter2facebook", query_batchs)
