import os
import networkx as nx
import numpy as np
import random
import torch
# from QueryGenerate import RawGraphWithCommunity
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
import torch_geometric
import nxmetis
from tqdm import tqdm
import time
from ogb.nodeproppred import PygNodePropPredDataset



def seed_all(seed: int =0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("set all seed!")

def evaluate_prediction(pred, targets):
    acc = accuracy_score(targets, pred)
    precision = precision_score(targets, pred)
    recall = recall_score(targets, pred)
    f1 = f1_score(targets, pred)
    return acc, precision, recall, f1

def np_save_if_not_existed(path, saved_data):
    if not os.path.exists(path):
        saved_data_numpy = np.array([saved_data], dtype=object)
        np.save(path, saved_data_numpy)


# def load_twitter_graphs(args, data_dir: str, use_embed_feats = False):
#     raw_data_list = list()
#     attr_info_list=list()
#     for file_name in os.listdir(data_dir):
#         if os.path.splitext(file_name)[1] != '.edges':
#             continue
#         ego_node_id = int(os.path.splitext(file_name)[0])
#         feat_dict = dict()
#         with open(os.path.join(data_dir, "{}.featnames".format(ego_node_id)), 'r') as feat_names:
#             for line in feat_names:
#                 tokens = line.strip().split()
#                 f_id = int(tokens[0])
#                 feat_name = '+'.join(tokens[1:])
#                 feat_dict[f_id] = feat_name
#             feat_names.close()

#         # load attributes
#         node_id_dict = dict()
#         node_attrs_dict=dict()
#         node_cnt = 0
#         with open(os.path.join(data_dir, "{}.feat".format(ego_node_id)), 'r') as feat:
#             lines = feat.readlines()
#             feats = np.zeros(shape=(len(lines) + 1, len(feat_dict)), dtype=float)
#             for line in lines:
#                 tokens = line.strip().split()
#                 node_attrs_dict[node_cnt]=list()
#                 for i, val in enumerate(tokens[1:]):
#                     if int(val) <= 0:
#                         continue
#                     idx = i
#                     feats[node_cnt][idx] = 1
#                     node_attrs_dict[node_cnt].append(str(i))
#                 node_id_dict[int(tokens[0])] = node_cnt
#                 node_cnt += 1
#             feat.close()
#         with open(os.path.join(data_dir, "{}.egofeat".format(ego_node_id)), 'r') as egofeat:
#             node_id_dict[ego_node_id] = node_cnt
#             node_attrs_dict[node_cnt]=list()
#             for line in egofeat:
#                 tokens = line.strip().split()
#                 for i, val in enumerate(tokens):
#                     if int(val) <= 0:
#                         continue
#                     idx = i
#                     feats[node_cnt][idx] = 1
#                     node_attrs_dict[node_cnt].append(str(i))
#             egofeat.close()

#         # load graph edges:
#         edge_list = list()
#         with open(os.path.join(data_dir, "{}.edges".format(ego_node_id)), "r") as edges:
#             for line in edges:
#                 tokens = line.strip().split()
#                 src, dst = int(tokens[0]), int(tokens[1])
#                 edge_list.append((node_id_dict[src], node_id_dict[dst]))
#             edges.close()
#             ego_edges = [(node_id_dict[ego_node_id], k) for k in node_id_dict.values()]
#             edge_list += ego_edges

#         # load communities info
#         communities = list()
#         with open(os.path.join(data_dir, "{}.circles".format(ego_node_id)), 'r') as circles:
#             for line in circles:
#                 tokens = line.strip().split()
#                 node_ids = [node_id_dict[int(token)] for token in tokens[1:]]
#                 communities.append(node_ids)
#             circles.close()
#         if len(communities)<1:
#             continue

#         graph = nx.Graph()
#         graph.add_edges_from(edge_list)
#         # filter those graphs that are not connected
#         if not nx.is_connected(graph):
#             print('skip')
#             continue
#         print("# of nodes/edges:", graph.number_of_nodes(), graph.number_of_edges(),"id:",ego_node_id)

#         embed_feats_path=data_dir+"/{}_emb.npy".format(ego_node_id)
#         embed_feats = torch.from_numpy(np.load(embed_feats_path,allow_pickle=True))
#         x_embed_feats = torch.zeros(size=(graph.number_of_nodes(), embed_feats.size(-1)), dtype=torch.float)
#         for _,node in enumerate(node_id_dict.keys()):
#             attrs_idx = torch.nonzero(torch.from_numpy(feats[node_id_dict[node]]))[:, 0].tolist()
#             if len(attrs_idx) <= 0:
#                 continue
#             x_embed_feats[node_id_dict[node]] = torch.mean(embed_feats[attrs_idx], dim=0, keepdim=False)
#         raw_data = RawGraphWithCommunity(graph, communities, feats, embed_feats, x_embed_feats,use_embed_feats=use_embed_feats, min_community_size=16)
#         if raw_data.number_of_queries<args.task_size:
#             print('skip')
#             continue
#         raw_data_list.append(raw_data)
#         attr_info_list.append((node_attrs_dict, edge_list))
#     print(len(raw_data_list))
#     return raw_data_list

# def load_custom_dataset(args, feat = True):
#     """
#     从自定义文件加载图数据
    
#     Args:
#         edge_file: 边文件路径，每行格式: node1 node2
#         feature_file: 特征文件路径，每行格式: node_id feat1 feat2 ... feat_n
#         label_file: 标签文件路径，每行格式: label node1 node2 ... node_k
#     """
#     print(f"Loading labels ...")
#     label_nodes_dict = {}
#     cmty_file = os.path.join(args.data_dir, args.data_set, f"{args.data_set}_cmty.txt")
#     with open(cmty_file, 'r') as f:
#         for line_num, line in enumerate(f, 1):
#             parts = line.strip().split()
#             if len(parts) < 2:  # 至少需要：1个标签 + 1个节点
#                 print(f"Warning: Line {line_num} has too few elements: {line}")
#                 continue
            
#             try:
#                 label = int(parts[0])  
#                 nodes = [int(x) for x in parts[1:]]  
#                 label_nodes_dict[label] = nodes
                
#             except ValueError as e:
#                 print(f"Warning: Error parsing line {line_num}: {e}")
#                 continue
    
#     print(f"Loading features ...")   
#     # 使用字典存储节点信息，因为节点ID可能不连续
#     node_features_dict = {}  # node_id -> features
#     num_features = None
#     if feat:
#         feat_file = os.path.join(args.data_dir, args.data_set, f"{args.data_set}_feat.txt")
#         with open(feat_file, 'r') as f:
#             for line_num, line in enumerate(f, 1):
#                 parts = line.strip().split()
#                 if len(parts) < 2:  # 至少需要：node_id + 1个特征
#                     print(f"Warning: Line {line_num} has too few elements: {line}")
#                     continue
                
#                 try:
#                     node_id = int(parts[0])  # 第一个元素是节点ID
#                     features = [float(x) for x in parts[1:]]  # 中间是特征
                    
#                     # 检查特征维度是否一致
#                     if num_features is None:
#                         num_features = len(features)
#                     elif len(features) != num_features:
#                         print(f"Warning: Line {line_num} has {len(features)} features, expected {num_features}")
#                         continue
                    
#                     node_features_dict[node_id] = features
                    
#                 except ValueError as e:
#                     print(f"Warning: Error parsing line {line_num}: {e}")
#                     continue
        
#         print(f"Loaded {len(node_features_dict)} nodes with {num_features} features each")
    
#     # 读取边信息
#     print(f"Loading edges ...")
#     edges = []
#     edge_set = set()  # 用于去重
#     node_ids = set()
#     edge_file = os.path.join(args.data_dir, args.data_set, f"{args.data_set}.txt")
#     with open(edge_file, 'r') as f:
#         for line_num, line in enumerate(f, 1):
#             parts = line.strip().split()
#             if len(parts) < 2:
#                 print(f"Warning: Line {line_num} in edge file has too few elements")
#                 continue
            
#             try:
#                 src = int(parts[0])
#                 dst = int(parts[1])
#                 node_ids.add(src)
#                 node_ids.add(dst)
                
#                 # 避免重复边（无向图）
#                 edge_tuple = (min(src, dst), max(src, dst)) if src != dst else (src, dst)
#                 if edge_tuple not in edge_set:
#                     edge_set.add(edge_tuple)
#                     edges.append(edge_tuple)
                    
#             except ValueError as e:
#                 print(f"Warning: Error parsing edge line {line_num}: {e}")
#                 continue
    
#     print(f"Loaded {len(edges)} unique edges")
    
#     # 创建节点ID到连续索引的映射
#     all_nodes = sorted(node_ids)
#     node_id_to_index = {node_id: idx for idx, node_id in enumerate(all_nodes)}
#     index_to_node_id = {idx: node_id for node_id, idx in node_id_to_index.items()}
    
#     # 创建特征矩阵和标签向量
#     num_nodes = len(all_nodes)
#     features_matrix = np.zeros((num_nodes, num_features), dtype=np.float32)
#     embed_feats, x_embed_feats = [], []
#     if feat:
#         for idx, node_id in enumerate(all_nodes):
#             features_matrix[idx] = node_features_dict[node_id]
        
#         embed_feats_path=os.path.join(args.data_dir, args.data_set, f"{args.data_set}_emb.npy")
#         # data_dir+"/emb/egotwitter_enhanced{}.emb.npy".format(ego_node_id)
#         embed_feats = torch.from_numpy(np.load(embed_feats_path,allow_pickle=True))
#         # print(embed_feats.shape)
#         # print(embed_feats)
#         x_embed_feats = torch.zeros(size=(num_nodes, embed_feats.size(-1)), dtype=torch.float)
#         # print(x_embed_feats.shape)
#         # print(x_embed_feats)
#         for _,node in enumerate(node_id_to_index.keys()):
#             attrs_idx = torch.nonzero(torch.from_numpy(features_matrix[node_id_to_index[node]]))[:, 0].tolist()
#             if len(attrs_idx) <= 0:
#                 continue
#             x_embed_feats[node_id_to_index[node]] = torch.mean(embed_feats[attrs_idx], dim=0, keepdim=False)
    
#     label_nid_dict = {}
#     for label, nodes in label_nodes_dict.items():
#         nids = []
#         for node_id in nodes:
#             idx = node_id_to_index[node_id]
#             nids.append(idx)
#         label_nid_dict[label] = nids
#             # labels_list[idx] = label_nodes
#             # print(node_id,label_nodes)
#     #         features_matrix[idx] = node_features_dict[node_id]
    
    
#     # 转换为Tensor
#     x = torch.tensor(features_matrix, dtype=torch.float32)  # [num_nodes, num_features]
#     # y = torch.tensor(labels_list, dtype=torch.long).unsqueeze(1)  # [num_nodes, 1]
    
#     # 映射边到连续索引
#     mapped_edges = []
#     for src_id, dst_id in edges:
#         src_idx = node_id_to_index[src_id]
#         dst_idx = node_id_to_index[dst_id]
#         mapped_edges.append([src_idx, dst_idx])
    
#     # 创建edge_index（双向边，因为是无向图）
#     edge_index = []
#     for src_idx, dst_idx in mapped_edges:
#         edge_index.append([src_idx, dst_idx])
#         edge_index.append([dst_idx, src_idx])  # 添加反向边
    
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
#     # 创建PyG Data对象
#     from torch_geometric.data import Data
#     data_pyg = Data(x=x, edge_index=edge_index)
#     data_pyg.num_nodes = num_nodes
#     # data_pyg.original_node_ids = index_to_node_id  # 保存原始ID映射
    
#     # print(f"Created PyG Data object:")
#     # print(f"  Nodes: {data_pyg.num_nodes}")
#     # print(f"  Edges: {edge_index.shape[1] // 2} (undirected)")  # 除以2因为添加了双向边
#     # print(f"  Features: {x.shape[1]}")
#     # print(f"  Classes: {len(np.unique(labels_list))}")    

#     return data_pyg, label_nid_dict, embed_feats, x_embed_feats
# def load_single_graph(args, feat = True):
    
#     training_raw_data_list =  list()
#     valid_raw_data_list = list()
#     test_raw_data_list=list()
    
#     data, glob_communities, embed_feats_all, x_embed_feats_all =load_custom_dataset(args, feat)
#     graph=torch_geometric.utils.to_networkx(data,to_undirected=True)
#     if not hasattr(graph, 'node'):
#         graph.node = graph.nodes
#     print(graph.number_of_nodes(), graph.number_of_edges())

#     print("begin partition---")
#     time_begin=time.time()
#     obj,subgraph_nodes=nxmetis.partition(graph,args.train_task_num+args.valid_task_num+args.test_task_num)
#     time_end=time.time()
#     print("end partition---")
#     print("time cost:{}".format(time_end-time_begin), "num of subgraphs:", len(subgraph_nodes))

#     #generate training/valid/test task
#     # 这里的一个for循环是一个task，每个task包含多个query，每个query是一个社区
#     for idx in tqdm(range(len(subgraph_nodes))):
#         # raw_data_name = f"arxiv_subgraph_of_{idx}.npy"
#         # raw_data_path = os.path.join(args.project_dir, 'saved_subgraph_arxiv', raw_data_name)
#         node_list=subgraph_nodes[idx]
#         print("subgraph node length:", len(node_list))
#         edge_list=graph.subgraph(node_list).edges()
#         old_graph=nx.Graph()
#         old_graph.add_edges_from(edge_list)
#         if not nx.is_connected(old_graph):
#             print('not connected, skip')
#             continue
#         node_id_dict = {l: n_id for n_id, l in enumerate(old_graph.nodes)}#key original id; value new id;
#         node_new2old = {n_id:l for n_id, l in enumerate(old_graph.nodes)}
#         res_graph=nx.Graph()
#         edge_list = [(node_id_dict[src], node_id_dict[dst]) for (src, dst) in edge_list]
#         res_graph.add_edges_from(edge_list)
#         # node_list=list(res_graph.nodes())
#         node_list=[node_new2old[l] for n_id,l in enumerate(res_graph.nodes)]
#         feats = []
#         embed_feats=[]
#         x_embed_feats=[]
#         if feat:
#             embed_feats= x_embed_feats_all[node_list]
#             x_embed_feats= x_embed_feats_all[node_list]
#             feats=data.x[node_list].numpy()
#         communities=list()
#         candidate_query_number=0
#         for k, val in glob_communities.items():#key:label value:node id
#             temp_comm = set(val).intersection(set(node_list))  # get the local community induced by node_list
#             temp_comm = [node_id_dict[node] for node in temp_comm]
#             communities.append(temp_comm)
#             candidate_query_number=candidate_query_number+len(temp_comm)
#         if len(communities)<2:
#             continue
        
        
        
#         raw_data=RawGraphWithCommunity(res_graph, communities, feats, embed_feats, x_embed_feats)
#         print("number_of_queries:",raw_data.number_of_queries)
#         # np_save_if_not_existed(raw_data_path, raw_data)

#         if raw_data.number_of_queries<args.task_size:
#             print('query is not enough in a task, skip')
#             continue

#         if idx % 10 ==0 or idx % 10==1 or idx % 10==2:
#             test_raw_data_list.append(raw_data)
#         elif idx % 10 ==3:
#             valid_raw_data_list.append(raw_data)
#         else:
#             training_raw_data_list.append(raw_data)
#         print("subgraph of {} size: {}, {},length of communities {},feats size ({},{})".format(idx,res_graph.number_of_nodes(), res_graph.number_of_edges(),len(communities),feats.shape[0],feats.shape[1]))
#     num_feat=feats.shape[1]
#     print(f"training_raw_data_num: {len(training_raw_data_list)}, valid_raw_data_list: {len(valid_raw_data_list)}, test_raw_data_list: {len(test_raw_data_list)}")
#     return training_raw_data_list, valid_raw_data_list, test_raw_data_list, num_feat


