from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from data_load import seed_all
from train_eval import IACS
from Model import CSIACSComp
import torch.optim as optim
import torch
import numpy as np
# import wandb
from tqdm import tqdm
from tqdm.contrib import tzip
import os
import time
import networkx as nx
from torch.utils.data import DataLoader
from signal import signal, SIGPIPE, SIG_DFL
from QueryGenerate import TaskData, RawQuery
signal(SIGPIPE,SIG_DFL)

def load_query_batches(save_dir, dataset_name):
    """
    从文件加载 support_batchs 和 query_batchs
    
    Args:
        save_dir: 保存目录
        dataset_name: 数据集名称
        num_shots: few-shot 数量
        
    Returns:
        support_batchs, query_batchs
    """
    load_path = os.path.join(save_dir, f"{dataset_name}_queries.pt")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"文件不存在: {load_path}")
    
    loaded_data = torch.load(load_path)

    query_batchs = loaded_data['query_batchs']
    
    print(f"从 {load_path} 加载数据成功")
    print(f"Graphs数量: {len(query_batchs)}")
    print(f"第一个Graph的query大小: {len(query_batchs[0]) if query_batchs else 0}")
    
    return query_batchs

# 'graph': {
#     'adj' : graph['adj'],
#     'num_nodes': graph['num_nodes'],
#     'node_embeddings': graph['node_embeddings'],
#     'attr_embeddings': graph['attr_embeddings'],
#     'edge_index': graph['edge_index'],
#     'num_attributes': graph['num_attributes'],
#     'attributes': graph['attributes'],
#     'hyperedge_index': graph['hyperedge_index']
#     },
# 'query': {
#     'nodes': query_nodes,
#     'attrs': query_attrs
#     # 'nodes_mask': query_nodes_mask,
#     # 'attrs_mask': query_attrs_mask
# },
# 'target': {
#     'community_indices': community,
#     'community_mask': community_mask
# }
def main(args):
    seed_all(args.seed)
    wandb_run = None
    # =wandb.init(config=args,project='IACS',dir='/home/shfang/IACS/wandb/',job_type="training",name="fe01{}_{}_train{}_valid{}_test{}_pos{}_neg{}".format(args.data_set,args.meta_method,args.train_task_num,args.valid_task_num,args.test_task_num,args.num_pos,args.num_neg),reinit=True)
    # task_size = args.task_size
    num_shots = args.num_shots
    num_pos, num_neg = args.num_pos, args.num_neg
    query_batches = load_query_batches(args.data_dir, args.data_set)
    node_feat = 128
    tasks = []
    for query_batch in query_batches:
        all_queries_data = list()
        for query in query_batch:
            graph = query['graph']
            g = RawQuery(graph['edge_index'], graph['attributes'], graph['attr_embeddings'], graph['node_embeddings'], True)
            q = query['query']
            t = query['target']
            all_queries_data.append(g.get_one_IACS_attribute_query_tensor(q['nodes'], q['attrs'], t['community_indices'],num_pos, num_neg))
        tasks.append(TaskData(all_queries_data, num_shots=num_shots))
    tn=len(tasks)
    print("len:{}".format(tn))
    train_tasks, valid_tasks, test_tasks = tasks[0:int(tn*0.6)], tasks[int(tn*0.6):int(tn*0.8)], tasks[int(tn*0.8):]
    # train_tasks, test_tasks, valid_tasks = tasks[0:int(tn*0.6)], tasks[int(tn*0.6):int(tn*0.8)], tasks[int(tn*0.8):]
    # # if args.data_set=='twitter':
    # if args.data_set in ('facebook', 'twitter'):
    #     #attribute community search
    #     raw_data_list = load_twitter_graphs(args, args.data_dir+args.data_set,use_embed_feats=args.use_embed_feats)#graph, communities, feats, embed_feats, x_embed_feats
    #     node_feat= 128
    #     raw_data_list = [raw_data for raw_data in raw_data_list # filter invalid raw data
    #                  if raw_data.num_communities > 1 and raw_data.num_query_attributes > 0]
    #     communities_list = [raw_data.get_communities(task_size, num_shots) for raw_data in raw_data_list]#在每个图里选择task_size个查询
    #     tasks = [raw_data.get_attributed_task(community_ids, num_shots, args.meta_method, num_pos, num_neg)#在task_size个查询里选num_shots个查询做支持集，剩下的做查询集
    #             for raw_data, community_ids in zip(raw_data_list, communities_list)]
    #     tn=len(tasks)
    #     print("len:{}".format(tn))
    #     for tk in tasks:
    #         print(len(tk.all_queries_data))
    #         print(f'support:{tk.num_support}, query:{tk.num_query}')
    #     train_tasks, valid_tasks, test_tasks = tasks[0:int(tn*0.7)], tasks[int(tn*0.7):int(tn*0.8)], tasks[int(tn*0.8):]


    # # elif args.data_set=='arxiv':
    # else:
    #     # 一个task对应一个图，一个task下有一组查询（task_size个查询），shot是每组查询的支持集大小，剩下的作为查询集，每个查询有1-3个节点
    #     #                                                       -|          -| 
    #     #                                                        |           |
    #     #task(graph) ->    采样query(,,) pos(,,) neg(,,)          |->  shot  |->  task_size
    #     #                                                       -|           |
    #     #                                                                   -|
    # # task集划分6:1:3，每个task作为一个batch进行训练，验证，测试
    #     if args.data_set in ('cora', 'citeseer'):
    #     #Non-attribute community search
    #     # raw_data_list_train, raw_data_list_valid, raw_data_list_test, node_feat = load_arxiv(args)
    #         raw_data_list_train, raw_data_list_valid, raw_data_list_test, node_feat = load_single_graph(args, feat=True)#把一个图进行分割，分割为任务数个子图，即一个task对应一个图，并按6:1:3进行划分训练集、验证集、测试集
    #     else:
            # raw_data_list_train, raw_data_list_valid, raw_data_list_test, node_feat = load_single_graph(args, feat=False)
    #     print('get queries!')
    #     queries_list_train = [raw_data.get_communities(task_size, num_shots) for raw_data in raw_data_list_train]#从子图里的节点-社区对中选出task-size个节点（看作社区的索引）去生成查询
    #     queries_list_valid = [raw_data.get_communities(task_size, num_shots) for raw_data in raw_data_list_valid]
    #     queries_list_test = [raw_data.get_communities(task_size, num_shots) for raw_data in raw_data_list_test]
    #     print('get tasks!')
    #     train_tasks = [raw_data.get_attributed_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
    #                tzip(raw_data_list_train, queries_list_train)]
    #     valid_tasks=[raw_data.get_attributed_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
    #               tzip(raw_data_list_valid, queries_list_valid)]
    #     test_tasks = [raw_data.get_attributed_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
    #               tzip(raw_data_list_test, queries_list_test)]
        
    print('train_tasks: {}, valid_tasks: {}, test_tasks: {}'.format(len(train_tasks), len(valid_tasks), len(test_tasks)))
    if args.meta_method in ["IACS", "iacs"]:
        if args.use_embed_feats==False:
            print('----------IACS for Non-attribute----------')
            model = CSIACSComp(args, node_feat_dim=node_feat + 1, edge_feat_dim=10, decoder_type=args.decoder_type)
            print(model)
            IACS1 = IACS(args, model, wandb_run)
            print('begin training...')
            t=IACS1.train_IACS(train_tasks,valid_tasks,test_tasks)#支持集编码，查询集解码
            print('begin test!')
            acc, precision, recall, f1, acc_com, precision_com, recall_com, f1_com,t2=IACS1.evaluate_IACS(test_tasks,args.epochs)#支持集测试
            print('train_time={:.4f}, test_time={:.4f}'.format(t,t2))
        else:
            print('----------IACS for Attribute CS----------')
            model = CSIACSComp(args, node_feat_dim=node_feat + 1, edge_feat_dim=10, decoder_type=args.decoder_type)
            print(model)
            IACS1 = IACS(args, model, wandb_run)
            print('begin training...')
            t=IACS1.train_IACS(train_tasks,valid_tasks,test_tasks)
            print('begin test!')
            # acc, precision, recall, f1, acc_com, precision_com, recall_com, f1_com,t2=IACS1.evaluate_IACS(test_tasks,args.epochs)
            acc, precision, recall, f1,t2=IACS1.evaluate_IACS(test_tasks,args.epochs)
            print('train_time={:.4f}, test_time={:.4f}'.format(t,t2))


if __name__ == "__main__":
    parser = ArgumentParser("IACS", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--num_layers", default=3, type=int,
                        help="number of gnn conv layers")
    parser.add_argument("--gnn_type", default="GAT", type=str, # GCN, GAT, GATBias
                        help="GNN type")
    parser.add_argument("--pool_type", default="avg", type=str,  # att, sum, avg
                        help="IACS Context Pool Type")
    parser.add_argument("--decoder_type", default="IP", type=str,
                        help="IACS Decoder Type")
    parser.add_argument("--film_type", default="no", type=str,  # gate, no, plain
                        help="Context FiLM Layer Type")
    parser.add_argument("--gnn_act_type", default="relu", type=str,
                        help="activation layer inside gnn aggregate/combine function")
    parser.add_argument("--act_type", default="relu", type=str,
                        help="activation layer function for MLP and between GNN layers")
    parser.add_argument("--embed_type", default="prone", type=str,
                        help="the node feature encoding type")
    parser.add_argument("--num_g_hid", default=128, type=int,
                        help="hidden dim for transforming nodes")
    parser.add_argument("--num_e_hid", default=128, type=int,
                        help="hidden dim for transforming edges")
    parser.add_argument("--gnn_out_dim", default=128, type=int,
                        help="number of output dimension")
    parser.add_argument("--mlp_hid_dim", default=512, type=int,
                        help="number of hidden units of MLP")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--batch_norm", default=False, type=bool)
    parser.add_argument("--use_embed_feats", action='store_true', default=True, help="input use the embed features")

    #Settings
    parser.add_argument("--meta_method", default="IACS", type=str,
                        help="The meta learning algorithm")
    parser.add_argument("--task_size", help='the number of query on a task', default=9, type=int)
    parser.add_argument("--num_shots", default=4, type=int)
    parser.add_argument("--num_pos", default=0.05, type=float)
    parser.add_argument("--num_neg", default=0.05, type=float)
    parser.add_argument("--train_task_num", type=int, help='the number of training task', default=128)
    parser.add_argument("--valid_task_num", type=int, help='the number of training task', default=32)
    parser.add_argument("--test_task_num", type=int, default=32, help='the number of test task')
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default= 1e-5, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--scheduler_type", default="exponential", type=str,
                        help="the node feature encoding type")
    parser.add_argument('--decay_factor', type=float, default=0.8,
                        help='decay rate of (gamma).')
    parser.add_argument('--decay_patience', type=int, default=10,
                        help='num of epochs for one lr decay.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--num_workers', type = int, default= 16,
                        help='number of workers for Dataset.')

    # Input/Output dir
    # parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--data_dir", type=str, default="../DataSet/")
    parser.add_argument("--project_dir", type=str, default="./")
    parser.add_argument("--data_set", default='facebook', type=str, help='dataset')
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--Test", default=False, action='store_true')
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    # set the hardware parameter
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:1' if args.cuda else 'cpu')

    if args.verbose:
        print(args)
    main(args)

