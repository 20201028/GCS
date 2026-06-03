#! usr/bin/python
import datetime
import random
import time
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
# import numpy as np
# import networkx as nx
import os
import os.path as osp

import torch

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
torch.set_num_threads(10)

# import kcore
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from GFCN import QD_GCN, CS_GCN, SimpleCS_GCN, CS_GCN_NoF
# from pygcn import GFCN, GCNAtt, GCNAtt_my,ResDeepGCN,DenseDeepGCN,GCN_BN,ResDeepGCN_BN8,\
#     ResDeepGCN_BN16, DenseDeepGCN_BN16, ResDeepGCN_BN16_ATN,ResGCN_BN,Self_ResGCN_BN,ResGCN_BN_ADD,ResGCN_BN_Update
# from SampleLoader_F import EmailDataset,SPcoraDataset,PhilDataset
#from datasetLoad import GraphDataset, WebKBDataset, GraphDataset_, GraphDataset__, GraphDataset___
from torch.utils.data.dataset import Dataset
import numpy as np
import numpy.random as npr
import networkx as nx
# import metric
from metric import *
import normalization as Norm
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score

class GraphDataset_(Dataset):
    def __init__(self, data_dir='./data/facebook_all_/', phase='train', ego=686, method='ATC'):
        super(GraphDataset_, self).__init__()

        data_dir = './data/facebook_all_/'
        data_dir = './data/'

        ego_feat = np.genfromtxt("{}{}.egofeat".format(data_dir, ego), dtype=np.dtype(str))
        feat = np.genfromtxt("{}{}.feat".format(data_dir, ego), dtype=np.dtype(str))
        ego_feat = ego_feat.astype(np.int32)
        feat = feat.astype(np.int32)
        features = np.vstack((ego_feat, feat[:, 1:]))
        print("all feature num", features.sum(), features.sum() / features.shape[0])
        # print("1 feature shape", features.shape)
        features = self.fnormalize(features)

        node_map = {}
        node_map[int(ego)] = 0
        for i in range(feat.shape[0]):
            node_map[feat[i, 0]] = i

        edges = np.genfromtxt("{}{}.edges".format(data_dir, ego), dtype=np.dtype(str))
        edges = edges.astype(np.int32)
        Edges = []

        for i in range(feat.shape[0]):
            Edges.append((node_map[int(ego)], i + 1))
        for e in edges:
            Edges.append((node_map[e[0]], node_map[e[1]]))

        G = nx.Graph()
        G.add_edges_from(Edges)
        A = nx.to_numpy_matrix(G)  # nx.adjacency_matrix(G).todense()
        A = np.array(A, dtype=np.float32)

        Adj = A

        for i in range(features.shape[0]):
            Adj[i, i] = 0
            A[i, i] = 1

        A = Norm.normalized_adjacency(A)

        feats = features
        adjs = A

        cur_in = np.genfromtxt("{}{}.samples".format(data_dir, ego), dtype=np.dtype(str))
        cur_in = cur_in.astype(np.int32)
        cur_out = np.genfromtxt("{}{}.labels".format(data_dir, ego), dtype=np.dtype(str))
        cur_out = cur_out.astype(np.int32)
        print("label shape", cur_out.shape)
        attr = cur_in

        samples_in = []
        samples_out = []
        samples_att = []

        if phase == 'train':
            for i in range(len(cur_in)):
                if i % 7 == 0 or i % 7 == 5 or i % 7 == 6:
                    samples_in.append(cur_in[i:i + 1])
                    samples_out.append(cur_out[i:i + 1])
                    samples_att.append(attr[i:i + 1])
        if phase == 'eval':
            for i in range(len(cur_in)):
                if i % 7 == 1 or i % 7 == 4:
                    samples_in.append(cur_in[i:i + 1])
                    samples_out.append(cur_out[i:i + 1])
                    samples_att.append(attr[i:i + 1])
        if phase == 'test1':
            for i in range(len(cur_in)):
                if i % 7 == 2 or i % 7 == 3:
                    samples_in.append(cur_in[i:i + 1])
                    samples_out.append(cur_out[i:i + 1])
                    samples_att.append(attr[i:i + 1])
        samples_in = np.concatenate(samples_in, 0)
        samples_out = np.concatenate(samples_out, 0)
        samples_att = np.concatenate(samples_att, 0)

        self.feats = feats
        self.adjs = adjs

        self.Adj = Adj
        self.samples_in = samples_in
        self.samples_out = samples_out
        self.samples_att = samples_att
        self.ego = ego
        self.phase = phase

    def __len__(self):
        return self.samples_in.shape[0]

    def __getitem__(self, item):

        cur_in = self.samples_in.copy()[item, :]
        cur_out = self.samples_out.copy()[item, :]
        cur_att = self.samples_att.copy()[item, :]

        cur_in = cur_in[:, np.newaxis]  # BN1
        cur_out = cur_out[:, np.newaxis]  # BN1
        cur_att = cur_att[:, np.newaxis]  # BN1

        cur_adj = self.adjs.copy()
        feats = self.feats.copy()

        input = cur_in
        return input, cur_att, cur_adj, feats, cur_out, self.Adj

    def fnormalize(self, mx):
        """Row-normalize sparse matrix"""

        mx = mx.transpose(0, 1)
        print("mx shape", mx.shape)
        rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        rowsum[rowsum == 0] = 1
        # print("rowsum shape", rowsum.shape)
        print("rowsum", rowsum[:24])
        mx = mx / rowsum[:, np.newaxis]
        mx = mx.transpose(0, 1)
        return mx

    def normalized_adjacency(self, adj):
        # adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt))

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(rowsum)
        r_inv = 1 / rowsum
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

        # rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        # rowsum[rowsum==0] = 1
        # mx = mx/rowsum
        # return mx

class GraphDataset(Dataset):
    def __init__(self, phase='train', feat=None,method=None, args=None, lists=None, G=None, number=50):
        super(GraphDataset, self).__init__()

        A = nx.to_numpy_matrix(G)  # nx.adjacency_matrix(G).todense()
        A = np.array(A, dtype=np.float32)

        Adj = A

        for i in range(len(feat)):
            Adj[i, i] = 0
            A[i, i] = 1

        feature = self.fnormalize(feat)

        # A=self.normalize(A)
        A = Norm.normalized_adjacency(A)
        #print("A", A.shape)

        samples_in = []
        samples_out = []
        samples_att = []
        print("--------------------", len(lists), number)
        if phase == 'train':
            #print(phase, len(lists))
            cur_in = np.zeros((number, feature.shape[0]), dtype=np.int32)
            cur_out = np.zeros((number, feature.shape[0]), dtype=np.int32)
            i = 0
            #print(lists[0])
            for qlist, comm in lists:
                if i>=number:
                    break
                for q in qlist:
                    cur_in[i][q] = 1
                for u in comm:
                    cur_out[i][u] = 1
                i = i + 1
            cur_in = cur_in.astype(np.int32)
            cur_out = cur_out.astype(np.int32)
            attr = np.zeros((cur_out.shape[0], feature.shape[0]), dtype=np.int32)
            attr = attr.astype(np.int32)
            for i in range(args.train_size):
                samples_in.append(cur_in[i:i + 1])
                samples_out.append(cur_out[i:i + 1])
                samples_att.append(attr[i:i + 1])

        if phase == 'eval':
            #print(phase, len(lists))
            cur_in = np.zeros((number, feature.shape[0]), dtype=np.int32)
            cur_out = np.zeros((number, feature.shape[0]), dtype=np.int32)
            i = 0
            for qlist, comm in lists:
                if i>=number:
                    break
                for q in qlist:
                    cur_in[i][q] = 1
                for u in comm:
                    cur_out[i][u] = 1
                i = i + 1
            cur_in = cur_in.astype(np.int32)
            cur_out = cur_out.astype(np.int32)
            attr = np.zeros((cur_out.shape[0], feature.shape[0]), dtype=np.int32)
            attr = attr.astype(np.int32)
            for i in range(args.eval_size):
                samples_in.append(cur_in[i:i + 1])
                samples_out.append(cur_out[i:i + 1])
                samples_att.append(attr[i:i + 1])

        if phase == 'test':
            #print(phase, len(lists))
            cur_in = np.zeros((number, feature.shape[0]), dtype=np.int32)
            cur_out = np.zeros((number, feature.shape[0]), dtype=np.int32)
            i = 0
            for qlist, comm in lists:
                if i>=number:
                    break
                for q in qlist:
                    cur_in[i][q] = 1
                for u in comm:
                    cur_out[i][u] = 1
                i = i + 1
            cur_in = cur_in.astype(np.int32)
            cur_out = cur_out.astype(np.int32)
            attr = np.zeros((cur_out.shape[0], feature.shape[0]), dtype=np.int32)
            attr = attr.astype(np.int32)
            for i in range(args.test_size):
                samples_in.append(cur_in[i:i + 1])
                samples_out.append(cur_out[i:i + 1])
                samples_att.append(attr[i:i + 1])

        # '''
        samples_in = np.concatenate(samples_in, 0)
        samples_out = np.concatenate(samples_out, 0)
        samples_att = np.concatenate(samples_att, 0)
        #print(len(samples_in))

        self.feats = feature
        self.adjs = A
        self.Adj = Adj

        self.samples_in = samples_in
        self.samples_out = samples_out
        self.samples_att = samples_att
        self.phase = phase

    def __len__(self):
        # return len(self.circle)
        # return 9
        return self.samples_in.shape[0]

    def __getitem__(self, item):

        cur_in = self.samples_in.copy()[item,:]
        cur_out = self.samples_out.copy()[item,:]
        cur_att = self.samples_att.copy()[item, :]

        # degree=self.degree[:,np.newaxis]
        # core = self.core[:, np.newaxis]
        # cluster = self.cluster[:, np.newaxis]
        # triangle=self.triangle[:, np.newaxis]

        cur_in = cur_in[:,np.newaxis] # BN1
        cur_out = cur_out[:,np.newaxis] # BN1
        cur_att = cur_att[:, np.newaxis]  # BN1
        cur_adj = self.adjs.copy()
        feats = self.feats#.copy()
        # input=np.concatenate((self.feats[ego].copy(), cur_dis), axis=1)  # BN(D+2)


        # input = cur_in
        # input=np.concatenate((input, cur_in), axis=1) # BN(D+1)

        # input=cur_dis
        # input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)
        #
        input = cur_in
        #input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)

        # return torch.FloatTensor(cur_feats), torch.FloatTensor(cur_out), torch.FloatTensor(cur_adj)
        return input, cur_att,cur_adj,feats, cur_out,self.Adj#self.savelable.copy()

    def fnormalize(self, mx):
        """Row-normalize sparse matrix"""

        mx = mx.transpose(0, 1)
        #print("mx shape", mx.shape)
        rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        rowsum[rowsum == 0] = 1
        # print("rowsum shape", rowsum.shape)
        #print("rowsum", rowsum[:24])
        mx = mx / rowsum[:, np.newaxis]
        mx = mx.transpose(0, 1)
        return mx

    def normalized_adjacency(self,adj):
        # adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt))

    def loadQuerys(self, dataset, root, train_n, val_n, test_n, train_path, val_path, test_path):
        path_train = root + dataset + '/' + dataset + train_path
        if not os.path.isfile(path_train):
            raise Exception("No such file: %s" % path_train)
        train_lists = []
        for line in open(path_train, encoding='utf-8'):
            q, pos, comm = line.split(",")
            q = int(q)
            pos = pos.split(" ")
            pos_ = [int(x) for x in pos if int(x) != q]
            comm = comm.split(" ")
            comm_ = [int(x) for x in comm]
            if len(train_lists) >= train_n+ val_n + test_n:
                break
            train_lists.append((q, pos_, comm_))

        val_lists_ = train_lists[train_n:train_n + val_n]
        test_lists_ = train_lists[train_n + val_n:]
        train_lists = train_lists[:train_n]
        val_lists = []
        for q, pos, comm in val_lists_:
            val_lists.append((q, pos, comm))
        test_lists = []
        for q, pos, comm in test_lists_:
            test_lists.append((q, comm))

        '''
        path_test = root + dataset + '/' + dataset + test_path
        if not os.path.isfile(path_test):
            raise Exception("No such file: %s" % path_test)
        test_lists = []
        for line in open(path_test, encoding='utf-8'):
            q, comm = line.split(",")
            q = int(q)
            comm = comm.split(" ")
            comm_ = [int(x) for x in comm]
            if len(test_lists)>=test_n:
                break
            test_lists.append((q, comm_))

        path_val = root + dataset + '/' + dataset + val_path
        if not os.path.isfile(path_val):
            raise Exception("No such file: %s" % path_val)
        val_lists = []
        for line in open(path_val, encoding='utf-8'):
            q, pos, comm = line.split(",")
            q = int(q)
            pos = pos.split(" ")
            pos_ = [int(x) for x in pos if int(x)!=q]
            comm = comm.split(" ")
            comm_ = [int(x) for x in comm]
            if len(val_lists)>=val_n:
                break
            val_lists.append((q, pos_, comm_))
        #'''

        return train_lists, val_lists, test_lists

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(rowsum)
        r_inv = 1 / rowsum
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

        # rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        # rowsum[rowsum==0] = 1
        # mx = mx/rowsum
        # return mx

def f1_score(sim, comm, s):
    comm_find = []
    for i in range(len(sim)):
        if sim[i] >= s:
            comm_find.append(i)
    comm_find = set(comm_find)
    comm_find = list(comm_find)
    comm = set(comm)
    comm = list(comm)
    lists = [x for x in comm_find if x in comm]
    if len(lists) == 0:
        #print("f1, pre, rec", 0.0, 0.0, 0.0)
        return 0.0, 0.0, 0.0
    pre = len(lists) * 1.0 / len(comm_find)
    rec = len(lists) * 1.0 / len(comm)
    f1 = 2 * pre * rec / (pre + rec)
    #print("f1, pre, rec", f1, pre, rec)
    return f1, pre, rec

def f1_score_(comm_find, comm):

    lists = [x for x in comm_find if x in comm]
    if len(lists) == 0:
        #print("f1, pre, rec", 0.0, 0.0, 0.0)
        return 0.0, 0.0, 0.0
    pre = len(lists) * 1.0 / len(comm_find)
    rec = len(lists) * 1.0 / len(comm)
    f1 = 2 * pre * rec / (pre + rec)
    #print("f1, pre, rec", f1, pre, rec)
    return f1, pre, rec

def NMI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = normalized_mutual_info_score(truthlabel, prelabel)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = adjusted_rand_score(truthlabel, prelabel)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm, n_nodes):
    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = jaccard_score(truthlabel, prelabel)
    #print("q, jac:", score)
    return score

def fnormalize(mx):
    mx = mx.transpose(0, 1)
    #print("mx shape", mx.shape)
    rowsum = mx.sum(1)
    # rowsum = rowsum[:,np.newaxis]
    rowsum[rowsum == 0] = 1
    # print("rowsum shape", rowsum.shape)
    #print("rowsum", rowsum[:24])
    mx = mx / rowsum[:, np.newaxis]
    mx = mx.transpose(0, 1)
    return mx

def load_data_():
    ego = 686
    data_dir = './data/facebook_all_/'
    ego_feat = np.genfromtxt("{}{}.egofeat".format(data_dir, ego), dtype=np.dtype(str))
    feat = np.genfromtxt("{}{}.feat".format(data_dir, ego), dtype=np.dtype(str))
    ego_feat = ego_feat.astype(np.int32)
    feat = feat.astype(np.int32)
    features = np.vstack((ego_feat, feat[:, 1:]))
    #print("all feature num", features.sum(), features.sum() / features.shape[0])
    # print("1 feature shape", features.shape)
    features = fnormalize(features)
    print(features.shape)
    node_map = {}
    rnode_map = {}
    node_map[int(ego)] = 0
    rnode_map[0] = int(ego)
    for i in range(feat.shape[0]):
        node_map[feat[i, 0]] = i+1
        rnode_map[i+1]=feat[i, 0]
    # print(node_map)
    edges = np.genfromtxt("{}{}.edges".format(data_dir, ego), dtype=np.dtype(str))
    edges = edges.astype(np.int32)
    Edges = []
    for i in range(feat.shape[0]):
        Edges.append((node_map[int(ego)], i + 1))
    for e in edges:
        Edges.append((node_map[e[0]], node_map[e[1]]))

    G = nx.Graph()
    G.add_edges_from(Edges)
    # A = nx.to_numpy_matrix(G)  # nx.adjacency_matrix(G).todense()
    # A = np.array(A, dtype=np.float32)
    # Adj = A
    # for i in range(features.shape[0]):
    #     Adj[i, i] = 0
    #     A[i, i] = 1
    # A = Norm.normalized_adjacency(A)
    # feats = features
    # adjs = A
    # node_idx = node_map
    cur_in = np.genfromtxt("{}{}.samples".format(data_dir, ego), dtype=np.dtype(str))
    cur_out = np.genfromtxt("{}{}.labels".format(data_dir, ego), dtype=np.dtype(str))
    cur_out = cur_out.astype(np.int32)
    cur_in = cur_in.astype(np.int32)
    print("label shape", cur_out.shape)
    train = []
    listst = {}
    listsv = {}
    listste = {}

    for i in range(len(cur_in)):
        if i % 7 == 0 or i % 7 == 5 or i % 7 == 6:
            query = np.nonzero(cur_in[i])[0].tolist()
            if len(query)==0:
                print(cur_in[i])
            comm = np.nonzero(cur_out[i])[0].tolist()
            train.append((query, comm))
            # print(query, comm)
            # print(rnode_map[query[0]], [rnode_map[x] for x in comm])
            # exit(0)
            comm = [str(x) for x in comm]
            comm = " ".join(comm)
            if comm not in listst:
                listst[comm]=0
            listst[comm] = listst[comm]+1
            # print(query, comm)
    val = []
    for i in range(len(cur_in)):
        if i % 7 == 1 or i % 7 == 4:
            query = np.nonzero(cur_in[i])[0].tolist()
            if len(query)==0:
                print(cur_in[i])
            comm = np.nonzero(cur_out[i])[0].tolist()
            val.append((query, comm))
            comm = [str(x) for x in comm]
            comm = " ".join(comm)
            if comm not in listsv:
                listsv[comm]=0
            listsv[comm]  = listsv[comm]+1
            #print(query, comm)
    test = []
    for i in range(len(cur_in)):
        if i % 7 == 2 or i % 7 == 3:
            query = np.nonzero(cur_in[i])[0].tolist()
            if len(query)==0:
                print(cur_in[i])
            comm = np.nonzero(cur_out[i])[0].tolist()
            test.append((query, comm))
            comm = [str(x) for x in comm]
            comm = " ".join(comm)
            if comm not in listste:
                listste[comm]=0
            listste[comm]  = listste[comm]+1
            #print(query, comm)
    # for x in lists:
    #     print(x)
    # print(len(lists))
    # print(node_map)
    print(listst)
    print(listste)
    for comm in listst:
        if comm in listste:
            print(comm, "-----------", listst[comm], listste[comm], len(listst), len(listste))
    print(len(listst), len(listste))
    exit(0)
    #'''
    '''
    for i in range(50):
        query = np.nonzero(cur_in[i])[0].tolist()
        if len(query) == 0:
            print(cur_in[i])
        comm = np.nonzero(cur_out[i])[0].tolist()
        train.append((query, comm))
        print(query, comm)
    val = []
    for i in range(50, 100):
        query = np.nonzero(cur_in[i])[0].tolist()
        if len(query) == 0:
            print(cur_in[i])
        comm = np.nonzero(cur_out[i])[0].tolist()
        val.append((query, comm))
    test1 = []
    for i in range(100, 150):
        query = np.nonzero(cur_in[i])[0].tolist()
        if len(query) == 0:
            print(cur_in[i])
        comm = np.nonzero(cur_out[i])[0].tolist()
        test1.append((query, comm))
    #'''
    nodes_adj = {}
    for id1, id2 in G.edges:
        if id1 not in nodes_adj:
            nodes_adj[id1] = [id2]
        else:
            nodes_adj[id1].append(id2)
        if id2 not in nodes_adj:
            nodes_adj[id2] = [id1]
        else:
            nodes_adj[id2].append(id1)

    return features, train, val, test, features.shape[1], features.shape[0], G, nodes_adj


def getConnected(nodeslists, nodes_adj, n_nodes):
    visited = [0]*n_nodes
    for u in nodeslists:
        visited[u] = 1

    cons = []
    for u in nodeslists:
        if visited[u]==0:
            continue
        queue = [u]
        con = [u]
        visited[u]=0
        #index = 0
        while len(queue)!=0:#index<len(queue):
            u = queue[0]
            if u in nodes_adj:
                for v in nodes_adj:
                    if visited[v]==1:
                        visited[v]=0
                        queue.append(v)
                        con.append(v)
            #index = index + 1
            del queue[0]
        cons.append(con)
    return cons

def load_data_():
    ego = 3437
    data_dir = './data/facebook_all_/'
    ego_feat = np.genfromtxt("{}{}.egofeat".format(data_dir, ego), dtype=np.dtype(str))
    feat = np.genfromtxt("{}{}.feat".format(data_dir, ego), dtype=np.dtype(str))
    ego_feat = ego_feat.astype(np.int32)
    feat = feat.astype(np.int32)
    features = np.vstack((ego_feat, feat[:, 1:]))
    #print("all feature num", features.sum(), features.sum() / features.shape[0])
    # print("1 feature shape", features.shape)
    features = fnormalize(features)
    print(features.shape)
    node_map = {}
    rnode_map = {}
    node_map[int(ego)] = 0
    rnode_map[0] = int(ego)
    for i in range(feat.shape[0]):
        node_map[feat[i, 0]] = i+1
        rnode_map[i+1]=feat[i, 0]
    # print(node_map)
    edges = np.genfromtxt("{}{}.edges".format(data_dir, ego), dtype=np.dtype(str))
    edges = edges.astype(np.int32)
    Edges = []
    for i in range(feat.shape[0]):
        Edges.append((node_map[int(ego)], i + 1))
    for e in edges:
        Edges.append((node_map[e[0]], node_map[e[1]]))

    G = nx.Graph()
    G.add_edges_from(Edges)
    # A = nx.to_numpy_matrix(G)  # nx.adjacency_matrix(G).todense()
    # A = np.array(A, dtype=np.float32)
    # Adj = A
    # for i in range(features.shape[0]):
    #     Adj[i, i] = 0
    #     A[i, i] = 1
    # A = Norm.normalized_adjacency(A)
    # feats = features
    # adjs = A
    # node_idx = node_map

    nodes_adj = {}
    for id1, id2 in G.edges:
        if id1 not in nodes_adj:
            nodes_adj[id1] = [id2]
        else:
            nodes_adj[id1].append(id2)
        if id2 not in nodes_adj:
            nodes_adj[id2] = [id1]
        else:
            nodes_adj[id2].append(id1)

    cur_in = np.genfromtxt("{}{}.samples".format(data_dir, ego), dtype=np.dtype(str))
    cur_out = np.genfromtxt("{}{}.labels".format(data_dir, ego), dtype=np.dtype(str))
    cur_out = cur_out.astype(np.int32)
    cur_in = cur_in.astype(np.int32)
    print("label shape", cur_out.shape)
    train = []
    lists = set()
    for i in range(len(cur_in)):
        query = np.nonzero(cur_in[i])[0].tolist()
        if len(query) == 0:
            print(cur_in[i])
        comm = np.nonzero(cur_out[i])[0].tolist()
        comm = [str(x) for x in comm]
        print(query, comm)
        exit(0)
        lists.add(" ".join(comm))

    lists = list(lists)
    print(len(lists))
    random.shuffle(lists)

    lists_t = lists[:5]
    lists_v= lists[5:10]
    lists_te = lists[10: len(lists)]

    train = []
    for i in range(len(cur_in)):
        query = np.nonzero(cur_in[i])[0].tolist()
        if len(query) == 0:
            print(cur_in[i])
        comm = np.nonzero(cur_out[i])[0].tolist()
        comm_ = [str(x) for x in comm]
        comm_ = " ".join(comm_)
        if comm_ in lists_t:
            #train.append((query, comm))
            train.append(([query[0]], comm))
            print(query, comm)
            cons = getConnected(query, nodes_adj, len(node_map))
            print("---------------------", len(cons))

    val = []
    for i in range(len(cur_in)):
        query = np.nonzero(cur_in[i])[0].tolist()
        if len(query) == 0:
            print(cur_in[i])
        comm = np.nonzero(cur_out[i])[0].tolist()
        comm_ = [str(x) for x in comm]
        comm_ = " ".join(comm_)
        if comm_ in lists_v:
            #val.append((query, comm))
            val.append(([query[0]], comm))
            print(query, comm)
            cons = getConnected(query, nodes_adj, len(node_map))
            print("---------------------", len(cons))

            #
    test = []
    for i in range(len(cur_in)):
        query = np.nonzero(cur_in[i])[0].tolist()
        if len(query) == 0:
            print(cur_in[i])
        comm = np.nonzero(cur_out[i])[0].tolist()
        comm_ = [str(x) for x in comm]
        comm_ = " ".join(comm_)
        if comm_ not in lists_v and comm_ not in lists_t:
            #test1.append((query, comm))
            test.append(([query[0]], comm))
            print(query, comm_)
            cons = getConnected(query, nodes_adj, len(node_map))
            print("---------------------", len(cons))
    #'''

    '''
    train = []
    while len(train)<100:
        cid = random.randint(0, len(lists_t)-1)
        comm = lists_t[cid].split(" ")
        comm = [int(x) for x in comm]
        if len(comm)<=1:
            continue
        qsize = 1#random.randint(1, 3)
        random.shuffle(comm)
        qlists = comm[:qsize]
        train.append([qlists, comm])
        print(qlists, comm)


    val = []
    while len(val)<100:
        cid = random.randint(0, len(lists_v)-1)
        comm = lists_v[cid].split(" ")
        comm = [int(x) for x in comm]
        if len(comm)<=1:
            continue
        qsize = 1#random.randint(1, 3)
        random.shuffle(comm)
        qlists = comm[:qsize]
        val.append([qlists, comm])

    test1 = []
    while len(test1)<100:
        cid = random.randint(0, len(lists_te)-1)
        comm = lists_te[cid].split(" ")
        comm = [int(x) for x in comm]
        if len(comm)<=1:
            continue
        qsize = 1#random.randint(1, 3)
        random.shuffle(comm)
        qlists = comm[:qsize]
        test1.append([qlists, comm])
        # print(qlists, comm)
    #'''

    return features, train, val, test, features.shape[1], features.shape[0], G, nodes_adj


def loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path):
    path_train = root + dataset + '/' + dataset + train_path
    if not os.path.isfile(path_train):
        raise Exception("No such file: %s" % path_train)
    train_lists = []
    for line in open(path_train, encoding='utf-8'):
        qlists, attrlists, comm  = line.split(",")
        qlists = qlists.split(" ")
        qlists = [int(x) for x in qlists]
        # pos = pos.split(" ")
        # pos_ = [int(x) for x in pos if int(x) != q]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(train_lists) >= train_n:# + val_n + test_n:
            break
        # random.shuffle(comm_)
        # comm_ = comm_[:3]
        train_lists.append((qlists, comm_))

    path_test = root + dataset + '/' + dataset + test_path
    #print(path_test)
    if not os.path.isfile(path_test):
        raise Exception("No such file: %s" % path_test)
    test_lists = []
    for line in open(path_test, encoding='utf-8'):
        qlists, attrlists, comm = line.split(",")
        qlists = qlists.split(" ")
        qlists = [int(x) for x in qlists]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(test_lists)>=test_n:
            break
        test_lists.append((qlists, comm_))

    path_val = root + dataset + '/' + dataset + val_path
    if not os.path.isfile(path_val):
        raise Exception("No such file: %s" % path_val)
    val_lists = []
    for line in open(path_val, encoding='utf-8'):
        qlists, attrlists, comm = line.split(",")
        qlists = qlists.split(" ")
        qlists = [int(x) for x in qlists]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(val_lists)>=val_n:
            break
        val_lists.append((qlists, comm_))
    #'''

    return train_lists, val_lists, test_lists

def load_data(dataset, root, train_n, val_n, test_n, feats_path, train_path,
              test_path, val_path):
    path = root + dataset + '/' + dataset + '.txt'
    max = 0
    edges = []

    for line in open(path, encoding='utf-8'):
        node1, node2 = line.split(" ")
        node1_ = int(node1)
        node2_ = int(node2)
        if node1_==node2_:
            continue
        if max < node1_:
            max = node1_
        if max < node2_:
            max = node2_
        edges.append([node1_, node2_])
    n_nodes = max + 1
    nodeslists = [x for x in range(n_nodes)]
    graphx = nx.Graph()
    graphx.add_nodes_from(nodeslists)
    graphx.add_edges_from(edges)
    print(graphx)
    del edges

    nodes_adj = {}
    for id1, id2 in graphx.edges:
        if id1 not in nodes_adj:
            nodes_adj[id1] = [id2]
        else:
            nodes_adj[id1].append(id2)
        if id2 not in nodes_adj:
            nodes_adj[id2] = [id1]
        else:
            nodes_adj[id2].append(id1)


    train, val, test = loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path)

    print("======================featd==================================")
    path_feat = root + dataset + '/' + feats_path
    if not os.path.isfile(path_feat):
        raise Exception("No such file: %s" % path_feat)
    feats_node = {}
    count = 1
    for line in open(path_feat, encoding='utf-8'):
        if count == 1:
            node_n_, node_in_dim = line.split()
            if n_nodes < int(node_n_):
                n_nodes = int(node_n_)
            node_in_dim = int(node_in_dim)
            count = count + 1
        else:
            emb = [int(x) for x in line.split()]
            id = emb[0]
            emb = emb[1:]
            feats = np.zeros(node_in_dim)
            feats[emb]=1.0
            feats_node[id] = feats
    features = []
    for i in range(0, n_nodes):
        if i not in feats_node:
            features.append([0.0] * node_in_dim)
        else:
            features.append(feats_node[i])

    nodes_feats = np.array(features)

    return nodes_feats, train, val, test, node_in_dim, n_nodes, graphx, nodes_adj

def main(args):
    n_hid1 = args.n_hid1
    dropout = args.dropout

    model_dir = args.model_dir
    batch_size = args.batch_size
    lr = args.learning_rate

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    lossCE = True
    method = "CS_GCN"
    now = datetime.datetime.now()
    ##################################       FaceBook

    # trainloader = DataLoader(GraphDataset_(phase='train', ego=ego, method=method),
    #                          batch_size=batch_size, shuffle=True, sampler=None)
    # # #
    # evalloader = DataLoader(GraphDataset_(phase='eval', ego=ego, method=method),
    #                         batch_size=1, shuffle=False, sampler=None)
    # # #
    # testloader = DataLoader(GraphDataset_(phase='test1', ego=ego, method=method),
    #                         batch_size=1, shuffle=False, sampler=None)

    # nodes_feats, train, val, test1, node_in_dim, n_nodes, g, nodes_adj \
    #     = load_data()

    nodes_feats, train, val, test, node_in_dim, n_nodes, g, nodes_adj \
        = load_data(args.dataset, args.root,
              args.train_size, args.eval_size, args.test_size,args.feats_path,
              args.train_path, args.test_path, args.eval_path)#'''

    print(len(train), len(val), len(test))

    trainloader = DataLoader(GraphDataset(phase='train',feat=nodes_feats,method=method, args=args, lists=train, G=g,
                                          number=len(train)),
                             batch_size=batch_size, shuffle=True, sampler=None)


    evalloader = DataLoader(GraphDataset(phase='eval',feat=nodes_feats,method=method, args=args, lists=val, G=g,
                                         number=len(val)),
                            batch_size=1, shuffle=False, sampler=None)

    testloader = DataLoader(GraphDataset(phase='test',feat=nodes_feats,method=method, args=args, lists = test, G=g,
                                         number=len(test)),
                            batch_size=1, shuffle=False, sampler=None)
    print(len(trainloader), len(evalloader), len(testloader))

    # assert 1<0, (len(trainloader), len(evalloader), len(testloader))

    for i_iter, batch in enumerate(trainloader):
        input, attr, adj, feat, label, Adj = batch
        max_dim = feat.shape[2]
        break
    #model = QD_GCN(nfeat=max_dim, nhid=n_hid1, nclass=args.n_classes, dropout=dropout)
    model = CS_GCN(nfeat=max_dim, nhid=n_hid1, nclass=args.n_classes, dropout=dropout)
    #model = CS_GCN_NoF(nfeat=max_dim, nhid=n_hid1, nclass=args.n_classes, dropout=dropout)
    #model = SimpleCS_GCN(nfeat=max_dim, nhid=n_hid1, nclass=args.n_classes, dropout=dropout)
    print("CS_GCN")
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    #device = torch.device('cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=args.weight_decay)
    t = time.time()

    best_val_score = [0, 0, 0, 0, 0, 0]
    test_score = [0, 0, 0, 0, 0, 0]

    for epoch in range(args.epoch):
        save = []
        if (epoch % 10 == 9):
            print("\nvalidation")
            save.append(epoch)
            eval_and_test(model, model_dir, evalloader, testloader, best_val_score, test_score, device
                  , nodes_adj, n_nodes)
        model.train()
        adjust_lr_poly(optimizer, args.learning_rate, epoch, args.epoch)
        lossb = 0.0
        for i_iter, batch in enumerate(trainloader):
            input, attr, adj, feat, label, Adj = batch
            input = input.float()  # .unsqueeze(-1)
            attr = attr.float()  # .unsqueeze(-1)
            feat = feat.float()
            adj = adj.float()
            label = label.float()  # .unsqueeze(-1)
            optimizer.zero_grad()
            input = input.to(device)
            attr = attr.to(device)
            adj = adj.to(device)
            feat = feat.to(device)
            label = label.to(device)
            output, xsave = model.forward(input, attr, adj, feat, feat, training=True)

            if lossCE:
                pos_lab = torch.zeros(label.shape).float()
                pos_lab = pos_lab.to(device)
                pos_lab[label > 0.6] = 1
                pos_lab[label < 0.6] = 1
                criterion = torch.nn.BCELoss(weight=pos_lab)
                loss = criterion(output, label)
            else:
                loss = iou_loss(output, label)

            loss.backward()
            optimizer.step()
            lossb = lossb + loss
        print(epoch, "loss", lossb)
    train_model_running_time = (datetime.datetime.now() - now).seconds
    now = datetime.datetime.now()
    eval_and_test(model, model_dir, evalloader, testloader, best_val_score, test_score, device
                  , nodes_adj, n_nodes)
    test_running_time = (datetime.datetime.now() - now).seconds
    print(model_dir)
    print('Final Results')
    print('best val score (precision, recall, f1score) :', best_val_score[0], best_val_score[1], best_val_score[2])
    print('test1 score (precision, recall, f1score) :', test_score[0], test_score[1], test_score[2])

    output = args.root + '/result/' + args.dataset + args.result
    with open(output, 'a+') as fh:
        line = str(args)+" train_model_running_time "+\
               str(train_model_running_time)+" test_running_time "+str(test_running_time)+" F1 "+str(test_score[2])\
               +" Pre "+str(test_score[0])+" Rec "+str(test_score[1])+" nmi_score "+str(test_score[3])+" ari_score "+str(test_score[4])\
               +" jac_score " + str(test_score[5])
        fh.write(line + "\n")
        fh.close()

    return test_score[2], test_score[0], test_score[1], test_score[3], test_score[4], test_score[5], \
           train_model_running_time, test_running_time

def adjust_lr_poly(optimizer, base_lr, epoch, tot_epoch, power=0.9):
    lr = base_lr * (1 - epoch * 1. / tot_epoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lr_exp(optimizer, base_lr, epoch, tot_epoch, power=0.1):
    lr = base_lr * (power ** (epoch * 1. / tot_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _eval(model, evalloader, device):
    ### eval ###
    model.eval()
    all_output = []
    all_label = []
    all_output_bin = []
    all_label_bin = []
    all_output_bin_04 = []
    all_label_bin_04 = []
    iou_thr = 0.35
    for i_iter, batch in enumerate(evalloader):
        if i_iter > 99:
            break
        input, attr, adj, feat, label, Adj = batch
        input = input.float()  # .unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()  # .unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        input = input.to(device)
        attr = attr.to(device)
        adj = adj.to(device)
        feat = feat.to(device)
        label = label
        output, xsave = model.forward(input, attr, adj, feat, feat, training=False)
        output = output.cpu()
        all_output = all_output + (output).view(-1).detach().numpy().tolist()
        all_label = all_label + (label).view(-1).detach().numpy().tolist()
        all_output_bin = all_output_bin + (output > 0.5).view(-1).detach().numpy().tolist()
        all_label_bin = all_label_bin + (label > 0.5).view(-1).detach().numpy().tolist()
        all_output_bin_04 = all_output_bin_04 + (output > iou_thr).view(-1).detach().numpy().tolist()
        all_label_bin_04 = all_label_bin_04 + (label > iou_thr).view(-1).detach().numpy().tolist()
        # print("output  label", output[:5], label[:5])
        # assert 1<0

    # print(all_output[:10])
    # print(all_label[:10])
    print('num  all, output label ', len(all_output), sum(all_output), sum(all_label))
    print('num  all, output_bin label_bin ', len(all_output), sum(all_output_bin), sum(all_label_bin))
    print('num  all, output_bin_0.4 label_bin_0.4 ', len(all_output), sum(all_output_bin_04), sum(all_label_bin_04))
    best_thr = 0
    best_f1 = 0
    for iou_thr_bin in range(5, 96, 5):
        iou_thr = iou_thr_bin / 100.

        precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label, iou_thr)
        print('iou_thr: %.2f, precision, recall, f1_score : %.3f %.3f %.3f' % (iou_thr,
                                                                               precision, recall, f1_score))
        if f1_score > best_f1:
            best_f1 = f1_score
            best_thr = iou_thr

            #
            # tem_dir = model_dir + ".tmp"
            # print('save weight to', tem_dir)
            # torch.save({
            # 	# 'step': ini_step + args.steps,
            # 	'model_state_dict': model.state_dict(),
            # 	# 'loss': loss.item()
            # }, tem_dir)

    if best_thr == 0.:
        best_thr = 0.5
    precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label, best_thr)

    return precision, recall, f1_score, best_thr

    # print('score on val set:')
    # print('precision:', precision)
    # print('recall:', recall)
    # print('f1_score:', f1_score)
    # if better:
    # 	# 	save weight
    # 	tem_dir=model_dir+".tmp"
    # 	print('save weight to', tem_dir)
    # 	torch.save({
    # 		# 'step': ini_step + args.steps,
    # 		'model_state_dict': model.state_dict(),
    # 		# 'loss': loss.item()
    # 	}, tem_dir)

def _test(model, testloader, best_thr, device):
    ### eval ###
    model.eval()
    all_output = []
    all_label = []
    all_output_bin = []
    all_label_bin = []
    all_output_bin_04 = []
    all_label_bin_04 = []
    iou_thr = 0.35
    save = []
    savelb = []
    # savelable={}
    count = 0
    totaltime = 0
    avg_out_dense1 = 0
    avg_lab_dense1 = 0
    avg_out_dense2 = 0
    avg_lab_dense2 = 0

    out_size = 0
    lab_size = 0

    all_cos = 0
    all_cos_lab = 0
    var_cos = []
    var_dens = []
    var_size = []
    for i_iter, batch in enumerate(testloader):
        # if i_iter > 99:
        #     break
        input, attr, adj, feat, label, Adj = batch
        input = input.float()  # .unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()  # .unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        test1 = time.time()
        input = input.to(device)
        attr = attr.to(device)
        adj = adj.to(device)
        feat = feat.to(device)
        # label = label.cuda()
        output, xsave = model.forward(input, attr, adj, feat, feat, training=False)
        output = output.cpu()
        test2 = time.time()
        totaltime = totaltime + test2 - test1
        print("test1 time, test1 num", test2 - test1)
        all_output = all_output + (output).view(-1).detach().numpy().tolist()
        all_label = all_label + (label).view(-1).detach().numpy().tolist()

        if False:
            # if True: # other metrics

            all_output_bin = []
            all_label_bin = []

            all_output_bin = (output > best_thr).view(-1)  # .detach().numpy()
            all_label_bin = (label > best_thr).view(-1)  # .detach().numpy()
            print('all nodes, nodes in output, nodes in ground-truth ', len(all_output), sum(all_output_bin),
                  sum(all_label_bin))

            output_edges = Adj[0][all_output_bin == 1][:, all_output_bin == 1].sum().detach().numpy()
            label_edges = Adj[0][all_label_bin == 1][:, all_label_bin == 1].sum().detach().numpy()

            print("output edge num", output_edges)
            print("ground-truth edge num", label_edges)

            all_output_bin = (output > best_thr).view(-1).detach().numpy()
            all_label_bin = (label > best_thr).view(-1).detach().numpy()
            out_size = out_size + sum(all_output_bin)
            var_size.append(sum(all_output_bin))
            lab_size = lab_size + sum(all_label_bin)
            print('all nodes, nodes in output, nodes in ground-truth ', len(all_output), sum(all_output_bin),
                  sum(all_label_bin))

            print('density in output, density in ground-truth ',
                  output_edges / (sum(all_output_bin) * (sum(all_output_bin) - 1)),
                  label_edges / (sum(all_label_bin) * (sum(all_label_bin) - 1)), )
            print('density in output, density in ground-truth ',
                  output_edges / (sum(all_output_bin)),
                  label_edges / (sum(all_label_bin)))
            if (sum(all_output_bin) > 1):
                avg_out_dense1 = avg_out_dense1 + output_edges / (sum(all_output_bin) * (sum(all_output_bin) - 1))
                var_dens.append(output_edges / (sum(all_output_bin) * (sum(all_output_bin) - 1)))
            else:
                var_dens.append(0)
            avg_lab_dense1 = avg_lab_dense1 + label_edges / (sum(all_label_bin) * (sum(all_label_bin) - 1))
            if (sum(all_output_bin) > 0):
                avg_out_dense2 = avg_out_dense2 + output_edges / (sum(all_output_bin))
            avg_lab_dense2 = avg_lab_dense2 + label_edges / (sum(all_label_bin))

            cos = 0
            cos_count = 0

            cos_lab = 0
            cos_lab_count = 0
            # Fcos=F.cosine_similarity()
            # print("feat ",feat.shape)
            # print('bin', all_output_bin.shape)
            assert all_output_bin.shape[0] <= feat.shape[1], (all_output_bin.shape, feat.shape)
            for node in range(all_output_bin.shape[0]):
                if all_output_bin[node] == 1:
                    for node2 in range(all_output_bin.shape[0]):
                        if node != node2 and all_output_bin[node2] == 1:
                            # print("cos", nn.CosineSimilarity(feat[0][node],feat[0][node2]))
                            # print(node, node2)
                            cos = cos + F.cosine_similarity(feat[0][node], feat[0][node2], dim=0)
                            cos_count = cos_count + 1
                if all_label_bin[node] == 1:
                    for node2 in range(all_label_bin.shape[0]):
                        if node != node2 and all_label_bin[node2] == 1:
                            # print("cos", nn.CosineSimilarity(feat[0][node],feat[0][node2]))
                            # print(node, node2)
                            cos_lab = cos_lab + F.cosine_similarity(feat[0][node], feat[0][node2], dim=0)
                            cos_lab_count = cos_lab_count + 1

            # print("cos",cos)
            if cos_count != 0:
                all_cos = all_cos + cos / cos_count
                var_cos.append(cos / cos_count)
                all_cos_lab = all_cos_lab + cos_lab / cos_lab_count
                print('average cos similarity in output,ground-truth', cos / cos_count, cos_lab / cos_lab_count, '\n')
            else:
                var_cos.append(0)

        count = count + 1

    print("varance size, dense, cos", np.var(var_size), np.var(var_dens), np.var(var_cos))
    print('average cos similarity in output,ground-truth', all_cos / count, all_cos_lab / count)
    print('average nodes in output, nodes in ground-truth ', out_size / count, lab_size / count)
    print('average density1 in output, density1 in ground-truth ', avg_out_dense1, avg_out_dense1 / count,
          avg_lab_dense1 / count)
    print('average density2 in output, density2 in ground-truth ', avg_out_dense2 / count, avg_lab_dense2 / count,
          "\n[][][][][]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    for iou_thr_bin in range(5, 96, 5):
        iou_thr = iou_thr_bin / 100.
        precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label, iou_thr)
        print('iou_thr: %.2f, precision, recall, f1_score : %.3f %.3f %.3f' % (iou_thr,
                                                                               precision, recall, f1_score))
    # 	if f1_score > best_val_score[2]:
    # 		best_val_score[0] = precision
    # 		best_val_score[1] = recall
    # 		best_val_score[2] = f1_score
    # 		best_thr=iou_thr
    #
    # 		#
    # 		# tem_dir = model_dir + ".tmp"
    # 		# print('save weight to', tem_dir)
    # 		# torch.save({
    # 		# 	# 'step': ini_step + args.steps,
    # 		# 	'model_state_dict': model.state_dict(),
    # 		# 	# 'loss': loss.item()
    # 		# }, tem_dir)

    print("test1 time, avg time", totaltime, 1000 * totaltime / len(testloader))
    precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label, best_thr)

    return precision, recall, f1_score

def _eval__(model, evalloader, device):
    ### eval ###
    model.eval()
    all_output = []
    all_label = []
    for i_iter, batch in enumerate(evalloader):
        if i_iter > 99:
            break
        input, attr, adj, feat, label, Adj = batch
        input = input.float()  # .unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()  # .unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        input = input.to(device)
        attr = attr.to(device)
        adj = adj.to(device)
        feat = feat.to(device)
        label = label
        output, xsave = model.forward(input, attr, adj, feat, feat, training=False)
        output = output.cpu()
        all_output.append((output).view(-1).detach().numpy().tolist())
        all_label.append((label).view(-1).detach().numpy().tolist())

    best_thr = 0
    best_f1 = 0
    for iou_thr_bin in range(5, 96, 5):
        iou_thr = iou_thr_bin / 100.

        precision, recall, f1_score = calc_f1_score_iouthr_(all_output, all_label, iou_thr)
        print('iou_thr: %.2f, precision, recall, f1_score : %.3f %.3f %.3f' % (iou_thr,
                                                                               precision, recall, f1_score))
        if f1_score > best_f1:
            best_f1 = f1_score
            best_thr = iou_thr

    if best_thr == 0.:
        best_thr = 0.5
    precision, recall, f1_score = calc_f1_score_iouthr_(all_output, all_label, best_thr)

    return precision, recall, f1_score, best_thr

def _test__(model, testloader, best_thr, device):
    ### eval ###
    model.eval()
    all_output = []
    all_label = []
    count = 0
    totaltime = 0
    avg_out_dense1 = 0
    avg_lab_dense1 = 0
    avg_out_dense2 = 0
    avg_lab_dense2 = 0

    out_size = 0
    lab_size = 0

    all_cos = 0
    all_cos_lab = 0
    var_cos = []
    var_dens = []
    var_size = []
    for i_iter, batch in enumerate(testloader):
        # if i_iter > 99:
        #     break
        input, attr, adj, feat, label, Adj = batch
        input = input.float()  # .unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()  # .unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        test1 = time.time()
        input = input.to(device)
        attr = attr.to(device)
        adj = adj.to(device)
        feat = feat.to(device)
        # label = label.cuda()
        output, xsave = model.forward(input, attr, adj, feat, feat, training=False)
        output = output.cpu()
        test2 = time.time()
        totaltime = totaltime + test2 - test1
        print("test1 time, test1 num", test2 - test1)
        all_output.append((output).view(-1).detach().numpy().tolist())
        all_label.append((label).view(-1).detach().numpy().tolist())

        count = count + 1

    print("varance size, dense, cos", np.var(var_size), np.var(var_dens), np.var(var_cos))
    print('average cos similarity in output,ground-truth', all_cos / count, all_cos_lab / count)
    print('average nodes in output, nodes in ground-truth ', out_size / count, lab_size / count)
    print('average density1 in output, density1 in ground-truth ', avg_out_dense1, avg_out_dense1 / count,
          avg_lab_dense1 / count)
    print('average density2 in output, density2 in ground-truth ', avg_out_dense2 / count, avg_lab_dense2 / count,
          "\n[][][][][]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    for iou_thr_bin in range(5, 96, 5):
        iou_thr = iou_thr_bin / 100.
        precision, recall, f1_score = calc_f1_score_iouthr_(all_output, all_label, iou_thr)
        print('iou_thr: %.2f, precision, recall, f1_score : %.3f %.3f %.3f' % (iou_thr,
                                                                               precision, recall, f1_score))

    print("test1 time, avg time", totaltime, 1000 * totaltime / len(testloader))
    precision, recall, f1_score = calc_f1_score_iouthr_(all_output, all_label, best_thr)

    return precision, recall, f1_score

def _eval_(model, evalloader, device):
    ### eval ###
    model.eval()
    all_output = []
    all_label = []
    for i_iter, batch in enumerate(evalloader):
        if i_iter > 99:
            break
        input,attr, adj,feat, label,Adj= batch
        input = input.float()#.unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()#.unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        input=input.to(device)
        attr=attr.to(device)
        adj=adj.to(device)
        feat=feat.to(device)
        label = label
        output, xsave= model.forward(input,attr, adj,feat,feat, training=False)
        output = output.cpu()
        all_output.append((output).view(-1).detach().numpy().tolist())
        all_label.append((label).view(-1).detach().numpy().tolist())

    best_thr=0
    best_f1=0
    for iou_thr_bin in range(5, 96, 5):
        iou_thr = iou_thr_bin/100.

        precision, recall, f1_score = calc_f1_score_iouthr_(all_output, all_label, iou_thr)
        print('iou_thr: %.2f, precision, recall, f1_score : %.3f %.3f %.3f' % (iou_thr,
                precision, recall, f1_score))
        if f1_score > best_f1:
            best_f1 = f1_score
            best_thr=iou_thr

    if best_thr == 0.:
        best_thr=0.5
    precision, recall, f1_score = calc_f1_score_iouthr_(all_output, all_label,best_thr)

    return precision, recall, f1_score,best_thr

def _test_(model, testloader, best_thr, nodes_adj, n_nodes, device):
    ### eval ###
    model.eval()
    all_output = []
    all_label = []
    all_output_ = []
    all_label_ = []
    count = 0
    totaltime = 0
    F1 = 0.0
    Pre = 0.0
    Rec = 0.0
    nmi_score = 0.0
    ari_score = 0.0
    jac_score = 0.0

    for i_iter, batch in enumerate(testloader):
        count = count+1

        input, attr, adj, feat, label, Adj = batch
        input = input.float()  # .unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()  # .unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        test1 = time.time()
        input = input.to(device)
        attr = attr.to(device)
        adj = adj.to(device)
        feat = feat.to(device)
        # label = label.cuda()
        output, xsave = model.forward(input, attr, adj, feat, feat, training=False)
        output = output.cpu()
        input = input.cpu()
        output = output.view(-1).detach().numpy().tolist()
        label = label.view(-1).detach().numpy().tolist()
        qlists = input.view(-1).detach().numpy().tolist()
        qlists = np.nonzero(qlists)[0]
        # q = qlists
        comm = np.nonzero(label)[0]

        comm_find = find_community(output, qlists, n_nodes, nodes_adj, best_thr)
        comm_find = set(comm_find)
        comm_find = list(comm_find)
        comm = set(comm)
        comm = list(comm)

        all_output = all_output + output
        all_label = all_label + label

        f1, pre, rec = f1_score_(comm_find, comm)
        F1 = F1 + f1
        Pre = Pre + pre
        Rec = Rec + rec
        print("count", count)
        print("f1, pre, rec", qlists, f1, pre, rec)
        print("f1, pre, rec", qlists, F1 / count, Pre / count, Rec / count)
        nmi = NMI_score(comm_find, comm, n_nodes)
        nmi_score = nmi_score + nmi
        ari = ARI_score(comm_find, comm, n_nodes)
        ari_score = ari_score + ari
        jac = JAC_score(comm_find, comm, n_nodes)
        jac_score = jac_score + jac

    F1 = F1 / count
    Pre = Pre / count
    Rec = Rec / count
    print("F1, Pre, Rec", F1, Pre, Rec)
    nmi_score = nmi_score / count
    print("NMI: ", nmi_score)
    ari_score = ari_score / count
    print("ARI: ", ari_score)
    jac_score = jac_score / count
    print("JAC: ", jac_score)  # '''


    print("test1 time, avg time", totaltime, 1000 * totaltime / len(testloader))

    precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label, best_thr)
    print("====================================", precision, recall, f1_score)
    # precision, recall, f1_score = calc_f1_score_iouthr(all_output_, all_label_, best_thr)
    # print("====================================", precision, recall, f1_score)

    return Pre, Rec, F1, nmi_score, ari_score, jac_score

def find_community(output_all, qlists, n_nodes, nodes_adj, best_thr):


    commf = []
    for u, score in enumerate(output_all):
        if score>=best_thr:
            commf.append(u)
    return commf#'''

    commf = [q for q in qlists]
    queue = [q for q in qlists]
    visited = np.zeros(n_nodes)
    visited[qlists] = 1
    prob = output_all
    while len(queue)!=0:
        u = queue[0]
        nei = nodes_adj[u]
        for v in nei:
            if u==v:
                continue
            if prob[v] >= best_thr and visited[v] == 0:
                visited[v] = 1
                queue.append(v)
                commf.append(v)
        del queue[0]
    return commf

def eval_and_test(model, model_dir, evalloader, testloader, best_val_score, test_score,
                  device, nodes_adj, n_nodes):
    # model.eval()
    precision, recall, f1_score, best_thr = _eval(model, evalloader, device)
    print("current best thr ", best_thr)
    print('current val score (precision, recall, f1score) :', precision, recall, f1_score)

    ret = False
    if f1_score > best_val_score[2]:
        best_val_score[0] = precision
        best_val_score[1] = recall
        best_val_score[2] = f1_score
        test1 = time.time()
        #precision, recall, f1_score = _test(model, testloader, best_thr, device)
        precision, recall, f1_score,nmi_score, ari_score, jaccard_score = \
            _test_(model, testloader, best_thr, nodes_adj, n_nodes, device)
        test2 = time.time()
        print("test1 time, test1 num", test2 - test1, len(testloader))
        test_score[0] = precision
        test_score[1] = recall
        test_score[2] = f1_score
        test_score[3] = nmi_score
        test_score[4] = ari_score
        test_score[5] = jaccard_score


    print('current best val score (precision, recall, f1score) :', best_val_score[0], best_val_score[1],
          best_val_score[2])
    print('corresponding test1 score (precision, recall, f1score) :', test_score[0], test_score[1], test_score[2])

    return ret


if __name__ == "__main__":
    parser = ArgumentParser("gcn", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    # Model settings
    parser.add_argument("--n_hid1", default=128, type=int,
                        help="first layer of GCN: number of hidden units")  # options [64, 128, 256]
    parser.add_argument("--n_hid2", default=256, type=int,
                        help="second layer of GCN: number of hidden units")  # options [64, 128, 256]
    parser.add_argument("--n_expert", default=256, type=int,
                        help="attention layer: number of experts")  # options [16, 32, 64, 128]
    parser.add_argument("--att_hid", default=256, type=int,
                        help="attention layer: hidden units")  # options [64, 128, 256]
    parser.add_argument("--model_dir", type=str, default="./GCN_model.pt")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--normalization", default="AugNormAdj",
                        help="The normalization on the adj matrix.")

    # Training settings
    parser.add_argument("--batch_size", default=4, type=int)  # options: [32, 64, 128]
    parser.add_argument("--steps", default=10000, type=int)  # options:  (1000, 2000, ... 40000)
    parser.add_argument("--learning_rate", default=0.001, type=float)  # options [1e-3, 1e-4]
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--earlystopping", type=int, default=0,
                        help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")

    # Others
    parser.add_argument("--extra_feats", default=0, type=int,
                        help="whether or not enable extra feats (e.g.,core num, etc.) 0 Disables/1 Enable")
    parser.add_argument("--input_data_folder", default="data/wv", help="Input data folder")
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--k", default=100, type=int, help="the k core to be collesped")  # options [20, 30, 40]
    parser.add_argument("--b", default=100, type=int, help="the result set size")
    parser.add_argument("--n_classes", default=1, type=int, help="the output classes number")
    parser.add_argument("--epoch", default=100, type=int, help="training epoch")

    parser.add_argument("--data_set", type=str, default="cornell")
    parser.add_argument("--dim", default=1703, type=int, help="attribute dim")

    parser.add_argument('--train_size', type=int, default=150)
    parser.add_argument('--eval_size', type=int, default=10)
    parser.add_argument('--test_size', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='cornell')
    parser.add_argument('--root', type=str, default='./data/')
    parser.add_argument('--feats_path', type=str, default='cornell_feats.txt')
    #parser.add_argument('--feats_path', type=str, default='facebook_686_core_emb_.txt')
    parser.add_argument(
        '--train_path', type=str, default='_train.txt')
    parser.add_argument('--test_path', type=str, default='_test.txt')
    parser.add_argument('--eval_path', type=str, default='_eval.txt')
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--result', type=str, default='_QDGNN_result.txt')

    # unused parameters
    '''
    parser.add_argument("--dev_data_file", default = "")
    parser.add_argument("--n_eval_data", default = 1000, type = int) # number of eval data to generate/load
    parser.add_argument('--lradjust',action='store_true', default=False, 
        help = 'Enable leraning rate adjust.(ReduceLROnPlateau)')
    parser.add_argument("--debug_samplingpercent", type=float, default=1.0, 
        help="The percent of the preserve edges (debug only)")
    '''
    # args = parser.parse_args()
    # if args.verbose:
    #     print(args)
    # main(args)

    args = parser.parse_args()
    if args.verbose:
        print(args)

    count = 0
    F1lists = []
    Prelists = []
    Reclists = []
    nmi_scorelists = []
    ari_scorelists = []
    jac_scorelists = []
    pre_process_time_A, train_model_running_time_A, test_running_time_A = 0.0, 0.0, 0.0

    for i in range(args.count):
        count = count + 1
        now = datetime.datetime.now()
        F1, Pre, Rec, nmi_score, ari_score, jac_score, \
        train_model_running_time, test_running_time = \
            main(args)
        F1lists.append(F1)
        Prelists.append(Pre)
        Reclists.append(Rec)
        nmi_scorelists.append(nmi_score)
        ari_scorelists.append(ari_score)
        jac_scorelists.append(jac_score)
        train_model_running_time_A = train_model_running_time_A + train_model_running_time
        test_running_time_A = test_running_time_A + test_running_time
        print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        running_time = (datetime.datetime.now() - now).seconds
        print('## Running Time:', running_time)
        print('= ' * 20)

    F1_std = np.std(F1lists)
    F1_mean = np.mean(F1lists)
    Pre_std = np.std(Prelists)
    Pre_mean = np.mean(Prelists)
    Rec_std = np.std(Reclists)
    Rec_mean = np.mean(Reclists)
    nmi_std = np.std(nmi_scorelists)
    nmi_mean = np.mean(nmi_scorelists)
    ari_std = np.std(ari_scorelists)
    ari_mean = np.mean(ari_scorelists)
    jac_std = np.std(jac_scorelists)
    jac_mean = np.mean(jac_scorelists)

    train_model_running_time_A = train_model_running_time_A / float(args.count)
    test_running_time_A = test_running_time_A / float(args.count)
    output = args.root + '/result/' + args.dataset + args.result
    with open(output, 'a+') as fh:
        line = "average " + str(args)  + " train_model_running_time " + \
               str(train_model_running_time_A) + " test_running_time " + str(test_running_time_A) + \
               " F1 mean " + str(F1_mean) + " F1 std " + str(F1_std) + " Pre mean " + str(Pre_mean) + " Pre std " + \
               str(Pre_std) + " Rec mean " + str(Rec_mean) + "Rec std " + str(Rec_std) + " nmi_score mean " + \
               str(nmi_mean) + " nmi std " + str(nmi_std) + " ari_score mean " + str(ari_mean) + " ari std " + \
               str(ari_std) + " jac mean " + str(jac_mean) + " jac std " + str(jac_std)
        fh.write(line + "\n")
        fh.close()
