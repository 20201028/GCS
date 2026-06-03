import argparse

import os

import random

import numpy as np


def getConnected_query(q, comm, nodes_adj, n_nodes):
    visitedc = [0]*n_nodes
    for u in comm:
        visitedc[u] = 1
    visitedq = [0]*n_nodes
    visitedq[q] = 1
    if q not in nodes_adj:
        return [q]
    queue = [q]
    index = 0
    qlists = [q]
    while index<len(queue):
        for u in nodes_adj[q]:
            if visitedc[u]==1 and visitedq[u]==0:
                visitedq[u]=1
                qlists.append(u)
                queue.append(u)

        index = index+1
    return qlists

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
                for v in nodes_adj[u]:
                    if visited[v]==1:
                        visited[v]=0
                        queue.append(v)
                        con.append(v)
            #index = index + 1
            del queue[0]
        cons.append(con)
    return cons

def generatetest(root, dataset, test_path, feats_node, map_comm, map_comm_attrs, map_comm_all, start, end):
    test_size = 200
    test_lists = []
    lists = [x for x in range(start, end)]
    random.shuffle(lists)
    truthCommste = [x for x in lists]
    while len(truthCommste)<test_size:
        random.shuffle(lists)
        truthCommste = truthCommste+[x for x in lists]
    random.shuffle(truthCommste)
    count = 0
    while len(test_lists)<test_size and count<len(truthCommste):
        comm_id = truthCommste[count]
        comm = map_comm[comm_id]
        size = random.randint(1, 3)
        if len(comm)<=2:
            count = count+1
            continue
        random.shuffle(comm)
        qlists = comm[:size]
        attrlists = []
        map_cout = map_comm_attrs[comm_id]
        map_cout_ = {}
        for key, value in map_cout.items():
            value = map_cout[key] - (map_comm_all[key] - map_cout[key])
            map_cout_[key] = value#'''
        map_cout_ = sorted(map_cout_.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        countsize = 0
        for key, value in map_cout_:
            if countsize < 3:
                attrlists.append(key)
                countsize = countsize + 1
        #attrlists = feats_node[qlists[0]]
        attrlists = [str(x) for x in attrlists]
        test_lists.append([qlists, attrlists, comm])
        count = count + 1


    path_test = root + dataset + '/' + dataset + test_path
    count = 0
    with open(path_test, 'w') as fh:
        for qlists, attrlists, comm in test_lists:
            qlists = [str(x) for x in qlists]
            comm = [str(x) for x in comm]
            line = " ".join(qlists)+","+" ".join(attrlists) + ","+" ".join(comm)
            fh.write(line + "\n")
            count = count+1
        fh.close()
    print(count)


def generateQuery(root, dataset, test_path, train_path):
    #seed = 0
    #random.seed(seed)

    path = root + dataset + '/' + dataset + '.txt'
    if not os.path.isfile(path):
        raise Exception("No such file: %s" % path)
    max = 0
    count = 0
    nodes_adj = {}
    for line in open(path, encoding='utf-8'):
        node1, node2 = line.split(" ")
        if node1 == node2:
            continue
        id1, id2 = int(node1), int(node2)
        if max < id1:
            max = id1
        if max < id2:
            max = id2
        if id1 not in nodes_adj:
            nodes_adj[id1] = [id2]
        else:
            nodes_adj[id1].append(id2)
        if id2 not in nodes_adj:
            nodes_adj[id2] = [id1]
        else:
            nodes_adj[id2].append(id1)
        count = count+1
    n_nodes = max+1
    print(n_nodes, count)
    truthComms = []
    truthComms_ = set()
    path_truth = root + dataset + '/' + dataset + '_cmty.txt'
    if not os.path.isfile(path_truth):
        raise Exception("No such file: %s" % path_truth)
    with open(path_truth) as fh:
        lines = fh.read().strip().split('\n')
        for line in lines:
            comm = line.replace('\n', '').replace('\r', '').split(" ")
            comm = comm[1:]
            comm_ = " ".join(comm)
            if comm_ in truthComms_:
                continue
            truthComms_.add(comm_)
            comm = list(map(int, comm))
            if len(comm)<=2:
                continue
            truthComms.append(comm)
        fh.close()
    print(len(truthComms))
    random.shuffle(truthComms)
    print("======================featd==================================")
    path_feat = args.root + args.dataset + '/' + args.dataset + "_feats.txt"
    if not os.path.isfile(path_feat):
        raise Exception("No such file: %s" % path_feat)
    feats_node = {}
    count = 0
    map_cout_all = {}
    for line in open(path_feat, encoding='utf-8'):
        if count==0:
            count = count+1
            continue
        attr = line.split(" ")
        id = int(attr[0])
        attr = attr[1:]
        attr = [int(x) for x in attr if x != '\n']
        feats_node[id] = attr
        for x in attr:
            feats_node[x + n_nodes] = []
            feats_node[x + n_nodes].append(x)
            if x not in map_cout_all:
                map_cout_all[x] = 0
            map_cout_all[x] = map_cout_all[x] + 1
    map_comm_attrs = {}
    map_comm = {}
    map_comm_all = {}
    for id, comm in enumerate(truthComms):
        map_comm[id]=comm
        map_cout = {}
        for u in comm:
            lists_attr = feats_node[u]
            for x in lists_attr:
                if x not in map_cout:
                    map_cout[x] = 0
                map_cout[x] = map_cout[x] + 1
        map_comm_attrs[id] = map_cout
        for key, value in map_cout.items():
            if key not in map_comm_all:
                map_comm_all[key]=value
            if value>map_comm_all[key]:
                map_comm_all[key]=value

    generatetest(root, dataset, args.test_path, feats_node, map_comm, map_comm_attrs, map_comm_all, 0, 39)
    generatetest(root, dataset, args.eval_path, feats_node, map_comm, map_comm_attrs, map_comm_all, 0, 39)
    generatetest(root, dataset, args.train_path, feats_node, map_comm, map_comm_attrs, map_comm_all, 0, 39)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook_1912')
    parser.add_argument('--root', type=str, default='./data/')
    parser.add_argument('--test_path', type=str, default='_test.txt')
    parser.add_argument('--train_path', type=str, default='_train.txt')
    parser.add_argument('--eval_path', type=str, default='_eval.txt')
    args = parser.parse_args()
    generateQuery(args.root, args.dataset,args.test_path, args.train_path)