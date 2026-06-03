import argparse

import os

import random

import numpy as np





def generateQuery(root, dataset, train_comms_r, val_comms_r, train_path, val_path, test_path):


    path = root + dataset + '/' + dataset + '.txt'
    # print(path)
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
    path_truth = root + dataset + '/' + dataset + '_cmty.txt'
    if not os.path.isfile(path_truth):
        raise Exception("No such file: %s" % path_truth)
    with open(path_truth) as fh:
        lines = fh.read().strip().split('\n')
        for line in lines:
            #print(line)
            comm = line.replace('\n', '').replace('\r', '').split(" ")
            #print(comm)
            comm = comm[1:]
            comm = list(map(int, comm))
            truthComms .append(comm)
            print(len(comm))
        fh.close()
    print(len(truthComms))


    random.shuffle(truthComms)
    train_comms_n = int(len(truthComms)*train_comms_r)
    val_comms_n = int(train_comms_n*val_comms_r)
    train_comms = truthComms[:train_comms_n-val_comms_n]
    val_comms = truthComms[train_comms_n-val_comms_n:train_comms_n]
    test_comms = truthComms[train_comms_n:]
    train_size = 5000
    val_size = 5000
    test_size = 5000
    train_lists = []
    while len(train_lists)<train_size:
        cid = random.randint(0, len(train_comms)-1)
        comm = train_comms[cid]
        #print(comm)
        qid = random.randint(0, len(comm)-1)
        q = comm[qid]
        train_lists.append([q, comm])
    val_lists = []
    while len(val_lists)<val_size:
        cid = random.randint(0, len(val_comms)-1)
        comm = val_comms[cid]
        qid = random.randint(0, len(comm)-1)
        q = comm[qid]
        val_lists.append([q, comm])
    test_lists = []
    while len(test_lists)<test_size:
        cid = random.randint(0, len(test_comms)-1)
        comm = test_comms[cid]
        qid = random.randint(0, len(comm)-1)
        q = comm[qid]
        test_lists.append([q, comm])


    path_train = root + dataset + '/' + dataset + train_path
    count1 = 0
    with open(path_train, 'w') as fh:
        for q, comm in train_lists:
            comm = [str(x) for x in comm]
            line = str(q) + "," + " ".join(comm)
            fh.write(line + "\n")
            count1 = count1+1
        fh.close()


    path_val = root + dataset + '/' + dataset + val_path
    count2 = 0
    with open(path_val, 'w') as fh:
        for q, comm in val_lists:
            comm = [str(x) for x in comm]
            line = str(q) + "," + " ".join(comm)
            fh.write(line + "\n")
            count2 = count2+1
        fh.close()


    path_test = root + dataset + '/' + dataset + test_path
    count3 = 0
    with open(path_test, 'w') as fh:
        for q, comm in test_lists:
            comm = [str(x) for x in comm]
            line = str(q) + "," + " ".join(comm)
            fh.write(line + "\n")
            count3 = count3+1
        fh.close()
    print(count1, count2, count3)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='syn_test')
    parser.add_argument('--root', type=str, default='./data/syn/')
    parser.add_argument('--train_comms_n', type=float, default=0.5)
    parser.add_argument('--val_comms_n', type=float, default=0.5)
    parser.add_argument('--train_path', type=str, default='_train.txt')
    parser.add_argument('--val_path', type=str, default='_val.txt')
    parser.add_argument('--test_path', type=str, default='_test.txt')
    args = parser.parse_args()
    generateQuery(args.root, args.dataset, args.train_comms_n, args.val_comms_n, args.train_path, args.val_path, args.test_path)