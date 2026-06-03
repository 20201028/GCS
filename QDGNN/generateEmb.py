import argparse
import os
import torch
import numpy as np

def generatecoreemb(n_nodes, nodes_adj, root, dataset):
    print("core number")
    md = -1
    deg = [0]*(n_nodes)
    for u in range(n_nodes):
        if u not in nodes_adj:
            continue
        deg[u] = len(nodes_adj[u])
        if deg[u]>md:
            md = deg[u]
    bin = [0]*(md+1)
    for i in range(n_nodes):
        bin[deg[i]] = bin[deg[i]] + 1
    start = 0
    for d in range(md+1):
        num = bin[d]
        bin[d] = start
        start = start+num
    pos = [0]*(n_nodes)
    vert = [0]*(n_nodes)
    for v in range(n_nodes):
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        bin[deg[v]] += 1;
    for d in range(md, 0, -1):
        bin[d] = bin[d - 1];
        #print(d)
    bin[0] = 0
    kmax = 0;
    for i in range(n_nodes):
        v = vert[i]
        if v in nodes_adj:
            for u in nodes_adj[v]:
                if deg[u]>deg[v]:
                    du = deg[u]
                    pu = pos[u]
                    pw = bin[du]
                    w = vert[pw]
                    if u!=w:
                        pos[u] = pw
                        vert[pu] = w
                        pos[w] = pu
                        vert[pw] = u
                    bin[du] += 1;
                    deg[u] -= 1;
    for i in range(n_nodes):
        if deg[i]>kmax:
            kmax = deg[i]

    deg = np.array(deg).astype(float)
    emb = deg
    emb = torch.tensor(emb)
    emb = torch.nn.functional.normalize(emb, p=1.0, dim=0, eps=1e-12, out=None)
    emb = emb.numpy()
    path_core_emb = root + dataset + '/' + dataset + '_core_emb_.txt'
    with open(path_core_emb, 'w') as fh:
        line = str(n_nodes-1)+" "+str(1)
        fh.write(line + "\n")
        for id, emb_ in enumerate(emb):
            line = str(id)+" "+str(emb_)
            fh.write(line + "\n")
        fh.close()

    return kmax, deg

def load_graph(dataset, root):
    path = root + dataset + '/' + dataset + '.txt'
    if not os.path.isfile(path):
        raise Exception("No such file: %s" % path)
    nodes_adj = {}
    max = 0
    for line in open(path, encoding='utf-8'):
        node1, node2 = line.split()
        if node1 == node2:
            continue
        id1, id2 = int(node1), int(node2)
        if max<id1:
            max = id1
        if max<id2:
            max = id2
        if id1 not in nodes_adj:
            nodes_adj[id1] = []
            nodes_adj[id1].append(id2)
        else:
            if id2 not in nodes_adj[id1]:
                nodes_adj[id1].append(id2)
        if id2 not in nodes_adj:
            nodes_adj[id2] = []
            nodes_adj[id2].append(id1)
        else:
            if id1 not in nodes_adj[id2]:
                nodes_adj[id2].append(id1)
    n_nodes = max+1
    return n_nodes, nodes_adj

if __name__ == '__main__':
    #dataset = 'facebook_'
    #dataset = 'email'
    #dataset = 'amazon'
    #dataset = 'dblp'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='karate')
    parser.add_argument('--root', type=str, default='./data/')
    args = parser.parse_args()
    n_nodes, nodes_adj = load_graph(args.dataset, args.root)
    generatecoreemb(n_nodes, nodes_adj, args.root, args.dataset)
    #generatetrussemb(n_nodes, nodes_adj)
