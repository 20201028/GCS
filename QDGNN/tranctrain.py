import argparse
import os
import random
import numpy as np


def tran(args):
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    print("tran")
    path = args.root + args.dataset + '/' + args.dataset+"_train.txt"
    if not os.path.isfile(path):
        raise Exception("No such file: %s" % path)

    train = []
    for line in open(path, encoding='utf-8'):
        q, comm = line.split(",")
        comm = comm.replace('\n', '').replace('\r', '').split(" ")
        train.append((q, comm))
    pos_size = args.pos_size
    train_lists = []
    for q, comm in train:
        comm_ = [x for x in comm if x != q]
        if len(comm_)<pos_size:
            continue
        random.shuffle(comm_)
        pos = comm_[:pos_size]
        train_lists.append((q, pos, comm))

    pathout = args.root + args.dataset + '/' + args.dataset + "_" +str(pos_size) +"_train_pos.txt"
    with open(pathout, 'w') as fh:
        for q, pos, comm in train_lists:
            line_ = str(q) + "," + " ".join(pos)+","+" ".join(comm)
            fh.write(line_ + "\n")
        fh.close()

    print("tran")
    path = args.root + args.dataset + '/' + args.dataset+"_val.txt"
    if not os.path.isfile(path):
        raise Exception("No such file: %s" % path)

    val = []
    for line in open(path, encoding='utf-8'):
        q, comm = line.split(",")
        comm = comm.replace('\n', '').replace('\r', '').split(" ")
        val.append((q, comm))
    pos_size = args.pos_size
    val_lists = []
    for q, comm in val:
        comm_ = [x for x in comm if x != q]
        if len(comm_)<pos_size:
            continue
        random.shuffle(comm_)
        pos = comm_[:pos_size]
        val_lists.append((q, pos, comm))

    pathout = args.root + args.dataset + '/' + args.dataset + "_" +str(pos_size) +"_val_pos.txt"
    with open(pathout, 'w') as fh:
        for q, pos, comm in val_lists:
            line_ = str(q) + "," + " ".join(pos)+","+" ".join(comm)
            fh.write(line_ + "\n")
        fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='syn_test')
    parser.add_argument('--root', type=str, default='./data/syn/')
    parser.add_argument('--pos_size', type=int, default=3)
    args = parser.parse_args()
    tran(args)