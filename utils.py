import torch
import numpy as np
from functools import wraps
from _thread import start_new_thread

""" some deprecated utils """

def get_onehot(idx, size):
    t = torch.zeros(size)
    t[idx] = 1.
    return t

def init_walk2posu(trainer, args):
    idx_list = []
    for i in range(args.walk_length):
        for j in range(i-args.window_size, i):
            if j >= 0:
                idx_list.append(j)
        for j in range(i+1, i+1+args.window_size):
            if j < args.walk_length:
                idx_list.append(j)
    
    if len(idx_list) != int(args.walk_length * args.window_size * 2 - args.window_size * (args.window_size + 1)):
        print("error idx list")
        print(len(idx_list))
        print(args.walk_length * args.window_size * 2 - args.window_size * (args.window_size + 1))
        exit(0)

    # [walk_length, num_item]
    walk2posu = torch.stack([get_onehot(idx, args.walk_length) for idx in idx_list]).to(trainer.device).T

    return walk2posu

def init_walk2posv(trainer, args):
    idx_list = []
    for i in range(args.walk_length):
        for j in range(i-args.window_size, i):
            if j >= 0:
                idx_list.append(i)
        for j in range(i+1, i+1+args.window_size):
            if j < args.walk_length:
                idx_list.append(i)

    walk2posv = torch.stack([get_onehot(idx, args.walk_length) for idx in idx_list]).to(trainer.device).T

    return walk2posv

def walk2input(trainer, args, walk, walk2posu, walk2posv):
    """ input one sequnce, unused """
    walk = walk.float().to(trainer.device)
    pos_u = walk.unsqueeze(0).mm(walk2posu).squeeze().long()
    pos_v = walk.unsqueeze(0).mm(walk2posv).squeeze().long()
    neg_u = walk.long()
    t = time.time()
    neg_v = torch.LongTensor(np.random.sample(trainer.dataset.neg_table, args.negative)).to(trainer.device)
    sample_time = time.time() - t
    
    return pos_u, pos_v, neg_u, neg_v, sample_time

def walks2input(trainer, args, walks, walk2posu, walk2posv):
    """ input sequences """
    # [batch_size, walk_length]
    bs = len(walks)
    walks = torch.stack(walks).to(trainer.device).float()
    # [batch_size, num_pos]
    pos_u = walks.mm(walk2posu).long()
    pos_v = walks.mm(walk2posv).long()
    # [batch_size, walk_length]
    neg_u = walks.long()
    t = time.time()
    neg_v = torch.LongTensor(np.random.choice(trainer.dataset.neg_table, bs * args.negative, replace=True))
    neg_v = neg_v.to(trainer.device).view(bs, args.negative)
    sample_time = time.time() - t
    
    return pos_u, pos_v, neg_u, neg_v, sample_time

def walks2input_chunk(trainer, args, walks, walk2posu, walk2posv):
    """ input sequences """
    # [batch_size, walk_length]
    bs = len(walks)
    walks = torch.stack(walks).to(trainer.device).float()
    # [batch_size, num_pos]
    pos_u = walks.mm(walk2posu).long()
    pos_v = walks.mm(walk2posv).long()
    # [batch_size, walk_length]
    neg_u = walks.long()
    t = time.time()
    neg_v = torch.LongTensor(np.random.choice(trainer.dataset.neg_table, bs * args.negative, replace=True))
    neg_v = neg_v.to(trainer.device).view(bs, args.negative)
    sample_time = time.time() - t
    
    return pos_u, pos_v, neg_u, neg_v, sample_time