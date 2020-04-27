import torch
import argparse
import dgl
import csv
import torch.multiprocessing as mp
import os
import random
import time
import numpy as np
from functools import wraps
from _thread import start_new_thread

from reading_data import DeepwalkDataset
from model import SkipGramModel

def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.
    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.
    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

class DeepwalkTrainer:
    '''
    train with negative sampling
    '''
    def __init__(self, args):
        self.args = args
        self.dataset = DeepwalkDataset(args)
        self.output_file_name = args.emb_file
        self.emb_size = len(self.dataset.net)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.mix = args.mix
        self.emb_model = None

    def init_emb(self):
        self.emb_model = SkipGramModel(self.emb_size, 
            self.emb_dimension, self.args)

    def save_emb(self):
        self.emb_model.save_embedding(self.dataset, self.output_file_name)

    def share_memory(self):
        self.emb_model.share_memory()

def set_device(trainer, args):
    choices = sum([args.only_gpu, args.only_cpu, args.mix])
    if choices != 1:
        print("Must choose only *one* training mode in [only_cpu, only_gpu, mix]")
        exit(1)
    trainer.init_emb()
    if args.only_gpu:
        print("Only GPU")
        trainer.emb_model.cuda()
    elif args.mix:
        print("Mix CPU & GPU")
        torch.set_num_threads(args.num_threads)
    else:
        print("Only CPU")
        torch.set_num_threads(args.num_threads)

#@thread_wrapped_func
def fast_train_mp(trainer, args):
    """ only cpu or multi-cpu """
    print("num batchs: %d" % int(len(trainer.dataset.walks) / args.batch_size))
    set_device(trainer, args)
    trainer.share_memory()
    random.shuffle(trainer.dataset.walks)

    #mp.set_start_method('spawn')

    start_all = time.time()
    ps = []

    l = len(trainer.dataset.walks)
    nt = args.num_threads
    for i in range(nt):
        walks = trainer.dataset.walks[int(i * l / nt): int((i + 1) * l / nt)]
        p = mp.Process(target=fast_train_sp, args=(trainer, args, walks, i))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()
    
    print("Used time: %.2fs" % (time.time()-start_all))
    trainer.save_emb()

def fast_train_sp(trainer, args, walks, gpu_id):
    """ a subprocess for fast_train_mp """
    num_batches = len(walks) / args.batch_size
    num_batches = int(np.ceil(num_batches))
    num_pos = 2 * args.walk_length * args.window_size - args.window_size * (args.window_size + 1)
    num_pos = int(num_pos)

    trainer.emb_model.set_device(gpu_id)

    start = time.time()
    with torch.no_grad():
        i = 0
        max_i = args.iterations * num_batches
        
        while True:
            lr = args.initial_lr * (max_i - i) / max_i
            if lr < 0.00001:
                lr = 0.00001

            # multi-sequence input
            i_ = int(i % num_batches)
            walks_ = walks[i_ * args.batch_size: \
                    (1+i_) * args.batch_size]
            if len(walks_) == 0:
                break

            if args.fast_neg:
                trainer.emb_model.fast_learn_super(walks_, lr)
            else:
                bs = len(walks_)
                neg_nodes = torch.LongTensor(
                    np.random.choice(trainer.dataset.neg_table, 
                        bs * num_pos * args.negative, 
                        replace=True))
                trainer.emb_model.fast_learn_super(walks_, lr, neg_nodes=neg_nodes)

            i += 1
            if i > 0 and i % args.print_interval == 0:
                print("Batch %d, tt: %.2fs" % (i, time.time()-start))
                start = time.time()
            if i_ == num_batches - 1:
                break

@thread_wrapped_func
def fast_train(trainer, args):
    """ only gpu, or mixed training with only one gpu"""
    # the number of postive node pairs of a node sequence
    num_pos = 2 * args.walk_length * args.window_size - args.window_size * (args.window_size + 1)
    num_pos = int(num_pos)
    num_batches = len(trainer.dataset.net) * args.num_walks / args.batch_size
    num_batches = int(np.ceil(num_batches))
    print("num batchs: %d" % num_batches)

    set_device(trainer, args)

    start_all = time.time()
    start = time.time()
    with torch.no_grad():
        i = 0
        max_i = args.iterations * num_batches
        for iteration in range(trainer.iterations):
            print("\nIteration: " + str(iteration + 1))
            random.shuffle(trainer.dataset.walks)

            while True:
                lr = args.initial_lr * (max_i - i) / max_i
                if lr < 0.00001:
                    lr = 0.00001

                # multi-sequence input
                i_ = int(i % num_batches)
                walks = trainer.dataset.walks[i_*args.batch_size: \
                        (1+i_)*args.batch_size]
                if len(walks) == 0:
                    break

                if args.fast_neg:
                    trainer.emb_model.fast_learn_super(walks, lr)
                else:
                    bs = len(walks)
                    neg_nodes = torch.LongTensor(
                        np.random.choice(trainer.dataset.neg_table, 
                            bs * num_pos * args.negative, 
                            replace=True))
                    trainer.emb_model.fast_learn_super(walks, lr, neg_nodes=neg_nodes)

                i += 1
                if i > 0 and i % args.print_interval == 0:
                    print("Batch %d, traning time: %.2fs" % (i, time.time()-start))
                    start = time.time()
                if i_ == num_batches - 1:
                    break

    print("Used time: %.2fs" % (time.time()-start_all))
    trainer.save_emb()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepWalk")
    parser.add_argument('--net_file', type=str, 
            help="network file")
    parser.add_argument('--emb_file', type=str, default="emb.txt",
            help='embedding file of txt format')
    parser.add_argument('--dim', default=128, type=int, 
            help="embedding dimensions")
    parser.add_argument('--window_size', default=5, type=int, 
            help="context window size")
    parser.add_argument('--num_walks', default=10, type=int, 
            help="number of walks for each node")
    parser.add_argument('--negative', default=5, type=int, 
            help="negative samples")
    parser.add_argument('--iterations', default=1, type=int, 
            help="iterations")
    parser.add_argument('--batch_size', default=10, type=int, 
            help="number of node sequences in each step")
    parser.add_argument('--print_interval', default=1000, type=int, 
            help="print interval")
    parser.add_argument('--walk_length', default=80, type=int, 
            help="walk length")
    parser.add_argument('--initial_lr', default=0.025, type=float, 
            help="learning rate")
    parser.add_argument('--neg_weight', default=1., type=float, 
            help="negative weight")
    parser.add_argument('--lap-norm', default=-1., type=float, 
            help="laplacian normalization weight")
    parser.add_argument('--mix', default=False, action="store_true", 
            help="mixed training with CPU and GPU")
    parser.add_argument('--only_cpu', default=False, action="store_true", 
            help="training with CPU")
    parser.add_argument('--only_gpu', default=False, action="store_true", 
            help="training with GPU")
    parser.add_argument('--fast_neg', default=False, action="store_true", 
            help="do negative sampling inside a batch")
    parser.add_argument('--num_threads', default=2, type=int, 
            help="number of threads if trained with CPU")
    parser.add_argument('--num_gpus', default=2, type=int, 
            help="number of GPUs when mixed training")
    args = parser.parse_args()

    trainer = DeepwalkTrainer(args)
    fast_train_mp(trainer, args)

    #fast_train(trainer, args)
