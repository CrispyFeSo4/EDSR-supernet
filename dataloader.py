import sys
import threading
import queue
import random
import collections

import torch
import torch.multiprocessing as multiprocessing

from torch._C import _set_worker_signal_handlers, _update_worker_pids, \
    _remove_worker_pids, _error_if_any_worker_fails
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _DataLoaderIter

from torch.utils.data.dataloader import ExceptionWrapper
from torch.utils.data.dataloader import _use_shared_memory
from torch.utils.data.dataloader import _worker_manager_loop
from torch.utils.data.dataloader import numpy_type_map
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import pin_memory_batch
from torch.utils.data.dataloader import _SIGCHLD_handler_set
from torch.utils.data.dataloader import _set_SIGCHLD_handler

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    # 应该是一个dataloader多线程实现方法
    # 参考：https://www.jianshu.com/p/98d3a23a2d62

    global _use_shared_memory
    _use_shared_memory = True # 共享内存
    _set_worker_signal_handlers() # 设置worker的signal？？

    torch.set_num_threads(1) # 设置线程数是1
    torch.manual_seed(seed) # 设置随机数的种子
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        # r:(1, [124]) 当前数据 在数据集中的 位置？
      
        try:
            idx_scale = 0
            '''if len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                # idx_scale是指定范围内的随机数
                dataset.set_scale(idx_scale)

            #print('thisway  11111111111111111')
            samples = collate_fn([dataset[i] for i in batch_indices])
            #print(samples)
            samples.append(idx_scale)
            #print(samples)'''
            # train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability
            if dataset.first_epoch and len(scale) > 1 and dataset.train:
                idx_integer_scale_list = [9, 19, 29]
                rand_idx = random.randrange(0, len(idx_integer_scale_list))
                dataset.set_scale(idx_integer_scale_list[rand_idx])

            # train on all scale factors for remaining epochs
            if not dataset.first_epoch and len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)

        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))
# 每个worker的工作其实就是从index序列中读取当前数据在数据集中的序号，
# 然后将对应的数据从数据集中取出来，扔到collate_fn中形成一个batch，
# 再把batch扔到数据序列中，完成一次工作循环。

# dataloader+iter的本类 参考https://blog.csdn.net/gdymind/article/details/82226509
class _MSDataLoaderIter(_DataLoaderIter):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [
                multiprocessing.Queue() for _ in range(self.num_workers)
            ]
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            base_seed = torch.LongTensor(1).random_()[0]
            self.workers = [
                multiprocessing.Process(
                    target=_ms_loop,
                    args=(
                        self.dataset,
                        self.index_queues[i],
                        self.worker_result_queue,
                        self.collate_fn,
                        self.scale,
                        base_seed + i,
                        self.worker_init_fn,
                        i
                    )
                )
                for i in range(self.num_workers)]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    # do not initialize cuda context if not necessary
                    maybe_device_id = None
                self.worker_manager_thread = threading.Thread(
                    target=_worker_manager_loop,
                    args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
                          maybe_device_id))
                self.worker_manager_thread.daemon = True
                self.worker_manager_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

class MSDataLoader(DataLoader):
    # 构造一个MS-Dataloader类，继承pytorch原有的datloader类
    # init函数是初始化，其中使用super调用了父类也就是dataloader的init函数
    # 并且设置了一个self.scale
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=default_collate, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):

        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale
        #print('seld.scale')
        #print(self.scale)

    def __iter__(self):
        return _MSDataLoaderIter(self)
