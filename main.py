import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) #所有GPU
torch.cuda.manual_seed(seed)     # 当前GPU
# CUDA有些算法是non deterministic, 需要限制    
#os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
#torch.use_deterministic_algorithms(True)

#torch.manual_seed(args.seed) 
checkpoint = utility.checkpoint(args)     ###setting the log and the train information

if checkpoint.ok:
    loader = data.Data(args)    # 初始化一个Data加载器，用于从命令行解析 看是要test还是train集
    model = model.Model(args, checkpoint) # 初始化model
    loss = loss.Loss(args, checkpoint) if not args.test_only else None # 如果train则初始化loss
    t = Trainer(args, loader, model, loss, checkpoint) # 上述几个塞进trainer 实例化一个t向量
    while not t.terminate():
        t.train()
        # 一个epoch只调用一次train函数
    

    checkpoint.done()

'''# 下面到loss前的代码，用于断点继续训练。
    state_dict = torch.load('/home/zwy/Meta-SR-Pytorch/experiment/metardn/model/model_128.pt')
    new_state_dict = {}
    for key, value in state_dict.items():
        name = 'model.module.'+key
        new_state_dict[name] = value
        #print('key:'+str(key)+'     name:'+str(name))
    model = model.Model(args, checkpoint) # 初始化model
    model.load_state_dict(new_state_dict)'''