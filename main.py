import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed) # 设置随机数种子 default=1 以便于接下去可复现
checkpoint = utility.checkpoint(args)     ###setting the log and the train information
# checkpoint 感觉是用来保存模型和断点续训的作用

if checkpoint.ok:
    loader = data.Data(args)    # 初始化一个Data加载器，用于从命令行解析 看是要test还是train集
    model = model.Model(args, checkpoint) # 初始化model
    loss = loss.Loss(args, checkpoint) if not args.test_only else None # 如果train则初始化loss
    t = Trainer(args, loader, model, loss, checkpoint) # 上述几个塞进trainer 实例化一个t向量
    while not t.terminate():
        t.train()
        # 一个epoch只调用一次train函数
    

    checkpoint.done() # 结束之后把checkpoint关了（这到底是什么东西

'''# 下面到loss前的代码，如果是重新训练，就注释掉。用于断点继续训练。
    state_dict = torch.load('/home/zwy/Meta-SR-Pytorch/experiment/metardn/model/model_128.pt')
    new_state_dict = {}
    for key, value in state_dict.items():
        name = 'model.module.'+key
        new_state_dict[name] = value
        #print('key:'+str(key)+'     name:'+str(name))
    model = model.Model(args, checkpoint) # 初始化model
    model.load_state_dict(new_state_dict)'''