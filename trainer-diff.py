# trainer on different probability to choose scale
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
from decimal import Decimal

import threading
import utility
import pdb
import torch
from torch.autograd import Variable
from tqdm import tqdm
import random

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        # 一个训练器需要一个data loader，一个model，一个loss ckp是checkpoint
        self.args = args
        self.scale = args.scale # scale是一个列表list
        self.scale2 = args.scale2
        self.scale_list = [round(x * 0.1 + 1, 1) for x in range(1, 31)]

        self.ckp = ckp
        self.loader_train = loader.loader_train
        #self.loader_train_lr = loader.loader_train_lr
        #self.loader_train_hr = loader.loader_train_hr
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8


    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network

    # 输入scale 还有inH inW，计算HW*3的MLP的input
    def input_matrix_wpn(self,inH, inW, scale,scale2, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = round(scale*inH), round(scale2*inW)
        #outH, outW = int(scale*inH), int(scale2*inW)
        #outH, outW = int(scale*inH), int(scale2*inW)
        #----------测试方便需要 到时候记得删掉---------------
        if scale in [1.4,2.3,2.8] or scale2 in [1.4,2.3,2.8]:
            outH, outW = int(scale*inH), int(scale2*inW)
        #----------测试方便需要 到时候记得删掉---------------
        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        scale_int2 = int(math.ceil(scale2))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH,  scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int2)
        mask_w = torch.zeros(1, inW, scale_int2)
        if add_scale:
            scale_mat = torch.zeros(1,2)
            scale_mat[0,0] = 1.0/scale
            scale_mat[0,1] = 1.0/scale2
            scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int*scale_int2)),0) 
            
        ####projection  coordinate  and caculate the offset 
        h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale) # i/j
        int_h_project_coord = torch.floor(h_project_coord) # i/j向下取整

        offset_h_coord = h_project_coord - int_h_project_coord # R(i)
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale2)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag,  0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int2 * inW), 2).view(-1, scale_int2 * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int2 * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int2 * inW), 2).view(-1, scale_int2 * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int2 * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int2*inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1,2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1,-1,2), pos_mat),2)

        return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW
        # return的position matrix应该是outH*outW*2的大小

    def select_element_with_probability(self):

    # 定义概率
        probabilities = [0.25, 0.5, 0.75, 1.0]
        probabilities = [p / sum(probabilities) for p in probabilities]

    # 定义元素范围
        ranges = [
            range(0, 7),
            range(7, 15),
            range(15, 22),
            range(22, 30)
        ]

    # 使用概率抽取元素
        selected_range = random.choices(ranges, probabilities)[0]
        selected_element = random.choice(selected_range)

        return selected_element # 返回元素本身 即返回一个scale的idx

    def train(self):
        self.scheduler.step() # 按照某种策略 对学习率进行调整
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        
        timer_data, timer_model = utility.timer(), utility.timer()

        len_of_trainset = len(self.loader_train.dataset)
        batch_range = int( len_of_trainset / self.args.batch_size) + 1
        for batch in range(batch_range):
            filenames = []           
            for i in range(self.args.batch_size):
                filename = random.randint(0, len_of_trainset)
                filename = str(filename).zfill(3)[:3]
                filenames.append(filename) # 抽取bs数量的样本

            idx_scale = self.select_element_with_probability()
            #print('test:scale------------'+str(self.scale[idx_scale])+'x'+str(self.scale2[idx_scale]))

            # Loss_all = 0

            datatime = 0
            modeltime = 0
            #for idx_scale in idxscales:

            lr,hr,filename = self.loader_train.dataset.get_img_by_filename(filenames,idx_scale)
            # print('------------------scale:'+str(idx_scale))
            # print(lr.shape)
            # print(hr.shape)
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            datatime += timer_data.release()
            timer_model.tic()
            N,C,H,W = lr.size()

            _,_,outH,outW = hr.size()

            scale_coord_map, mask = self.input_matrix_wpn(H,W,self.args.scale[idx_scale],self.args.scale2[idx_scale])  ###  get the position matrix, mask

            if self.args.n_GPUs>1 and not self.args.cpu:
                scale_coord_map = torch.cat([scale_coord_map]*self.args.n_GPUs,0)
            else:
                scale_coord_map = scale_coord_map.to(device)
            
                

            sr = self.model(lr, idx_scale, scale_coord_map) 
            re_sr = torch.masked_select(sr,mask.to(device))
            re_sr = re_sr.contiguous().view(N,C,outH,outW)
            loss = self.loss(re_sr, hr) # 计算loss

            if loss.item() < self.args.skip_threshold * self.error_last:
                # Loss_all += loss
                pass
                #print('loss:'+str(loss.item()))
                #print('Loss_all:'+str(Loss_all.item()))
                #print('-------------end')
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()
            modeltime += timer_model.release()

               
                # 这里输出的每一个loss 是累积了100个batch的平均loss
                # 相当于 之前100个batch的所有样本的loss进行平均
                # 原始代码的loss_init文件中只需要除以batch数 但是当前一个batch有3组样本 所以除以batch*3                    
            timer_data.tic()
            
            #print('------------------------------batch:'+str(batch)+'  Lossall:'+str(Loss_all.item()))
            #Loss_all = Loss_all / 3
            loss.backward() # 计算梯度
            self.optimizer.step() # 更新
            self.optimizer.zero_grad() # 归零

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s\tdata=[ {}x{} ]'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        #timer_model.release(),
                        modeltime,
                        #timer_data.release(),
                        datatime,
                        self.args.scale[idx_scale],
                        self.args.scale2[idx_scale],
                        ))

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        if self.args.n_GPUs == 1:
            target = self.model
        else:
            target = self.model  #.module

        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir,'model', 'model_{}.pt'.format(epoch))
        )
        ## save models 保存模型
        # 一个epoch调用一次train函数，train函数中由MSDataloader的loader-train返回lr和hr和对应的
        # 和对应的scale，传入input-matrix计算posiion，传出来之后调用model算出sr图像
        # 与GT做loss，计算梯度，更新。
        # 也就是，一次train是不同scale的。

    def test(self):  
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        timer_test = utility.timer()
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                eval_acc_ssim = 0
                self.loader_test.dataset.set_scale(idx_scale)
                #tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(self.loader_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    N,C,H,W = lr.size()
                    #print('test-----------')
                    #print(lr.size())
                    #print(hr.shape)
                    scale = self.args.scale[idx_scale]
                    scale2 = self.args.scale2[idx_scale]
                    outH,outW = int(H*scale),int(W*scale2)
                    #_,_,outH,outW = hr.size()
                    #timer_test.tic()

                    scale_coord_map, mask = self.input_matrix_wpn(H,W,self.args.scale[idx_scale],self.args.scale2[idx_scale])
                    #position, mask = self.pos_matrix(H,W,self.args.scale[idx_scale])
                    #print(timer_test.toc())
                    if self.args.n_GPUs>1 and not self.args.cpu:
                        scale_coord_map = torch.cat([scale_coord_map]*self.args.n_GPUs,0)
                    else:
                        scale_coord_map = scale_coord_map.to(device)

                    timer_test.tic()
                    sr = self.model(lr, idx_scale,scale_coord_map)
                    #print('sr:-------')
                    #print(sr.shape)
                    timer_test.hold()
                    re_sr = torch.masked_select(sr,mask.to(device))
                    #print('re_sr.shape:--------------')
                    #print(re_sr.shape)
                    sr = re_sr.contiguous().view(N,C,outH,outW)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    #timer_test.hold()
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_acc_ssim += utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        a=1
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
               # print(timer_test.acc/100)
                self.ckp.write_log(
                    '[{} x{}_x{}]\tPSNR: {:.3f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        scale2,
                        self.ckp.log[-1, idx_scale],
                        eval_acc_ssim / len(self.loader_test),
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self): # 条件是if not terminate
        if self.args.test_only:# 如果进入这个分支 说明是test 调用test函数 并且不调用train函数
            self.test()
            return True
        else: # 存在train的需求
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs # 如果false 说明还没结束训练 回去就继续调用train
            # 如果大于了 说明已经够训练轮次
            # 所以train函数会被调用很多次！

