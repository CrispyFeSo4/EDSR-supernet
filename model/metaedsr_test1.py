from model import common
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import model.arch_util as arch_util
from model.arch_util import USConv2d

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return EDSR(args, dilated.dilated_conv)
    else:
        return EDSR(args)



class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(4,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        scale2 = args.scale2[0]
        act = nn.ReLU(True)
        self.scale_list = [round(x * 0.1 + 1, 1) for x in range(1, 31)]

        self.args = args
        self.scale_idx = 0
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        '''# define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)] # 头部不变,普通卷积即可

        # define body module #只修改每层的滤波器
        m_body = []
        m_body.append(common.ResidualBlock_noBN_first(n_feats))
        n_res = n_resblock - 2
        for _ in range(n_res):
            m_body.append(common.ResidualBlock_noBN(n_feats))
        m_body.append(common.ResidualBlock_noBN_last(n_feats)) # 塞进16个ResBlock，都是可变宽度卷积
        # 头尾两个resblock特殊处理，中间相同。

        m_body.append(conv(n_feats, n_feats, kernel_size)) # 尾部不变 普通卷积即可'''

        # 头部卷积 缩减到nf*w
        m_head = [USConv2d(3, n_feats, 3, 1, 1, bias=True, us=[False,True])]

        # define body module 
        m_body = []
        for _ in range(n_resblock):
            m_body.append(common.ResidualBlock_noBN_t1(n_feats))
         # 塞进16个ResBlock，都是可变宽度卷积

        m_body.append(USConv2d(n_feats, n_feats, 3, 1, 1, bias=True, us=[True,True])) 
        # body的尾部卷积
        # body部分通道数为nf*width，尾部不恢复nf，否则无法残差叠加

        m_tail = [USConv2d(n_feats, n_feats, 3, 1, 1, bias=True, us=[True,False])]
        # 最终添加一层卷积 残差叠加后恢复到nf维度

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        ## position to weight

        self.P2W = Pos2Weight(inC=args.n_feats)

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        scale2_int = math.ceil(self.scale2)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale2_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat):
        x = self.sub_mean(x)
        
        # 这里其实只训练了对称数据
        max_scale = max(self.scale,self.scale2)
        if max_scale in self.scale_list[:7]:
            width = 0.25
            #width = 1.0
        elif max_scale in self.scale_list[7:15]:
            width = 0.5
            #width = 0.75
        elif max_scale in self.scale_list[15:22]:
            width = 0.75
            #width = 0.5
        else:
            width = 1.0
            #width = 0.25
        
        #print('-------------------now scale:'+str(self.scale)+' x '+str(self.scale2)+'     width = '+str(width))
        #print('     width = '+str(width)+' start x shape:'+str(x.shape))
        self.head[0].set_width_mult(width)
        x = self.head(x)
        res = x

        #print('     width = '+str(width)+' after head res shape:'+str(res.shape))
        for i in range(self.args.n_resblocks):
            self.body[i].set_width(width)
            res = self.body[i](res)
            #print(str(i+1)+' layer res shape:'+str(res.shape))
            
        self.body[-1].set_width_mult(width)
        res = self.body[-1](res)
        res += x
        #print('     width = '+str(width)+' after body res shape:'+str(res.shape))
        self.tail[0].set_width_mult(width)      
        res = self.tail(res)
        #print('     width = '+str(width)+' after tail res shape:'+str(res.shape))
        #print('---------------------------------')

        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        up_x = self.repeat_x(res)     ### the output is (N*r*r,inC,inH,inW)

        # N*r^2 x [inC * kH * kW] x [inH * inW]
        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)
        scale_int2 = math.ceil(self.scale2)

        cols = cols.contiguous().view(cols.size(0)//(scale_int*scale_int2),scale_int*scale_int2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int2,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int*scale_int2, x.size(2)*x.size(3),-1, 3)

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int2,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int2*x.size(3))
        out = self.add_mean(out)
        ###
        return out




    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale[scale_idx]
        self.scale2 = self.args.scale2[scale_idx]
