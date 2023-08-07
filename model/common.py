import math

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import model.arch_util as arch_util
from model.arch_util import USConv2d

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualBlock_noBN(nn.Module): # resblock from arm-net(cbh)
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,True])

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def set_width(self, width):
        self.conv1.set_width_mult(width)
        self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_v2(nn.Module): # resblock for v2
        # 64->32->64

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_v2, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,False])

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)
    
    def set_width(self, width):
        self.conv1.width_mult = width
        self.conv2.width_mult = width

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_t4(nn.Module): # resblock for test4
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResidualBlock_noBN_t4, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.width = None
        self.nf = n_feats

    def set_width(self, width):
        self.width = width

    def forward(self, x):
        # width=1.0 不修改 为正常resblock
        if self.width == 1.0:
            res = self.body[0](x)
        else:
            # width=0.25/0.5/0.75，模块第一层卷积的weight仅保留out_channel的前width
            weight = torch.clone(self.body[0].weight)
            mask_w = torch.zeros(weight.size())
            mask_w[:int(self.width*self.nf),:,:,:] = 1 # 仅保留前面的width*nf（nf为总channel数）
            weight = weight.to(torch.device("cuda"))  
            mask_w = mask_w.to(torch.device("cuda"))  
            weight = torch.mul(weight,mask_w)
            # weight[int((1-self.width)*self.nf):, :, :, :] = 0

            bias = torch.clone(self.body[0].bias) # bias做相同处理，仅保留前面width
            mask_b = torch.zeros(bias.size())
            mask_b[:int(self.width*self.nf)] = 1
            bias = bias.to(torch.device("cuda"))  
            mask_b = mask_b.to(torch.device("cuda")) 
            bias = torch.mul(bias,mask_b)
            # bias[int((1-self.width)*self.nf):] = 0

            res = F.conv2d(x,weight,bias,stride = 1, padding = 1)
        
        tmp = F.relu(res)
        res = self.body[1](tmp) # 下面是正常的resblock，通过relu和第二层卷积，叠加跳跃连接，return
        tmp = res + x
        return tmp

class ResidualBlock_noBN_11conv3(nn.Module): # resblock for 1*1 Conv*3, not share
    def __init__(self, nf=64):
        super(ResidualBlock_noBN_11conv3, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = nn.Conv2d(nf, nf, 3, stride = 1, padding = 1)
        self.conv25 = nn.Conv2d(nf//4, nf*3//4, 1, stride = 1, padding = 0)
        self.conv50 = nn.Conv2d(nf//2, nf//2, 1, stride = 1, padding = 0)
        self.conv75 = nn.Conv2d(nf*3//4, nf//4, 1, stride = 1, padding = 0)
        self.n_feats = nf

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv25, self.conv50, self.conv75 ], 0.1) # ver2

    def set_width(self, width):
        self.conv1.set_width_mult(width)

    def forward(self, x):
        identity = x
        out = self.conv1(x)

        ch = out.shape[1] # 根据输出的channel数量判断width，传入对应卷积分支
        if ch < self.n_feats:
            if ch/self.n_feats == 0.25:
                tmp = self.conv25(out)
            elif ch/self.n_feats == 0.5:
                tmp = self.conv50(out)
            elif ch/self.n_feats == 0.75:
                tmp = self.conv75(out)
            out = torch.cat((out, tmp), dim=1)

        out = F.relu(out)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_11USConv(nn.Module): # resblock for 1*1 USconv2d share

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_11USConv, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = nn.Conv2d(nf, nf, 3, stride = 1, padding = 1)
        self.conv3 = USConv2d(nf, nf, 1, 1, 0, bias=True, us=[True,True])
        self.n_feats = nf

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def set_width(self, width):
        self.conv1.set_width_mult(width)
        self.conv3.set_width_mult(1-width)

    def forward(self, x):
        identity = x
        #out = F.relu(self.conv1(x), inplace=True)
        # 为啥原始代码inplace=true？似乎没必要原地操作
        out = self.conv1(x)
        ch = out.shape[1]
        if ch < self.n_feats:
            tmp = self.conv3(out)
            out = torch.cat((out, tmp), dim=1)
        out = F.relu(out)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_0copy(nn.Module): # resblock for 0copy
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN_0copy, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,True])
        self.conv2 = nn.Conv2d(nf, nf, 3, stride = 1, padding = 1)
        self.n_feats = nf
        # conv2直接nf-nf，不需要改变宽度，因此使用普通卷积

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def set_width(self, width):
        self.conv1.set_width_mult(width)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)

        num_channels = out.size(1)
        if num_channels < self.n_feats:
            out = F.pad(out, (0, 0, 0, 0, 0, self.n_feats - num_channels))

        out = self.conv2(out)
        return identity + out
       
# ------------以上部分是目前常用block的类定义，检查至此即可------------------------------
# ------------------------------------------------------------------------------------

class ResidualBlock_noBN_v2new(nn.Module):


    def __init__(self, nf=64):
        super(ResidualBlock_noBN_v2new, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,False])

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def set_width(self, width):
        self.conv1.set_width_mult(width)
        self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out   

class ResidualBlock_noBN_v2new_2(nn.Module):

    def __init__(self, nf=64, res_scale=1):
        super(ResidualBlock_noBN_v2new_2, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,False])
        self.res_scale = res_scale

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def set_width(self, width):
        self.conv1.set_width_mult(width)
        self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        #print('this is in resblock: before conv:'+str(x.shape))
        out = F.relu(self.conv1(x), inplace=True)
        #print('this is in resblock: after conv1:'+str(out.shape))
        out = self.conv2(out).mul(self.res_scale)
        #print('this is in resblock: after conv2:'+str(out.shape))
        return identity + out 



class ResidualBlock_noBN_v4(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_v4, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,True])
        self.n_feats = nf
        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)
    
    def set_width(self, width):
        self.conv1.width_mult = width
        #self.conv2.width = width
        self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        #print('this is in resblock:  shape1 = '+str(out.shape))
        out = self.conv2(out)
        #print('this is in resblock:  shape2 = '+str(out.shape))
        # copy
        while out.shape[1]<self.n_feats:
            out = torch.cat((out, out), dim=1) 
        #print('this is in resblock:  shape3 = '+str(out.shape))
        out = out[:,:self.n_feats,:,:]
        #print('this is in resblock:  shape4 = '+str(out.shape))
        return identity + out


class ResidualBlock_noBN_v4_2(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_v4_2, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,True])
        self.n_feats = nf
        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)
    
    def set_width(self, width):
        self.conv1.width_mult = width
        #self.conv2.width = width
        self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        #print('this is in resblock:  shape1 = '+str(out.shape))
        out = self.conv2(out)
        #print('this is in resblock:  shape2 = '+str(out.shape))
        
        return out


class ResidualBlock_noBN_t1(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_t1, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,True])
        self.nf = nf

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def set_width(self, width):
        self.conv1.set_width_mult(width)
        self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        #print('this is input x shape:'+str(x.shape))
        # copy to nf
        while x.shape[1]<self.nf:
            x = torch.cat((x, identity), dim=1)
        x = x[:,:self.nf,:,:]
        #print('this is copy1 x shape:'+str(x.shape))
        out = F.relu(self.conv1(x), inplace=True)
        #print('this is after conv1 x shape:'+str(out.shape))
        # copy to nf
        tmp = out
        while out.shape[1]<self.nf:
            out = torch.cat((out,out), dim=1)
        out = out[:,:self.nf,:,:]
        #print('this is copy2 x shape:'+str(out.shape))
        out = self.conv2(out)
        #print('this is after conv2 x shape:'+str(out.shape))
        return identity + out

class ResidualBlock_noBN_t2(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_t2, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,True])
        self.n_feats = nf
        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)
    
    def set_width(self, width):
        self.conv1.width_mult = width
        #self.conv2.width = width
        self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        #print('this is in resblock:  shape0 = '+str(x.shape))
        out = F.relu(self.conv1(x), inplace=True)
        #print('this is in resblock:  shape1 = '+str(out.shape))
        out = self.conv2(out)
        #print('this is in resblock:  shape2 = '+str(out.shape))
        # copy 0 to nf
        num_channels = out.size(1)
        if num_channels < self.n_feats:
            out = F.pad(out, (0, 0, 0, 0, 0, self.n_feats - num_channels))
        #print('this is in resblock:  shape4 = '+str(out.shape))
        #print('sum:'+str(torch.sum(out[:, -(self.n_feats - num_channels):, ...])))
        return identity + out
class ResidualBlock_noBN_t3(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_t3, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,True])
        # 1*1 conv
        # stride = 1 , padding = 0
        self.conv3 = USConv2d(nf, nf, 1, 1, 0, bias=True, us=[True,True])
        self.n_feats = nf
        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)
    
    def set_width(self, width):
        self.conv1.width_mult = width
        #self.conv2.width = width
        self.conv2.set_width_mult(width)
        # 这里conv3其实可以直接set width = 1-width anyway设在后面了

    def forward(self, x):
        identity = x
        #print('this is in resblock:  shape0 = '+str(x.shape))
        out = F.relu(self.conv1(x), inplace=True)
        #print('this is in resblock:  shape1 = '+str(out.shape))
        out = self.conv2(out)
        #print('this is in resblock:  shape2 = '+str(out.shape))
        
        num_channels = out.shape[1]
        if num_channels < self.n_feats:
            #self.conv3.set_width_mult(self.n_feats - num_channels)
            self.conv3.width_mult = (self.n_feats - num_channels) / self.n_feats
            #print('self.nfeat:'+str(self.n_feats))
            #print('num-channels:'+str(num_channels))
            #print(str(self.n_feats - num_channels))
            tmp = self.conv3(out)
            #print('this is in resblock:  shape tmp = '+str(tmp.shape))
            out = torch.cat((out,tmp), dim=1)
        #print('this is in resblock:  shape4 = '+str(out.shape))
        return identity + out


        class ResidualBlock_noBN_v5(nn.Module): # resblock for v5
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN_v5, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = nn.Conv2d(nf, nf, 3, stride = 1, padding = 1)
        self.conv25 = nn.Conv2d(nf//4, nf*3//4, 1, stride = 1, padding = 0)
        self.n_feats = nf

        # initialization
        #arch_util.initialize_weights([self.conv1, self.conv2], 0.1) # ver1
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv25 ], 0.1) # ver2

    def set_width(self, width):
        self.conv1.set_width_mult(width)
        #self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        #print('----------')
        #print('this is in resblock, x shape:'+str(x.shape))
        #out = F.relu(self.conv1(x), inplace=True)
        out = self.conv1(x)
        #print('this is in resblock, after conv1:'+str(out.shape))
        ch = out.shape[1]
        weight_size = self.conv25.weight.shape
        if ch < self.n_feats:
            if ch/self.n_feats == 0.25:
                tmp = self.conv25(out)
            elif ch/self.n_feats == 0.5:
                weight = self.conv25.weight[:int(weight_size[0]*2/3), :, :, :]
                weight = torch.cat((weight,weight),dim=1)
                bias = self.conv25.bias[:int(weight_size[0]*2/3)]
                tmp = F.conv2d(out, weight, bias, stride = 1, padding = 0)
            
            elif ch/self.n_feats == 0.75:
                #y = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
                weight = self.conv25.weight[:int(weight_size[0]*1/3), :, :, :]
                weight1 = torch.cat((weight,weight),dim=1)
                weight = torch.cat((weight1,weight),dim=1)
                bias = self.conv25.bias[:int(weight_size[0]*1/3)]
                tmp = F.conv2d(out, weight, bias, stride = 1, padding = 0)

            #print('rate:'+str(ch/self.n_feats)+' tmp:'+str(tmp.shape))
            out = torch.cat((out, tmp), dim=1)
            #print('after cat:'+str(out.shape))

        out = F.relu(out)
        out = self.conv2(out)
        #print('this is in resblock, after conv2:'+str(out.shape))
        #print('----------')
        return identity + out

class ResidualBlock_noBN_v5_train(nn.Module): # resblock for v5
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN_v5_train, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = nn.Conv2d(nf, nf, 3, stride = 1, padding = 1)
        self.conv25 = nn.Conv2d(nf//4, nf*3//4, 1, stride = 1, padding = 0)
        self.n_feats = nf

        # initialization
        #arch_util.initialize_weights([self.conv1, self.conv2], 0.1) # ver1
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv25 ], 0.1) # ver2

    def set_width(self, width):
        self.conv1.set_width_mult(width)
        #self.conv2.set_width_mult(width)

    def forward(self, x):
        identity = x
        #print('----------')
        #print('this is in resblock, x shape:'+str(x.shape))
        out = self.conv1(x)
        #print('this is in resblock, after conv1:'+str(out.shape))
        ch = out.shape[1]
        if ch < self.n_feats: # =0.25
            tmp = self.conv25(out)
            #print('rate:'+str(ch/self.n_feats)+' tmp:'+str(tmp.shape))
            out = torch.cat((out, tmp), dim=1)
            #print('after cat:'+str(out.shape))

        out = F.relu(out)
        out = self.conv2(out)
        #print('this is in resblock, after conv2:'+str(out.shape))
        #print('----------')
        return identity + out

# ------------以上部分主要用于对比测试，在最上面常用部分的基础上改的-----------------
# -------------------------------------------------------------------------------------

class Upsampler(nn.Sequential): # 这个函数没用到，先不改
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

