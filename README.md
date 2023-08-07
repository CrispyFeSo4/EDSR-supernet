# EDSR-supernet
Official implementation of **Meta-SR: A Magnification-Arbitrary Network for Super-Resolution(CVPR2019)(PyTorch)**
  
[Paper](https://arxiv.org/pdf/1903.00875.pdf)

This code is built on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and Meta-SR.

# Notice when checking the code

## Main code:
* /main.py
* /trainer.py: trainer initialization; define the method of obtaining training dataset and the training probability for each scale
* /option.py: define default parameters

main.py和option.py没有修改过；trainer.py增加了 select_element_with_probability() 函数，用于以一定概率控制抽取scale，此外修改了train()函数，以指定数值抽取特定scale训练数据。



## Model code:
* /model/arch_util.py: class USConv2d from ARM-Net(cbh)
* /model/common.py：class ResBlock + Resblock各种变体的类定义
* /model/metaedsrori.py: ori code (conv1[64->64]->relu->conv2[64->64])
* /model/edsr_layer_v2.py: v2 (conv1[64->32]->relu->conv2[32->64])
* /model/edsr_layer_0copy.py: (conv1[64->32]->relu->补0回64->conv2[64->64])
* /model/edsr_layer_test4.py: (conv1[64->64，后32权重×0]->relu->conv2[64->64])
* /model/edsr_layer_11conv3.py: (conv1[64->32]->relu->3个固定宽度1*1卷积补回64->conv2[64->64])
* /model/edsr_layer_11usconv.py: (conv1[64->32]->relu->1个可变宽度1*1卷积补回64->conv2[64->64])

以上channel举例建立在width=0.5基础上。


## !!!Notice
1. v2、0copy、test4理论上应等价，但实际实验中，0copy和test4性能均低于v2；（可能这部分代码有错）

2. ori和v2相比，ori的参数量更多，却性能更差；（为什么？）

3. 1×1conv×3和v2相比，v2的参数更多，却性能更差；（为什么？）

检查时可以按照这样的顺序：arch_util.py（看USConv2d）、common.py(看各个模型类定义及forward)、v2、11conv3、...

其余有些代码是冗余代码，为训练方便所用，可暂不关注。


# Requirements

* Pytorch 1.1.0
* Python 3.6
* numpy
* skimage
* imageio
* cv2  
* tqdm

*note that the dataloader.py has been rewritten to accommodate the 1.1.0 version of the pytorch.


# Install
download the code
```
git clone https://github.com/CrispyFeSo4/EDSR-supernet.git
cd EDSR-supernet
```


# Train and Test

## prepare dataset

  * Download the train dataset [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) and the test dataset benchmark.
  * Run the /prepare_dataset/geberate_LR_metasr_X1_X4.m on matlab to generate LR images.Remember to modify path_src = DIV2K HR image folder path.
  * Upload the dataset
  * Modify the option.py file:
dir_data = "/path to your DIV2K and testing dataset'(keep the training and test dataset in the same folder: test dataset under the benchmark folder and training dataset rename to DIV2K, or change the data_train to your folder name)

## train 
```
cd /EDSR-supernet
python main.py --model edsr_layer_v2 --save edsr_layer_v2 --ext sep --lr_decay 200 --epochs 100 --n_GPUs 2 --batch_size 8  
```
## test 
```
python main.py --model edsr_layer_v2 --save edsr_layer_v2 --ext sep --pre_train ./experiment/edsr_layer_v2/model/model_100.pt --test_only --data_test Set5  --scale 3.2 --scale2 3.2 --n_GPUs 1
```


# Citation
```
@article{hu2019meta,
  title={Meta-SR: A Magnification-Arbitrary Network for Super-Resolution},
  author={Hu, Xuecai and Mu, Haoyuan and Zhang, Xiangyu and Wang, Zilei  and Tan, Tieniu and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
# Contact
Xuecai Hu (huxc@mail.ustc.edu.cn)
