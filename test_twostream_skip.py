#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-2-4 19:58:30 2020年3月2日14:53:36

@author: bdus

https://gluon-cv.mxnet.io/build/examples_action_recognition/dive_deep_ucf101.html#start-training-now

测试两个模型用
分别导入两个模型和参数文件，然后max mean

1 resnet34_v1b_ucf101 param_rgb_resnet34_v1b_ucf101_seg8 0.8655-ucf101-resnet34_v1b_ucf101-059-best.params
2 resnet34_v1b_k400_ucf101 param_rgb_resnet34_v1b_k400_ucf101_seg8 0.9212-ucf101-resnet34_v1b_k400_ucf101-010-best


#resnet34_v1b_k400_ucf101 param_cvMOG2_resnet34_v1b_k400_ucf101_seg8  0.6178-ucf101-resnet34_v1b_k400_ucf101-054-best

3 
resnet34_v1b_k400_ucf101 param_cvMOG2_resnet34_v1b_k400_ucf101_seg8_1 0.8097-ucf101-resnet34_v1b_k400_ucf101-049-best
new_length == 5 
reshape_type tsn_newlength

2 + 3
seg = 3
avg
val top1 =0.912434 top5=0.987566 val loss=0.329578 time = 237.020239
opt.fusion_method :  bgs
val top1 =0.911111 top5=0.986508 val loss=0.333180 time = 143.069351
opt.fusion_method :  fgs
val top1 =0.761905 top5=0.924603 val loss=0.968371 time = 142.306687
opt.fusion_method :  avg
val top1 =0.912434 top5=0.987566 val loss=0.329578 time = 142.208594
done.


two stream == 3
k400
opt.fusion_method :  bgs
val top1 =0.856085 top5=0.974603 val loss=0.537513 time = 127.967459
opt.fusion_method :  fgs
val top1 =0.761905 top5=0.924603 val loss=0.968371 time = 125.402178
opt.fusion_method :  avg
val top1 =0.900000 top5=0.983333 val loss=0.375273 time = 125.079985
done.
IMAGENET
opt.fusion_method :  bgs
val top1 =0.808730 top5=0.954497 val loss=0.745272 time = 128.198030
opt.fusion_method :  fgs
val top1 =0.761905 top5=0.924603 val loss=0.968371 time = 127.346968
opt.fusion_method :  avg
val top1 =0.875661 top5=0.975132 val loss=0.443755 time = 127.579867
done.

16
opt.fusion_method :  bgs
val top1 =0.808730 top5=0.954497 val loss=0.745272 time = 844.378210
opt.fusion_method :  fgs
val top1 =0.816931 top5=0.951323 val loss=0.696584 time = 750.350157
opt.fusion_method :  avg
val top1 =0.887566 top5=0.981746 val loss=0.388227 time = 749.209106
done.




num_segments_bgs=16,num_segments_fgs=16,
opt.fusion_method :  avg
val top1 =0.927660 top5=0.992021 val loss=0.267080 time = 956.126978
opt.fusion_method :  avg
val top1 =0.927660 top5=0.992021 val loss=0.267080 time = 820.728109
opt.fusion_method :  avg
val top1 =0.927660 top5=0.992021 val loss=0.267080 time = 808.047090
done.

num_segments_bgs=8,num_segments_fgs=8
opt.fusion_method :  avg
val top1 =0.923138 top5=0.990691 val loss=0.277033 time = 688.619913
opt.fusion_method :  avg
val top1 =0.923138 top5=0.990691 val loss=0.277033 time = 395.748669
opt.fusion_method :  avg
val top1 =0.923138 top5=0.990691 val loss=0.277033 time = 400.638762
done.

num_segments_bgs=4,num_segments_fgs=4
opt.fusion_method :  avg
val top1 =0.913564 top5=0.990957 val loss=0.293181 time = 189.081963
opt.fusion_method :  avg
val top1 =0.913564 top5=0.990957 val loss=0.293181 time = 187.755379
opt.fusion_method :  avg
val top1 =0.913564 top5=0.990957 val loss=0.293181 time = 187.495777
done.


num_segments_bgs=1,num_segments_fgs=4

opt.fusion_method :  avg
val top1 =0.903723 top5=0.986968 val loss=0.332307 time = 166.249061
opt.fusion_method :  avg
val top1 =0.903723 top5=0.986968 val loss=0.332307 time = 168.985077
opt.fusion_method :  avg
val top1 =0.903723 top5=0.986968 val loss=0.332307 time = 165.744605
done.

num_segments_bgs=1,num_segments_fgs=8
opt.fusion_method :  avg
val top1 =0.908511 top5=0.986170 val loss=0.321312 time = 339.186612
opt.fusion_method :  avg
val top1 =0.908511 top5=0.986170 val loss=0.321312 time = 343.205953
opt.fusion_method :  avg
 val top1 =0.908511 top5=0.986170 val loss=0.321312 time = 340.908357
done.


num_segments_bgs=1,num_segments_fgs=16,
val top1 =0.909574 top5=0.985904 val loss=0.314644 time = 748.758741
opt.fusion_method :  avg
val top1 =0.909574 top5=0.985904 val loss=0.314644 time = 736.712456
opt.fusion_method :  avg
^[^[val top1 =0.909574 top5=0.985904 val loss=0.314644 time = 736.739714
done.


resnet101_v1b_k400_ucf101 param_rgb_resnet101_v1b_k400_ucf101_seg8 0.9344-ucf101-resnet101_v1b_k400_ucf101-021-best
num_segments_bgs=1,num_segments_fgs=16
val top1 =0.915160 top5=0.988564 val loss=0.331644 time = 739.039340
num_segments_bgs=2,num_segments_fgs=16
val top1 =0.930319 top5=0.990957 val loss=0.256274 time = 749.338600

hmdb51
101
resnet34_v1b_ucf101 param_rgb_resnet34_v1b_hmdb51_seg8_ucf101pretrainedk400 0.5941-hmdb51-resnet34_v1b_ucf101-012-best.params
resnet34_v1b_k400_ucf101 param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_1_ucf101pretrained 0.5340-hmdb51-resnet34_v1b_k400_ucf101-061-best.params

num_segments_bgs=8,num_segments_fgs=8,
[config:new_length_fgs=5,new_length_bgs=1,fusion_method=avg,bgsmodel=resnet34_v1b_ucf101,fgsmodel=resnet34_v1b_k400_ucf101,save_dir=logs/test4,bgs_path=logs/param_rgb_resnet34_v1b_hmdb51_seg8_ucf101pretrainedk400,fgs_path=logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_1_ucf101pretrained,bgs_params=logs/param_rgb_resnet34_v1b_hmdb51_seg8_ucf101pretrainedk400/0.5941-hmdb51-resnet34_v1b_ucf101-012-best.params,fgs_params=logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_1_ucf101pretrained/0.5340-hmdb51-resnet34_v1b_k400_ucf101-061-best.params,num_classes=101,root_bgs=/home/hp/.mxnet/datasets/ucf101/rawframes,root_fgs=/media/hp/mypan/BGSDecom/cv_MOG2/fgs,train_setting=/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt,val_setting=/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt,root_bgs_hmdb51=/home/hp/.mxnet/datasets/hmdb51/rawframes,root_fgs_hmdb51=/media/hp/8tB/BGSDecom_hmdb51/cv_MOG2/fgs,val_setting_hmdb51=/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt,logging_file=train.log,name_pattern=img_%05d.jpg,dataset=hmdb51,input_size=224,new_height=256,new_width=340,input_channel_fgs=15,input_channel_bgs=3,num_segments_bgs=8,num_segments_fgs=8,num_workers=2,num_gpus=1,per_device_batch_size=20,lr=0.01,lr_decay=0.1,warmup_lr=0,warmup_epochs=0,momentum=0.9,wd=0.0005,prefetch_ratio=1.0,use_amp=False,epochs=100,lr_decay_epoch=[30, 60, 80],dtype=float32,use_pretrained=False,partial_bn=False,clip_grad=40,log_interval=10,lr_mode=step,resume_epoch=1,reshape_type_bgs=tsn,reshape_type_fgs=tsn_newlength]
Load 76 training samples.
val top1 =0.594737 top5=0.854605 val loss=1.578756 time = 158.360253 
val top1 =0.534868 top5=0.815132 val loss=1.959082 time = 160.414971 
val top1 =0.672368 top5=0.892763 val loss=1.248599 time = 164.093125 
val top1 =0.602632 top5=0.872368 val loss=1.560393 time = 164.736566 

51
resnet34_v1b_ucf101  param_rgb_resnet34_v1b_hmdb51_seg8  0.5222-hmdb51-resnet34_v1b_ucf101-013-best.params
resnet34_v1b_k400_ucf101 param_rgb_resnet34_v1b_k400_hmdb51_seg8 0.6719-hmdb51-resnet34_v1b_k400_ucf101-010-best.params

resnet34_v1b_k400_ucf101 param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8 0.4948-hmdb51-resnet34_v1b_k400_ucf101-045-best.params
resnet34_v1b_k400_ucf101 param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_pt1 0.5007-hmdb51-resnet34_v1b_k400_ucf101-048-best.params


[config:new_length_fgs=5,new_length_bgs=1,fusion_method=avg,bgsmodel=resnet34_v1b_ucf101,fgsmodel=resnet34_v1b_k400_ucf101,save_dir=logs/test4,bgs_path=logs/param_rgb_resnet34_v1b_hmdb51_seg8,fgs_path=logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8,bgs_params=logs/param_rgb_resnet34_v1b_hmdb51_seg8/0.5222-hmdb51-resnet34_v1b_ucf101-013-best.params,fgs_params=logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8/0.4948-hmdb51-resnet34_v1b_k400_ucf101-045-best.params,num_classes=51,root_bgs=/home/hp/.mxnet/datasets/ucf101/rawframes,root_fgs=/media/hp/mypan/BGSDecom/cv_MOG2/fgs,train_setting=/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt,val_setting=/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt,root_bgs_hmdb51=/home/hp/.mxnet/datasets/hmdb51/rawframes,root_fgs_hmdb51=/media/hp/8tB/BGSDecom_hmdb51/cv_MOG2/fgs,val_setting_hmdb51=/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt,logging_file=train.log,name_pattern=img_%05d.jpg,dataset=hmdb51,input_size=224,new_height=256,new_width=340,input_channel_fgs=15,input_channel_bgs=3,num_segments_bgs=8,num_segments_fgs=8,num_workers=2,num_gpus=1,per_device_batch_size=20,lr=0.01,lr_decay=0.1,warmup_lr=0,warmup_epochs=0,momentum=0.9,wd=0.0005,prefetch_ratio=1.0,use_amp=False,epochs=100,lr_decay_epoch=[30, 60, 80],dtype=float32,use_pretrained=False,partial_bn=False,clip_grad=40,log_interval=10,lr_mode=step,resume_epoch=1,reshape_type_bgs=tsn,reshape_type_fgs=tsn_newlength]
Load 76 training samples.
opt.fusion_method :  bgs
val top1 =0.523026 top5=0.832895 val loss=1.695794 time = 159.960592
opt.fusion_method :  fgs
val top1 =0.496053 top5=0.781579 val loss=2.299345 time = 160.342455
opt.fusion_method :  avg
val top1 =0.609868 top5=0.875000 val loss=1.456212 time = 165.639398
opt.fusion_method :  max
val top1 =0.542105 top5=0.848684 val loss=1.821423 time = 163.943371
done.

[config:new_length_fgs=5,new_length_bgs=1,fusion_method=avg,bgsmodel=resnet34_v1b_k400_ucf101,fgsmodel=resnet34_v1b_k400_ucf101,save_dir=logs/test4,bgs_path=logs/param_rgb_resnet34_v1b_k400_hmdb51_seg8,fgs_path=logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_pt1,bgs_params=logs/param_rgb_resnet34_v1b_k400_hmdb51_seg8/0.6719-hmdb51-resnet34_v1b_k400_ucf101-010-best.params,fgs_params=logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_pt1/0.5007-hmdb51-resnet34_v1b_k400_ucf101-048-best.params,num_classes=51,root_bgs=/home/hp/.mxnet/datasets/ucf101/rawframes,root_fgs=/media/hp/mypan/BGSDecom/cv_MOG2/fgs,train_setting=/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt,val_setting=/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt,root_bgs_hmdb51=/home/hp/.mxnet/datasets/hmdb51/rawframes,root_fgs_hmdb51=/media/hp/8tB/BGSDecom_hmdb51/cv_MOG2/fgs,val_setting_hmdb51=/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt,logging_file=train.log,name_pattern=img_%05d.jpg,dataset=hmdb51,input_size=224,new_height=256,new_width=340,input_channel_fgs=15,input_channel_bgs=3,num_segments_bgs=8,num_segments_fgs=8,num_workers=2,num_gpus=1,per_device_batch_size=20,lr=0.01,lr_decay=0.1,warmup_lr=0,warmup_epochs=0,momentum=0.9,wd=0.0005,prefetch_ratio=1.0,use_amp=False,epochs=100,lr_decay_epoch=[30, 60, 80],dtype=float32,use_pretrained=False,partial_bn=False,clip_grad=40,log_interval=10,lr_mode=step,resume_epoch=1,reshape_type_bgs=tsn,reshape_type_fgs=tsn_newlength]
Load 76 training samples.
opt.fusion_method :  bgs
val top1 =0.673684 top5=0.909868 val loss=1.165904 time = 156.726478 
opt.fusion_method :  fgs
val top1 =0.494737 top5=0.789474 val loss=2.267650 time = 161.684037 
opt.fusion_method :  avg
val top1 =0.668421 top5=0.913158 val loss=1.209727 time = 161.845934 
opt.fusion_method :  max
val top1 =0.609211 top5=0.882237 val loss=1.545654 time = 161.387405 
done.

[config:new_length_fgs=5,new_length_bgs=1,fusion_method=avg,bgsmodel=resnet34_v1b_ucf101,fgsmodel=resnet34_v1b_k400_ucf101,save_dir=logs/test4,bgs_path=logs/param_rgb_resnet34_v1b_hmdb51_seg8,fgs_path=logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_pt1,bgs_params=logs/param_rgb_resnet34_v1b_hmdb51_seg8/0.5222-hmdb51-resnet34_v1b_ucf101-013-best.params,fgs_params=logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_pt1/0.5007-hmdb51-resnet34_v1b_k400_ucf101-048-best.params,num_classes=51,root_bgs=/home/hp/.mxnet/datasets/ucf101/rawframes,root_fgs=/media/hp/mypan/BGSDecom/cv_MOG2/fgs,train_setting=/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt,val_setting=/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt,root_bgs_hmdb51=/home/hp/.mxnet/datasets/hmdb51/rawframes,root_fgs_hmdb51=/media/hp/8tB/BGSDecom_hmdb51/cv_MOG2/fgs,val_setting_hmdb51=/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt,logging_file=train.log,name_pattern=img_%05d.jpg,dataset=hmdb51,input_size=224,new_height=256,new_width=340,input_channel_fgs=15,input_channel_bgs=3,num_segments_bgs=8,num_segments_fgs=8,num_workers=2,num_gpus=1,per_device_batch_size=20,lr=0.01,lr_decay=0.1,warmup_lr=0,warmup_epochs=0,momentum=0.9,wd=0.0005,prefetch_ratio=1.0,use_amp=False,epochs=100,lr_decay_epoch=[30, 60, 80],dtype=float32,use_pretrained=False,partial_bn=False,clip_grad=40,log_interval=10,lr_mode=step,resume_epoch=1,reshape_type_bgs=tsn,reshape_type_fgs=tsn_newlength]
Load 76 training samples.
opt.fusion_method :  bgs
val top1 =0.523026 top5=0.832895 val loss=1.695794 time = 160.113551 
opt.fusion_method :  fgs
val top1 =0.494737 top5=0.789474 val loss=2.267650 time = 160.815939 
opt.fusion_method :  avg
val top1 =0.611842 top5=0.877632 val loss=1.435926 time = 162.629882 
opt.fusion_method :  max
val top1 =0.552632 top5=0.855921 val loss=1.748185 time = 163.572623 
done.


"""
from __future__ import division

import argparse, time, logging, os, sys, math
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
os.environ['CUDA_VISIBLE_DEVICES']='1' #0,1

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter

from gluoncv.data.transforms import video
from gluoncv.data import UCF101, Kinetics400, SomethingSomethingV2, HMDB51
#from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory

from model_zoo import get_model as myget

# from DualBlock import DualBlock,get_dualnet
# from ucf101_bgs.classification import UCF101_Bgs 
from ucf101_bgs.twostream import UCF101_2stream


class AttrDisplay:
  def gatherAttrs(self):
    return ",".join("{}={}"
            .format(k, getattr(self, k))
            for k in self.__dict__.keys())
    # attrs = []
    # for k in self.__dict__.keys():
    #   item = "{}={}".format(k, getattr(self, k))
    #   attrs.append(item)
    # return attrs
    # for k in self.__dict__.keys():
    #   attrs.append(str(k) + "=" + str(self.__dict__[k]))
    # return ",".join(attrs) if len(attrs) else 'no attr'
  def __str__(self):
    return "[{}:{}]".format(self.__class__.__name__, self.gatherAttrs())

class config(AttrDisplay):
    def __init__(self):
        self.new_length_fgs = 5
        self.new_length_bgs = 1
        self.new_step_fgs=1
        self.new_step_bgs=1
        self.fusion_method = 'avg'# avg max bgs fgs
        self.bgsmodel = 'resnet34_v1b_ucf101'#'resnet34_v1b_ucf101'
        self.fgsmodel = 'resnet34_v1b_k400_ucf101'
        self.save_dir = 'logs/test4'
        self.bgs_path = 'logs/param_rgb_resnet34_v1b_hmdb51_seg8'
        self.fgs_path = 'logs/param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_pt1'#param_cvMOG2_resnet34_v1b_k400_hmdb51_seg8_1_ucf101pretrained'#param_cvMOG2_resnet34_v1b_k400_ucf101_seg8_1'
        self.bgs_params = os.path.join(self.bgs_path, '0.5222-hmdb51-resnet34_v1b_ucf101-013-best.params')#0.5222-hmdb51-resnet34_v1b_ucf101-013-best.params')#0.5941-hmdb51-resnet34_v1b_ucf101-012-best.params')##'0.9344-ucf101-resnet101_v1b_k400_ucf101-021-best.params')#'0.8655-ucf101-resnet34_v1b_ucf101-059-best.params')#
        self.fgs_params = os.path.join(self.fgs_path, '0.5007-hmdb51-resnet34_v1b_k400_ucf101-048-best.params')#0.5340-hmdb51-resnet34_v1b_k400_ucf101-061-best.params')#'0.8097-ucf101-resnet34_v1b_k400_ucf101-049-best.params'
        # self.new_length_diff = self.new_length +1 
        self.root_bgs = os.path.expanduser('~/.mxnet/datasets/ucf101/rawframes')
        self.root_fgs = os.path.expanduser('/media/hp/mypan/BGSDecom/cv_MOG2/fgs')#'~/.mxnet/datasets/ucf101/rawframes')
        self.train_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
        self.val_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
        self.root_bgs_hmdb51 = os.path.expanduser('~/.mxnet/datasets/hmdb51/rawframes')#
        self.root_fgs_hmdb51 = os.path.expanduser('/media/hp/8tB/BGSDecom_hmdb51/cv_MOG2/fgs')
        self.val_setting_hmdb51 = '/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt'			
        self.logging_file = 'train.log'
        self.name_pattern='img_%05d.jpg'
        self.dataset = 'hmdb51'#'ucf101'
        self.num_classes = 101 if self.dataset == 'ucf101' else 51
        self.input_size=224#112#204#112
        self.new_height=256#128#256#128
        self.new_width=340#171#340#171
        self.input_channel_fgs=15
        self.input_channel_bgs=3
        self.num_segments_bgs=8
        self.num_segments_fgs=8        
        self.num_workers = 2
        self.num_gpus = 1
        self.per_device_batch_size = 20
        self.lr = 0.01
        self.lr_decay = 0.1
        self.warmup_lr = 0
        self.warmup_epochs = 0
        self.momentum = 0.9
        self.wd = 0.0005        
        self.prefetch_ratio = 1.0
        self.use_amp = False
        self.epochs = 100
        self.lr_decay_epoch = [30,60,80]
        self.dtype = 'float32'
        self.use_pretrained = False
        self.partial_bn = False
        self.clip_grad = 40
        self.log_interval = 10
        self.lr_mode = 'step'        
        self.resume_epoch = 1
        self.reshape_type_bgs = 'tsn' # c3d tsn tsn_newlength
        self.reshape_type_fgs = 'tsn'#'tsn_newlength' # c3d tsn tsn_newlength
      

opt = config()

makedirs(opt.save_dir)

filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)
logger.info(opt)
    
# number of GPUs to use
num_gpus = opt.num_gpus
ctx = [mx.gpu(i) for i in range(num_gpus)]
#ctx = [mx.gpu(1)]

# Get the model 
#net = get_dualnet(fgs_model=opt.fgsmodel,bgs_model=opt.bgsmodel,fgs_path=opt.fgs_params,bgs_path=opt.bgs_params, nclass=opt.num_classes, num_segments=opt.num_segments,input_channel=opt.input_channel,fusion_method=opt.fusion_method)
net_bgs = myget(name=opt.bgsmodel, nclass=opt.num_classes, num_segments=opt.num_segments_bgs,input_channel=opt.input_channel_bgs,batch_normal=opt.partial_bn)
net_fgs = myget(name=opt.fgsmodel, nclass=opt.num_classes, num_segments=opt.num_segments_fgs,input_channel=opt.input_channel_fgs,batch_normal=opt.partial_bn)
net_bgs.cast(opt.dtype)
net_bgs.collect_params().reset_ctx(ctx)
net_fgs.cast(opt.dtype)
net_fgs.collect_params().reset_ctx(ctx)
#logger.info(net)
if opt.bgs_params is not '':
    net_bgs.load_parameters(opt.bgs_params, ctx=ctx,allow_missing=True)
if opt.fgs_params is not '':
    net_fgs.load_parameters(opt.fgs_params, ctx=ctx)

transform_train = video.VideoGroupTrainTransform(size=(opt.input_size, opt.input_size), scale_ratios=[1.0, 0.875, 0.75, 0.66], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = video.VideoGroupValTransform(size=opt.input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Batch Size for Each GPU
per_device_batch_size = opt.per_device_batch_size
# Number of data loader workers
num_workers = opt.num_workers
# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

# Set train=True for training data. Here we only use a subset of UCF101 for demonstration purpose.
# The subset has 101 training samples, one sample per class.

if opt.dataset == 'ucf101':
    val_dataset = UCF101_2stream(train=False, num_segments_bgs=opt.num_segments_bgs, 
                                          num_segments_fgs=opt.num_segments_fgs,
                                             transform=transform_test,
                                             root_bgs=opt.root_bgs,
                                             root_fgs=opt.root_fgs,
                                             setting=opt.val_setting,
                                             name_pattern=opt.name_pattern,
                         new_width=opt.new_width,new_height=opt.new_height, 
                         new_step_bgs=opt.new_step_bgs, new_step_fgs=opt.new_step_fgs, 
                         new_length_bgs=opt.new_length_bgs, new_length_fgs=opt.new_length_fgs,
                           target_width=opt.input_size, target_height=opt.input_size
                                             )
elif opt.dataset == 'hmdb51':
    val_dataset = UCF101_2stream(train=False, num_segments_bgs=opt.num_segments_bgs, 
                                          num_segments_fgs=opt.num_segments_fgs,
                                             transform=transform_test,
                                             root_bgs=opt.root_bgs_hmdb51,
                                             root_fgs=opt.root_fgs_hmdb51,
                                             setting=opt.val_setting_hmdb51,
                                             name_pattern=opt.name_pattern,
                         new_width=opt.new_width,new_height=opt.new_height, 
                         new_step_bgs=opt.new_step_bgs, new_step_fgs=opt.new_step_fgs, 
                         new_length_bgs=opt.new_length_bgs, new_length_fgs=opt.new_length_fgs,
                           target_width=opt.input_size, target_height=opt.input_size
                                             )


val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 prefetch=int(opt.prefetch_ratio * num_workers), last_batch='discard')
logger.info('Load %d training samples.' % len(val_data))
def batch_fn(batch, ctx):
    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label

# Define our trainer for net
#trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

train_history = TrainingHistory(['training-acc','val-top1-acc','val-top5-acc'])

lr_decay_count = 0
best_val_score = 0

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)


def test(ctx,val_data):
    acc_top1.reset()
    acc_top5.reset()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    num_test_iter = len(val_data)
    val_loss_epoch = 0
    for i, batch in enumerate(val_data):
        data_bgs = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        data_fgs = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        val_outputs = []
        for _, (X_bgs,X_fgs) in enumerate(zip(data_bgs,data_fgs)):
            #print('X_bgs',X_bgs.shape) # (10, 8, 3, 224, 224)
            #print('X_fgs',X_fgs.shape) # (10, 8, 3, 224, 224)                       
            
            if opt.reshape_type_bgs == 'tsn':
                X_bgs = X_bgs.reshape((-1,) + X_bgs.shape[2:])
            elif opt.reshape_type_bgs == 'tsn_newlength':
                X_bgs = X_bgs.reshape((-3,-3,-2))
            else:
                pass
            
            if opt.reshape_type_fgs == 'tsn':
                X_fgs = X_fgs.reshape((-1,) + X_fgs.shape[2:])
            elif opt.reshape_type_fgs == 'tsn_newlength':
                X_fgs = X_fgs.reshape((-3,-3,-2))
            else:
                pass                   
                                    
            #print('X_bgs',X_bgs.shape) #(80, 3, 224, 224)
            #print('X_fgs',X_fgs.shape) #(80, 3, 224, 224)
            x_bgs = net_bgs(X_bgs)
            x_fgs = net_fgs(X_fgs)            
            if opt.fusion_method == 'avg':
                x = nd.stack(x_bgs,x_fgs) 
                x = nd.mean(x, axis=0)                          
            elif opt.fusion_method == 'max':
                x = nd.stack(x_bgs,x_fgs) 
                x = nd.max(x,axis=0)
            elif opt.fusion_method == 'bgs':
                x = x_bgs
            elif opt.fusion_method == 'fgs':
                x = x_fgs
            else:
                raise ValueError('fusion_method not supported')
            pred=x                        
            #pred = net(X_bgs,X_fgs)
            val_outputs.append(pred)
            
        loss = [L(yhat, y) for yhat, y in zip(val_outputs, label)]
        
        acc_top1.update(label, val_outputs)
        acc_top5.update(label, val_outputs)
        
        val_loss_epoch += sum([l.mean().asscalar() for l in loss]) / len(loss)
    
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    val_loss = val_loss_epoch / num_test_iter
    
    return (top1, top5, val_loss)


# training 

#acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
#logger.info('val top1 =%f top5=%f val loss=%f' %
 #       (acc_top1_val, acc_top5_val, loss_val ))




opt.fusion_method = 'bgs'
print('opt.fusion_method : ',opt.fusion_method)

tic = time.time()
acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
        (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))


opt.fusion_method = 'fgs'
print('opt.fusion_method : ',opt.fusion_method)
       
tic = time.time()
acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
        (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size )) 

opt.fusion_method = 'avg'
print('opt.fusion_method : ',opt.fusion_method)

tic = time.time()
acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
        (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))

opt.fusion_method = 'max'
print('opt.fusion_method : ',opt.fusion_method)

tic = time.time()
acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
        (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))


print('done.')