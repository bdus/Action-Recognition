#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-2-4 19:58:30

@author: bdus

https://gluon-cv.mxnet.io/build/examples_action_recognition/dive_deep_ucf101.html#start-training-now

第一个实验测试 比较

20 batch size 1 GPU

resnet18_v1b_ucf101  param_rgb_resnet18_v1b_ucf101  0.7505-ucf101-resnet18_v1b_ucf101-085-best.params
val top1 =0.750463 top5=0.939730 val loss=0.910375 time = 19.721219
val top1 =0.750463 top5=0.939730 val loss=0.910375 time = 19.850660
val top1 =0.750463 top5=0.939730 val loss=0.910375 time = 19.212582
time per clip : 0.506932


resnet18_v1b_k400_ucf101 param_rgb_resnet18_v1b_k400_ucf101 0.8620-ucf101-resnet18_v1b_k400_ucf101-082-best.
val top1 =0.813376 top5=0.952154 val loss=0.738654 time = 19.535844
val top1 =0.813376 top5=0.952154 val loss=0.738654 time = 19.611750
val top1 =0.813376 top5=0.952154 val loss=0.738654 time = 19.054030
time per clip : 0.502749


resnet34_v1b_ucf101 param_rgb_resnet34_v1b_ucf101_seg8 0.8655-ucf101-resnet34_v1b_ucf101-059-best.params
val top1 =0.808882 top5=0.954533 val loss=0.743578 time = 20.120073
val top1 =0.808882 top5=0.954533 val loss=0.743578 time = 19.229614
val top1 =0.808882 top5=0.954533 val loss=0.743578 time = 18.905875
time per clip : 0.498842


resnet34_v1b_k400_ucf101 param_rgb_resnet34_v1b_k400_ucf101_seg8 0.9212-ucf101-resnet34_v1b_k400_ucf101-010-best
val top1 =0.855670 top5=0.974623 val loss=0.541763 time = 20.033465
val top1 =0.855670 top5=0.974623 val loss=0.541763 time = 20.089865
val top1 =0.855670 top5=0.974623 val loss=0.541763 time = 19.186008
time per clip : 0.506231

resnet50_v1b_ucf101 param_rgb_resnet50_v1b_ucf101_seg8 0.8736-ucf101-resnet50_v1b_ucf101-026-best.params
val top1 =0.813640 top5=0.948454 val loss=0.872196 time = 18.741732
val top1 =0.813640 top5=0.948454 val loss=0.872196 time = 18.691594
val top1 =0.813640 top5=0.948454 val loss=0.872196 time = 19.068352
time per clip : 0.503126

resnet50_v1b_k400_ucf101 param_rgb_resnet50_v1b_k400_ucf101_seg8 0.9244-ucf101-resnet50_v1b_k400_ucf101-024-best
val top1 =0.868358 top5=0.970394 val loss=0.584157 time = 18.496983
val top1 =0.868358 top5=0.970394 val loss=0.584157 time = 19.506078
val top1 =0.868358 top5=0.970394 val loss=0.584157 time = 18.517787
time per clip : 0.488600

resnet101_v1b_k400_ucf101 param_rgb_resnet101_v1b_k400_ucf101_seg8 0.9344-ucf101-resnet101_v1b_k400_ucf101-021-best
val top1 =0.867565 top5=0.971187 val loss=0.721035 time = 18.768587
val top1 =0.867565 top5=0.971187 val loss=0.721035 time = 19.044027

val top1 =0.867565 top5=0.971187 val loss=0.721035 time = 18.717636
val top1 =0.867565 top5=0.971187 val loss=0.721035 time = 20.150343
val top1 =0.867565 top5=0.971187 val loss=0.721035 time = 19.247428
time per clip : 0.507852

resnet152_v1b_k400_ucf101 param_rgb_resnet152_v1b_k400_ucf101_seg8 0.9292-ucf101-resnet152_v1b_k400_ucf101-025-best
val top1 =0.857785 top5=0.963521 val loss=0.878501 time = 23.915603
val top1 =0.857785 top5=0.963521 val loss=0.878501 time = 24.255287

val top1 =0.857785 top5=0.963521 val loss=0.878501 time = 23.940408
val top1 =0.857785 top5=0.963521 val loss=0.878501 time = 24.047341
val top1 =0.857785 top5=0.963521 val loss=0.878501 time = 24.073693
time per clip : 0.635194

pretrained

resnet34_v1b_ucf101     param_rgb_resnet34_v1b_ucf101_seg8  0.8641-ucf101-resnet34_v1b_ucf101-014-best.params
resnet34_v1b_k400_ucf101     param_rgb_resnet34_v1b_k400_ucf101_seg8     0.9165-ucf101-resnet34_v1b_k400_ucf101-006-best.params



resnet34_v1b_k400_ucf101 param_rgb_resnet34_v1b_ucf101_seg8_scratch_real 0.4475-ucf101-resnet34_v1b_ucf101-051-best.params
seg=1
val top1 =0.392546 top5=0.666402 val loss=2.942931 time = 19.610184
val top1 =0.392546 top5=0.666402 val loss=2.942931 time = 18.966630
val top1 =0.392546 top5=0.666402 val loss=2.942931 time = 18.700061
time per clip : 0.493409
seg=8

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
        self.new_length = 1
        self.model = 'resnet34_v1b_k400_ucf101'
        self.save_dir = 'logs/test1'
        self.model_file = 'logs/param_rgb_resnet34_v1b_ucf101_seg8_scratch_real'
        self.resume_params = os.path.join(self.model_file,'0.4475-ucf101-resnet34_v1b_ucf101-051-best.params')
        self.num_classes = 101
        self.new_length_diff = self.new_length +1 
        self.train_dir = os.path.expanduser('~/.mxnet/datasets/ucf101/rawframes')#'/media/hp/mypan/BGSDecom/cv_MOG2/fgs')#
        self.train_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
        self.val_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
        self.logging_file = 'train.log'
        self.name_pattern='img_%05d.jpg'
        self.dataset = 'ucf101'
        self.input_size=224#224#112#204#112
        self.new_height=256#256#128#256#128
        self.new_width=340#340#171#340#171
        self.input_channel=3 
        self.num_segments=8
        self.num_workers = 1
        self.num_gpus = 1
        self.per_device_batch_size = 10
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
        self.partial_bn = True
        self.clip_grad = 40
        self.log_interval = 10
        self.lr_mode = 'step'        
        self.resume_epoch = 0  
        self.reshape_type = 'tsn' # c3d tsn tsn_newlength
#self.resume_states = os.path.join(self.model_file,'0.6122-ucf101-resnet18_v1b_k400_ucf101-098-best.states')      

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
net = myget(name=opt.model, nclass=opt.num_classes, num_segments=opt.num_segments,input_channel=opt.input_channel,batch_normal=opt.partial_bn)
net.cast(opt.dtype)
net.collect_params().reset_ctx(ctx)
#logger.info(net)
if opt.resume_params is not '':
    net.load_parameters(opt.resume_params, ctx=ctx)#,allow_missing=True,ignore_extra=True)

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

#train_dataset = UCF101(setting=opt.train_setting, root=opt.train_dir, train=True,
                      # new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,
                      # target_width=opt.input_size, target_height=opt.input_size,
                      # num_segments=opt.num_segments, transform=transform_train)
val_dataset = UCF101(setting=opt.val_setting, root=opt.train_dir, train=False,test_mode=True,
                     new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,
                     target_width=opt.input_size, target_height=opt.input_size,
                     num_segments=opt.num_segments, transform=transform_test)


#train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   #prefetch=int(opt.prefetch_ratio * num_workers), last_batch='rollover')
val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 prefetch=int(opt.prefetch_ratio * num_workers), last_batch='discard')


#train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                   #shuffle=True, num_workers=num_workers)
#logger.info('Load %d training samples.' % len(train_dataset))
val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    
def batch_fn(batch, ctx):
    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label
# Learning rate decay factor
lr_decay = opt.lr_decay
# Epochs where learning rate decays
lr_decay_epoch = opt.lr_decay_epoch



# Define our trainer for net
#trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

train_history = TrainingHistory(['training-acc','val-top1-acc','val-top5-acc'])

lr_decay_count = 0
best_val_score = 0

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)


#perclip_time = []



def test(ctx,val_data):
    acc_top1.reset()
    acc_top5.reset()    
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    num_test_iter = len(val_data)
    val_loss_epoch = 0
      
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        #tic = time.time()
        val_outputs = []
        for _, X in enumerate(data):
            if opt.reshape_type == 'tsn':
                X = X.reshape((-1,) + X.shape[2:])
            elif opt.reshape_type == 'c3d':
                X = nd.transpose(data=X,axes=(0,2,1,3,4))
            elif opt.new_length != 1 and opt.reshape_type == 'tsn_newlength':
                X = X.reshape((-3,-3,-2))
            else:
                pass
            pred = net(X)
            val_outputs.append(pred)
        #perclip_time.append( time.time() - tic )
        loss = [L(yhat, y) for yhat, y in zip(val_outputs, label)]
        
        acc_top1.update(label, val_outputs)
        acc_top5.update(label, val_outputs)
        
        val_loss_epoch += sum([l.mean().asscalar() for l in loss]) / len(loss)
    
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    val_loss = val_loss_epoch / num_test_iter
    
    return (top1, top5, val_loss)

# training 


tic = time.time()
acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
        (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))

tic = time.time()
acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
        (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))
       
tic = time.time()
acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
        (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))        

logger.info('time per clip : %f ' % ((time.time() - tic)* batch_size / len(val_data) ) )        
print('done.')