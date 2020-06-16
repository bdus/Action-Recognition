#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-2-4 19:58:30 2020年3月2日14:53:36

@author: bdus

https://gluon-cv.mxnet.io/build/examples_action_recognition/dive_deep_ucf101.html#start-training-now

测试两个模型用
分别导入两个模型和参数文件，然后max mean

resnet34_v1b_ucf101 param_rgb_resnet34_v1b_ucf101_seg8 0.8655-ucf101-resnet34_v1b_ucf101-059-best.params
resnet34_v1b_k400_ucf101 param_rgb_resnet34_v1b_k400_ucf101_seg8 0.9212-ucf101-resnet34_v1b_k400_ucf101-010-best


resnet34_v1b_k400_ucf101 param_cvMOG2_resnet34_v1b_k400_ucf101_seg8  0.6178-ucf101-resnet34_v1b_k400_ucf101-054-best

imagenet
seg=1
val top1 =0.755405 top5=0.930541 val loss=0.905163 time = 56.943942
val top1 =0.755405 top5=0.930541 val loss=0.905163 time = 17.028004
val top1 =0.755405 top5=0.930541 val loss=0.905163 time = 17.048095
done.
seg16
val top1 =0.852116 top5=0.964815 val loss=0.534640 time = 259.290843
val top1 =0.852116 top5=0.964815 val loss=0.534640 time = 258.373130
val top1 =0.852116 top5=0.964815 val loss=0.534640 time = 259.765228
done.


k400
seg1
val top1 =0.786772 top5=0.938624 val loss=0.788636 time = 18.362909
val top1 =0.786772 top5=0.938624 val loss=0.788636 time = 17.855403
val top1 =0.786772 top5=0.938624 val loss=0.788636 time = 17.434568
done.
seg=16
val top1 =0.890212 top5=0.980159 val loss=0.417256 time = 261.250653
val top1 =0.890212 top5=0.980159 val loss=0.417256 time = 263.182225
val top1 =0.890212 top5=0.980159 val loss=0.417256 time = 260.857275
done.
seg8
val top1 =0.882804 top5=0.980423 val loss=0.431342 time = 331.013417
val top1 =0.882804 top5=0.980423 val loss=0.431342 time = 139.658914
val top1 =0.882804 top5=0.980423 val loss=0.431342 time = 138.303504
done.
seg4
val top1 =0.877249 top5=0.975661 val loss=0.455060 time = 236.635324
val top1 =0.877249 top5=0.975661 val loss=0.455060 time = 68.848208
val top1 =0.877249 top5=0.975661 val loss=0.455060 time = 67.900019
done.




k=4

opt.fusion_method :  bgs
val top1 =0.914595 top5=0.990270 val loss=0.315720 time = 219.335953
opt.fusion_method :  fgs
val top1 =0.582973 top5=0.827027 val loss=1.727975 time = 66.478450
opt.fusion_method :  avg
val top1 =0.876486 top5=0.975405 val loss=0.456596 time = 66.391539
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
from ucf101_bgs.classification import UCF101_Bgs 
from DualBlock import DualBlock,get_dualnet

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
        self.fusion_method = 'avg'# avg max bgs fgs
        self.bgsmodel = 'resnet34_v1b_k400_ucf101'
        self.fgsmodel = 'resnet34_v1b_k400_ucf101'
        self.save_dir = 'logs/test3'
        self.bgs_path = 'logs/param_rgb_resnet34_v1b_k400_ucf101_seg8'#param_rgb_resnet18_v1b_k400_ucf101_1'
        self.fgs_path = 'logs/param_cvMOG2_resnet34_v1b_k400_ucf101_seg8'
        self.bgs_params = os.path.join(self.bgs_path,'0.9212-ucf101-resnet34_v1b_k400_ucf101-010-best.params')
        self.fgs_params = os.path.join(self.fgs_path,'0.6178-ucf101-resnet34_v1b_k400_ucf101-054-best.params')
        self.num_classes = 101
        self.new_length_diff = self.new_length +1 
        self.root_bgs = os.path.expanduser('~/.mxnet/datasets/ucf101/rawframes')
        self.root_fgs = os.path.expanduser('/media/hp/mypan/BGSDecom/cv_MOG2/fgs')#'~/.mxnet/datasets/ucf101/rawframes')
        self.train_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
        self.val_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
        self.logging_file = 'train.log'
        self.name_pattern='img_%05d.jpg'
        self.dataset = 'ucf101'
        self.input_size=224#112#204#112
        self.new_height=256#128#256#128
        self.new_width=340#171#340#171
        self.input_channel=3 
        self.num_segments=4
        self.num_workers = 2
        self.num_gpus = 1
        self.per_device_batch_size = 100
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
        self.reshape_type = 'tsn' # c3d tsn tsn_newlength
      

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
net_bgs = myget(name=opt.bgsmodel, nclass=opt.num_classes, num_segments=opt.num_segments,input_channel=opt.input_channel,batch_normal=opt.partial_bn)
net_fgs = myget(name=opt.fgsmodel, nclass=opt.num_classes, num_segments=opt.num_segments,input_channel=opt.input_channel,batch_normal=opt.partial_bn)
net_bgs.cast(opt.dtype)
net_bgs.collect_params().reset_ctx(ctx)
net_fgs.cast(opt.dtype)
net_fgs.collect_params().reset_ctx(ctx)
#logger.info(net)
if opt.bgs_params is not '':
    net_bgs.load_parameters(opt.bgs_params, ctx=ctx)
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

val_dataset = UCF101_Bgs(train=False, num_segments=opt.num_segments, transform=transform_test,
                                             root_bgs=opt.root_bgs,
                                             root_fgs=opt.root_fgs,
                                             setting=opt.val_setting,
                                             name_pattern=opt.name_pattern,
                         new_width=opt.new_width,new_height=opt.new_height, new_length=opt.new_length,
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
            X_bgs = X_bgs.reshape((-1,) + X_bgs.shape[2:])
            X_fgs = X_fgs.reshape((-1,) + X_fgs.shape[2:])
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

# tic = time.time()
# acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
# logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
#         (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))

# tic = time.time()
# acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
# logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
#         (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))
       
# tic = time.time()
# acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
# logger.info('val top1 =%f top5=%f val loss=%f time = %f ' %
#         (acc_top1_val, acc_top5_val, loss_val,   time.time() - tic)) #np.mean(perclip_time)/opt.per_device_batch_size ))        



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



print('done.')