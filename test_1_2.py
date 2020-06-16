#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-2-4 19:58:30

@author: bdus

https://gluon-cv.mxnet.io/build/examples_action_recognition/dive_deep_ucf101.html#start-training-now

第一个实验测试 比较

10 batch size 1 GPU

resnet18_v1b_k400_ucf101 param_cv_MOG2_resnet18_v1b_ucf101_seg8 0.6122-ucf101-resnet18_v1b_k400_ucf101-098-best
bs=10
seg=1
val top1 =0.398625 top5=0.680412 val loss=2.768809 time = 94.522674
val top1 =0.398625 top5=0.680412 val loss=2.768809 time = 18.661016
val top1 =0.398625 top5=0.680412 val loss=2.768809 time = 18.652699
time per clip : 0.492159
done.
seg=32
val top1 =0.648163 top5=0.870738 val loss=1.480092 time = 873.981898 
val top1 =0.648163 top5=0.870738 val loss=1.480092 time = 563.493895 
val top1 =0.648163 top5=0.870738 val loss=1.480092 time = 536.439693 
time per clip : 14.154086 
seg=16
val top1 =0.629923 top5=0.866772 val loss=1.506231 time = 573.770005
val top1 =0.629923 top5=0.866772 val loss=1.506231 time = 260.322968
val top1 =0.629923 top5=0.866772 val loss=1.506231 time = 253.920753
time per clip : 6.699763
seg=8
val top1 =0.612213 top5=0.852762 val loss=1.559015 time = 304.349115
val top1 =0.612213 top5=0.852762 val loss=1.559015 time = 133.359614
val top1 =0.612213 top5=0.852762 val loss=1.559015 time = 134.404725
time per clip : 3.546304
seg=4
val top1 =0.584192 top5=0.834523 val loss=1.657429 time = 168.806255
val top1 =0.584192 top5=0.834523 val loss=1.657429 time = 64.435573
val top1 =0.584192 top5=0.834523 val loss=1.657429 time = 64.637182
time per clip : 1.705470
seg=3
val top1 =0.537933 top5=0.804124 val loss=1.806468 time = 121.188954
val top1 =0.537933 top5=0.804124 val loss=1.806468 time = 49.608016
val top1 =0.537933 top5=0.804124 val loss=1.806468 time = 48.632724
time per clip : 1.283189
seg=2
val top1 =0.494845 top5=0.770817 val loss=2.017980 time = 83.751986
val top1 =0.494845 top5=0.770817 val loss=2.017980 time = 36.578550
val top1 =0.494845 top5=0.770817 val loss=2.017980 time = 36.923380
time per clip : 0.974236


bs=100
seg=1
val top1 =0.398625 top5=0.680412 val loss=2.769875 time = 15.398483
val top1 =0.398625 top5=0.680412 val loss=2.769875 time = 15.245614
val top1 =0.398625 top5=0.680412 val loss=2.769875 time = 15.332225
time per clip : 40.348468
seg=2
val top1 =0.494845 top5=0.770817 val loss=2.016981 time = 108.761697
val top1 =0.494845 top5=0.770817 val loss=2.016981 time = 29.226839
val top1 =0.494845 top5=0.770817 val loss=2.016981 time = 29.459365
time per clip : 77.525035



resnet34_v1b_k400_ucf101 param_cvMOG2_resnet34_v1b_k400_ucf101_seg8  0.6178-ucf101-resnet34_v1b_k400_ucf101-054-best
bs=100
val top1 =0.432725 top5=0.711340 val loss=3.164477 time = 15.900223
val top1 =0.432725 top5=0.711340 val loss=3.164477 time = 15.960358
val top1 =0.432725 top5=0.711340 val loss=3.164477 time = 15.699095
time per clip : 41.313779

seg=3
val top1 =0.561195 top5=0.815755 val loss=1.875545 time = 49.073775
val top1 =0.561195 top5=0.815755 val loss=1.875545 time = 49.320150
val top1 =0.561195 top5=0.815755 val loss=1.875545 time = 49.302299
time per clip : 1.300856

resnet50_v1b_k400_ucf101 param_cvMOG2_resnet50_v1b_k400_ucf101_seg8_1 0.4647-ucf101-resnet50_v1b_k400_ucf101-062-best 
val top1 =0.313772 top5=0.619350 val loss=3.207081 time = 15.706569
val top1 =0.313772 top5=0.619350 val loss=3.207081 time = 15.556771
val top1 =0.313772 top5=0.619350 val loss=3.207081 time = 15.677877
time per clip : 41.257968

seg=3
val top1 =0.419244 top5=0.723500 val loss=2.278135 time = 49.922375
val top1 =0.419244 top5=0.723500 val loss=2.278135 time = 50.316654
val top1 =0.419244 top5=0.723500 val loss=2.278135 time = 50.284872
time per clip : 1.326782

resnet34_v1b_k400_ucf101 param_cvMOG2_resnet34_v1b_k400_ucf101_seg8_1 0.8097-ucf101-resnet34_v1b_k400_ucf101-049-best
new_length == 5 
reshape_type tsn_newlength


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
        self.new_length = 5
        self.model = 'resnet34_v1b_k400_ucf101'
        self.save_dir = 'logs/test2'
        self.model_file = 'logs/param_cvMOG2_resnet34_v1b_k400_ucf101_seg8_1'
        self.resume_params = os.path.join(self.model_file,'0.8097-ucf101-resnet34_v1b_k400_ucf101-049-best.params')
        self.num_classes = 101
        self.new_length_diff = self.new_length +1 
        self.train_dir = os.path.expanduser('/media/hp/mypan/BGSDecom/cv_MOG2/fgs')#
        self.train_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
        self.val_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
        self.logging_file = 'train.log'
        self.name_pattern='img_%05d.jpg'
        self.dataset = 'ucf101'
        self.input_size=224#224#112#204#112
        self.new_height=256#256#128#256#128
        self.new_width=340#340#171#340#171
        self.input_channel=15
        self.num_segments=3
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
        self.reshape_type = 'tsn_newlength' # c3d tsn tsn_newlength
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