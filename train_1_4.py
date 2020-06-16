#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-2-6 14:12:37

@author: bdus

https://gluon-cv.mxnet.io/build/examples_action_recognition/dive_deep_ucf101.html#start-training-now

训练两个模型用
分别导入两个模型和参数文件，然后max mean

不直接合并 concat后fc的

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
#from DualBlock_fc import DualBlock,get_dualnet
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
        self.fusion_method = 'avg'
        self.bgsmodel = 'resnet18_v1b_k400_ucf101'
        self.fgsmodel = 'eco_resnet18_v1b_k400_ucf101'
        self.model = self.bgsmodel,self.fgsmodel
        self.save_dir = 'logs/bfgs_RGBcvMOG2_ucf101_2'
        self.bgs_path = 'logs/param_rgb_resnet18_v1b_k400_ucf101_1'
        self.fgs_path = 'logs/param_cvMOG2_eco_resnet18_v1b_k400_ucf101_2'
        self.bgs_params = os.path.join(self.bgs_path,'0.8258-ucf101-resnet18_v1b_k400_ucf101-048-best.params')
        self.fgs_params = os.path.join(self.fgs_path,'0.5919-ucf101-eco_resnet18_v1b_k400_ucf101-062-best.params')
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
        self.num_segments=8
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
        self.epochs = 85
        self.lr_decay_epoch = [30,60,80]
        self.dtype = 'float32'
        self.use_pretrained = False
        self.partial_bn = False
        self.clip_grad = 40
        self.log_interval = 10
        self.lr_mode = 'step'        
        self.resume_epoch = 1
        self.reshape_type = 'tsn' # c3d tsn tsn_newlength
        self.resume_states = ''
        self.resume_params = ''
      

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
net = get_dualnet(fgs_model=opt.fgsmodel,bgs_model=opt.bgsmodel,fgs_path=opt.fgs_params,bgs_path=opt.bgs_params, nclass=opt.num_classes, num_segments=opt.num_segments,input_channel=opt.input_channel,fusion_method=opt.fusion_method)
net.cast(opt.dtype)
net.collect_params().reset_ctx(ctx)
#logger.info(net)


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

train_dataset = UCF101_Bgs(train=True, num_segments=opt.num_segments, transform=transform_train,
                                             root_bgs=opt.root_bgs,
                                             root_fgs=opt.root_fgs,
                                             setting=opt.train_setting,
                                             name_pattern=opt.name_pattern,
                           new_width=opt.new_width,new_height=opt.new_height, new_length=opt.new_length,
                           target_width=opt.input_size, target_height=opt.input_size
                                             )

val_dataset = UCF101_Bgs(train=False, num_segments=opt.num_segments, transform=transform_train,
                                             root_bgs=opt.root_bgs,
                                             root_fgs=opt.root_fgs,
                                             setting=opt.val_setting,
                                             name_pattern=opt.name_pattern,
                         new_width=opt.new_width,new_height=opt.new_height, new_length=opt.new_length,
                           target_width=opt.input_size, target_height=opt.input_size
                                             )

train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   prefetch=int(opt.prefetch_ratio * num_workers), last_batch='rollover')
val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 prefetch=int(opt.prefetch_ratio * num_workers), last_batch='discard')
logger.info('Load %d training samples.' % len(train_dataset))
#def batch_fn(batch, ctx):
#    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
#    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
#    return data, label


# Learning rate decay factor
lr_decay = opt.lr_decay
# Epochs where learning rate decays
lr_decay_epoch = opt.lr_decay_epoch

# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

num_batches = len(train_data)
lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=opt.warmup_lr, target_lr=opt.lr,
                nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
    LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                nepochs=opt.epochs - opt.warmup_epochs,
                iters_per_epoch=num_batches,
                step_epoch=lr_decay_epoch,
                step_factor=lr_decay, power=2)
])
optimizer_params['lr_scheduler'] = lr_scheduler

if opt.partial_bn:
    train_patterns = None
    if 'inceptionv3' in opt.model:
        train_patterns = '.*weight|.*bias|inception30_batchnorm0_gamma|inception30_batchnorm0_beta|inception30_batchnorm0_running_mean|inception30_batchnorm0_running_var'
    else:
        logger.info('Current model does not support partial batch normalization.')

    trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, optimizer_params, update_on_kvstore=False)
else:
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, update_on_kvstore=False)

if opt.resume_states is not '':
    trainer.load_states(opt.resume_states)


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
            pred = net(X_bgs,X_fgs)
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

for epoch in range(opt.resume_epoch, opt.epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0
    btic = time.time()

    # Learning rate decay
#    if epoch == lr_decay_epoch[lr_decay_count]:
#        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
#        lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data_bgs = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        data_fgs = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        # AutoGrad
        with ag.record():
            output = []
            for _, (X_bgs,X_fgs) in enumerate(zip(data_bgs,data_fgs)):
                if opt.reshape_type == 'tsn':
                    X_bgs = X_bgs.reshape((-1,) + X_bgs.shape[2:])
                    X_fgs = X_fgs.reshape((-1,) + X_fgs.shape[2:])
                elif opt.reshape_type == 'c3d' or '3d' in opt.model:
                    X_bgs = nd.transpose(data=X_bgs,axes=(0,2,1,3,4))
                    X_fgs = nd.transpose(data=X_fgs,axes=(0,2,1,3,4))
                    #X = nd.transpose(data=X,axes=(0,2,1,3,4))
                elif opt.new_length != 1 and opt.reshape_type == 'tsn_newlength':
                    #X = X.reshape((-3,-3,-2))
                    X_bgs = X_bgs.reshape((-3,-3,-2))
                    X_fgs = X_fgs.reshape((-3,-3,-2))
                else:
                    pass
                pred = net(X_bgs,X_fgs)
                output.append(pred)
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        trainer.step(batch_size,ignore_stale_grad=True)        

        # Update metrics
        train_loss += sum([l.mean().asscalar() for l in loss])
        train_metric.update(label, output)
        if i % opt.log_interval == 0:
            name, acc = train_metric.get()
            logger.info('[Epoch %d] [%d | %d] train=%f loss=%f time: %f' %
                  (epoch,i,len(train_data), acc, train_loss / (i+1), time.time()-btic) )
            btic = time.time()

    name, acc = train_metric.get()
    
    # test
    acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)

    # Update history and print metrics
    train_history.update([acc,acc_top1_val,acc_top5_val])
    logger.info('[Epoch %d] train=%f loss=%f time: %f' %
        (epoch, acc, train_loss / (i+1), time.time()-tic))
    logger.info('[Epoch %d] val top1 =%f top5=%f val loss=%f,lr=%f' %
        (epoch, acc_top1_val, acc_top5_val, loss_val ,trainer.learning_rate ))    
    if acc_top1_val > best_val_score and epoch > 5:
        best_val_score = acc_top1_val
        net.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, opt.model, epoch))
        trainer.save_states('%s/%.4f-%s-%s-%03d-best.states'%(opt.save_dir, best_val_score, opt.dataset, opt.model, epoch))            

# We can plot the metric scores with:
train_history.plot(save_path=os.path.join(opt.save_dir,'trainlog.jpg'))
