#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020年3月6日18:52:46

@author: bdus

(5,3,16,112,112)

"""
from __future__ import division

import argparse, time, logging, os, sys, math
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
os.environ['CUDA_VISIBLE_DEVICES']='0' #0,1

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
        self.new_length = 16
        self.new_step = 2
        self.model = 'r2plus1d_resnet18_kinetics400_custom'
        self.use_kinetics_pretrain = False#True
        self.TranConv_model = 'r2plus1d_resnet34_tranconv_lateral'
        self.use_lateral=True
        self.freeze_lateral=False #True
        self.save_dir = 'logs/param_rgb_r2plus1d_resnet18_kinetics400_custom_hmdb51_nlength16_lateral_scratch_lr0_1'
        self.num_classes = 51#101
        self.new_length_diff = self.new_length +1
        self.dataset = 'hmdb51'#'ucf101'
        self.train_dir = os.path.expanduser('~/.mxnet/datasets/ucf101/rawframes')#'/media/hp/mypan/BGSDecom/cv_MOG2/fgs')#        
        self.train_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
        self.val_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
        self.train_dir_hmdb51 = os.path.expanduser('~/.mxnet/datasets/hmdb51/rawframes')
        self.train_setting_hmdb51 = '/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_train_split_1_rawframes.txt'
        self.val_setting_hmdb51 = '/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt'
        self.logging_file = 'train.log'
        self.name_pattern='img_%05d.jpg'        
        self.input_size=112#204#112
        self.new_height=128#256#128
        self.new_width=171#340#171
        self.input_channel=3 
        self.num_segments= 1
        self.num_workers =1
        self.num_gpus = 1
        self.per_device_batch_size = 10
        self.lr = 0.1
        self.lr_decay = 0.1
        self.warmup_lr = 0.001
        self.warmup_epochs = 8
        self.momentum = 0.9
        self.wd = 0.0001      
        self.prefetch_ratio = 1.0
        self.use_amp = False
        self.epochs = 100
        self.lr_decay_epoch = [30,60,80]
        self.lr_decay_period = 0
        self.scale_ratios = [1.0, 0.8]#[1.0, 0.875, 0.75, 0.66]
        self.dtype = 'float32'
        self.pretrained_lateral_path = '/home/hp/lcx/Action-Recognition/logs/param_rgb_r2plus1d_resnet18_kinetics400_custom_ucf101_nlength16_lateral_1'
        self.pretrained_lateral_file = '0.9315-ucf101-r2plus1d_resnet34_tranconv_lateral-079-best.params'
        self.use_pretrained = True #False
        self.partial_bn = False
        self.train_patterns = 'r2plus1d0_dense'#'r2plus1d1_dense'
        self.use_train_patterns = False#True
        self.freeze_patterns = ''
        self.freeze_lr_mult = 0.01 #set freezed base layer lr = self.lr * self.freeze_lr_mult
        self.use_mult = False
        self.clip_grad = 40
        self.log_interval = 10
        self.lr_mode = 'cosine'        
        self.resume_epoch = 0 #32
        self.resume_path = 'logs/param_rgb_r2plus1d_resnet18_kinetics400_custom_hmdb51_nlength16_lateral_scratch'
        self.resume_params = os.path.join(self.resume_path,'0.2163-hmdb51-r2plus1d_resnet18_kinetics400_custom-066-best.params')
        self.resume_states = os.path.join(self.resume_path,'0.2163-hmdb51-r2plus1d_resnet18_kinetics400_custom-066-best.states')
        self.reshape_type = 'tsn' #mxc3d c3d tsn tsn_newlength
      

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
net = myget(name=opt.model, nclass=opt.num_classes, num_segments=opt.num_segments,input_channel=opt.input_channel,batch_normal=opt.partial_bn,use_lateral=opt.use_lateral,use_kinetics_pretrain=opt.use_kinetics_pretrain)
net.cast(opt.dtype)
net.collect_params().reset_ctx(ctx)

net1 = myget(name=opt.TranConv_model)
net1.cast(opt.dtype)
net1.collect_params().reset_ctx(ctx)

#logger.info(net)
if opt.resume_params is not '':
    net.load_parameters(opt.resume_params, ctx=ctx)

#if opt.use_pretrained:
    #net.features_3d.load_parameters(opt.pretrained_ECOfeature3d,ctx=ctx,allow_missing=True)
    #net.output.load_parameters(opt.pretrained_ECOoutput,ctx=ctx,allow_missing=True)
    #logger.info('use pretrained model : %s , %s',opt.pretrained_ECOfeature3d,opt.pretrained_ECOoutput)
if opt.use_pretrained:
    modelpath = os.path.join(opt.pretrained_lateral_path,opt.pretrained_lateral_file)
    modelfile = os.path.expanduser(modelpath)
    net1.load_parameters(modelfile,ctx=ctx,allow_missing=True)
    logger.info('use pretrained model : %s',modelfile)
    
if opt.use_mult:
    net.collect_params(opt.freeze_patterns).setattr('lr_mult',opt.freeze_lr_mult)

logger.info(net)
net.collect_params().reset_ctx(ctx)
    
transform_train = video.VideoGroupTrainTransform(size=(opt.input_size, opt.input_size), scale_ratios=opt.scale_ratios, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    train_dataset = UCF101(setting=opt.train_setting, root=opt.train_dir, train=True,
                       new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,new_step=opt.new_step,
                       target_width=opt.input_size, target_height=opt.input_size,
                       num_segments=opt.num_segments, transform=transform_train)
    val_dataset = UCF101(setting=opt.val_setting, root=opt.train_dir, train=False,
                     new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,new_step=opt.new_step,
                     target_width=opt.input_size, target_height=opt.input_size,
                     num_segments=opt.num_segments, transform=transform_test)
elif opt.dataset == 'hmdb51':
    train_dataset = HMDB51(setting=opt.train_setting_hmdb51, root=opt.train_dir_hmdb51, train=True,
           new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
           target_width=opt.input_size, target_height=opt.input_size, 
           num_segments=opt.num_segments, transform=transform_train)
    val_dataset = HMDB51(setting=opt.val_setting_hmdb51, root=opt.train_dir_hmdb51, train=False,
           new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
           target_width=opt.input_size, target_height=opt.input_size, 
           num_segments=opt.num_segments, transform=transform_test)
else:
    logger.info('Dataset %s is not supported yet.' % (opt.dataset))



train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   prefetch=int(opt.prefetch_ratio * num_workers), last_batch='rollover')
val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 prefetch=int(opt.prefetch_ratio * num_workers), last_batch='discard')


train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
logger.info('Load %d training samples.' % len(train_dataset))
val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    
def batch_fn(batch, ctx):
    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label
# Learning rate decay factor
lr_decay = opt.lr_decay
# Epochs where learning rate decays
#lr_decay_epoch = opt.lr_decay_epoch
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = opt.lr_decay_epoch
lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]

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
elif opt.partial_bn == False and opt.use_train_patterns == True:
    logger.info('========\n %s' % net.collect_params() )
    trainer = gluon.Trainer(net.collect_params(opt.train_patterns), optimizer, optimizer_params, update_on_kvstore=False)
    logger.info('trainner.patterns: %s.' % opt.train_patterns )
    logger.info('========\n %s' % net.collect_params(opt.train_patterns) )
elif opt.use_lateral and not opt.freeze_lateral:
    print("============== use_lateral")
    lst = list(net.collect_params().values()) + list(net1.collect_params().values())
    trainer = gluon.Trainer(lst, optimizer, optimizer_params, update_on_kvstore=False)
else:
    print("============== training net0. net1 is not included in training")
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, update_on_kvstore=False)

if opt.resume_states is not '':
    trainer.load_states(opt.resume_states)

# Define our trainer for net
#trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
loss_l2 = gluon.loss.L2Loss(weight=1.0)
loss_l2.initialize()


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
    val_loss_epoch1 = 0
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)

        val_outputs = []
        if opt.use_lateral:
            l2out = []
        for _, X in enumerate(data):
            if opt.reshape_type == 'tsn':
                X = X.reshape((-1,) + X.shape[2:])
            elif opt.reshape_type == 'c3d':
                X = nd.transpose(data=X,axes=(0,2,1,3,4))
            elif opt.new_length != 1 and opt.reshape_type == 'tsn_newlength':
                X = X.reshape((-3,-3,-2))
            else:
                pass
            if not opt.use_lateral:
                pred = net(X.astype(opt.dtype, copy=False))               
            else:
                pred, latel= net(X.astype(opt.dtype, copy=False))
                x_hat = net1(latel[0].astype(opt.dtype, copy=False),
                             latel[1].astype(opt.dtype, copy=False),latel[2].astype(opt.dtype, copy=False))
                l2out.append(x_hat) 
            #pred = net(X.astype(opt.dtype, copy=False))
            val_outputs.append(pred)
        
        if not opt.use_lateral:
            loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(val_outputs, label)]
        else:
            loss0 = [loss_fn(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(val_outputs, label)]
            loss1 = [loss_l2(xhat,x.astype(opt.dtype, copy=False)) for xhat, x in zip(l2out,data ) ]
            loss = [(l1+l2) for l1, l2 in zip(loss0,loss1)]
        #loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(val_outputs, label)]
        
        acc_top1.update(label, val_outputs)
        acc_top5.update(label, val_outputs)
        
        val_loss_epoch += sum([l.mean().asscalar() for l in loss]) / len(loss)
        val_loss_epoch1 += sum([l.mean().asscalar() for l in loss1]) / len(loss1)
    
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    val_loss = val_loss_epoch / num_test_iter
    loss_mse = val_loss_epoch1 / num_test_iter
    
    return (top1, top5, val_loss, loss_mse)

# training 

for epoch in range(opt.resume_epoch, opt.epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0
    mse_loss = 0
    btic = time.time()

    # Learning rate decay 不要修改这个 LRScheduler已经被定义了
    # if epoch == lr_decay_epoch[lr_decay_count]:
    #     trainer.set_learning_rate(trainer.learning_rate*opt.lr_decay)
    #     lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data, label = batch_fn(batch, ctx)
        # AutoGrad
        with ag.record():
            output = []
            l2out = []
            for _, X in enumerate(data):
                if opt.reshape_type == 'tsn':
                    X = X.reshape((-1,) + X.shape[2:])
                elif opt.reshape_type == 'c3d':                    
                    X = nd.transpose(data=X,axes=(0,2,1,3,4))
                elif opt.new_length != 1 and opt.reshape_type == 'tsn_newlength':
                    X = X.reshape((-3,-3,-2))
                else:
                    pass
                #pred = net(X.astype(opt.dtype, copy=False))
                if not opt.use_lateral:
                    pred = net(X.astype(opt.dtype, copy=False))                    
                else:
                    pred, latel= net(X.astype(opt.dtype, copy=False))
                    x_hat = net1(latel[0].astype(opt.dtype, copy=False),
                             latel[1].astype(opt.dtype, copy=False),latel[2].astype(opt.dtype, copy=False))
                    l2out.append(x_hat) 
                output.append(pred)
            #loss = [loss_fn(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(output, label)]
            if not opt.use_lateral:
                loss = [loss_fn(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(output, label)]
            else:
                #print(l2out)
                loss0 = [loss_fn(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(output, label)]
                loss1 = [loss_l2(xhat,x.astype(opt.dtype, copy=False)) for xhat, x in zip(l2out,data )]
                loss = [(l1+l2) for l1, l2 in zip(loss0,loss1)]

        # Backpropagation
        if not opt.use_lateral:
            for l in loss:
                l.backward()
        else:
            for l in loss:
                l.backward()

        # Optimize
        trainer.step(batch_size,ignore_stale_grad=True)        

        # Update metrics
        train_loss += sum([l.mean().asscalar() for l in loss])
        mse_loss  += sum([l.mean().asscalar() for l in loss1]) / len(loss1)
        train_metric.update(label, output)
        if i % opt.log_interval == 0:
            name, acc = train_metric.get()
            logger.info('[Epoch %d] [%d | %d] train=%f loss=%f mseloss=%f time: %f' %
                  (epoch,i,len(train_data), acc, train_loss / (i+1),mse_loss/(i+1), time.time()-btic) )
            btic = time.time()

    name, acc = train_metric.get()
    
    # test
    #acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)
    acc_top1_val, acc_top5_val, loss_val, loss_mse = test(ctx, val_data) 

    # Update history and print metrics
    train_history.update([acc,acc_top1_val,acc_top5_val])
    train_history.plot(save_path=os.path.join(opt.save_dir,'trainlog.jpg'))
    logger.info('[Epoch %d] train=%f loss=%f time: %f' %
        (epoch, acc, train_loss / (i+1), time.time()-tic))
    #logger.info('[Epoch %d] val top1 =%f top5=%f val loss=%f,lr=%f' %
     #   (epoch, acc_top1_val, acc_top5_val, loss_val ,trainer.learning_rate ))    
    logger.info('[Epoch %d] val top1 =%f top5=%f val loss=%f,mesloss=%f,lr=%f' %
        (epoch, acc_top1_val, acc_top5_val, loss_val ,loss_mse,trainer.learning_rate ))
    if acc_top1_val > best_val_score and epoch > 5:
        best_val_score = acc_top1_val
        net.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, opt.model, epoch))
        trainer.save_states('%s/%.4f-%s-%s-%03d-best.states'%(opt.save_dir, best_val_score, opt.dataset, opt.model, epoch))
        if opt.use_lateral:
            net1.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, opt.TranConv_model, epoch))


# We can plot the metric scores with:
train_history.plot(save_path=os.path.join(opt.save_dir,'trainlog_final.jpg'))
