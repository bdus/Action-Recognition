#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:19:32 2019

@author: bdus

https://gluon-cv.mxnet.io/build/examples_action_recognition/dive_deep_ucf101.html#start-training-now

"""
from __future__ import division

import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data.transforms import video
#from gluoncv.data import ucf101
from gluoncv.data import UCF101
#from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory
from mxnet.contrib import amp

from model_zoo import get_model as myget

class config:
    def __init__(self):
        self.model = 'inceptionv3_ucf101_lstm'
        self.num_classes = 101
        self.train_dir = '/media/hp/mypan/BGSDecom/cv_MOG2/fgs'#'/media/hp/data/mxnet/rawframes'
        self.train_setting = '/media/hp/data/mxnet/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
        self.val_setting = '/media/hp/data/mxnet/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
        self.save_dir = 'logs/para_incv3_ucf101_lstm_seg7ch15'
        self.logging_file = 'train.log'
        self.name_pattern='img_%05d.jpg'
        self.dataset = 'ucf101'
        self.input_size=299
        self.new_height=340
        self.new_width=450
        self.input_channel=15
        self.new_length = 5
        self.new_length_diff = self.new_length
        self.num_layers=3
        self.num_segments=7
        self.num_workers = 1
        self.num_gpus = 1
        self.per_device_batch_size = 10
        self.batch_size = self.per_device_batch_size * self.num_gpus
        self.lr = 0.001
        self.warmup_lr = 0
        self.warmup_epochs = 0
        self.momentum = 0.9
        self.wd = 0.0005
        self.lr_decay = 0.1 
        self.prefetch_ratio = 0
        self.use_amp = False
        self.epochs = 2
        self.lr_decay_epoch = '30,60,80'
        self.dtype = 'float32'
        self.use_pretrained = False
        self.partial_bn = True
        self.clip_grad = 40
        self.log_interval = 10
        self.scale_ratios = [1.0, 0.875, 0.75, 0.66]
        self.save_frequency = 5
        self.mode = None#'hybrid'
        self.lr_mode ='step'
        self.accumulate = 1.0

opt = config()
makedirs(opt.save_dir)
# number of GPUs to use
#ctx = [mx.gpu(i) for i in range(opt.num_gpus)]
ctx = [mx.gpu(1)]
#ctx = [mx.cpu()]
# Get the model 
net = myget(name=opt.model, nclass=opt.num_classes, num_segments=opt.num_segments,input_channel=opt.input_channel,partial_bn=opt.partial_bn)
net.cast(opt.dtype)
net.collect_params().reset_ctx(ctx)
#print(net)

if opt.mode == 'hybrid':
    net.hybridize(static_alloc=True, static_shape=True)
    
transform_train = video.VideoGroupTrainTransform(size=(opt.input_size, opt.input_size), scale_ratios=opt.scale_ratios, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = video.VideoGroupValTransform(size=opt.input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Calculate effective total batch size

#
train_dataset = UCF101(setting=opt.train_setting, root=opt.train_dir, train=True,
                               new_width=opt.new_width, new_height=opt.new_height, new_length= (opt.new_length if opt.new_length==opt.new_length_diff else opt.new_length_diff),
                               target_width=opt.input_size, target_height=opt.input_size,
                               num_segments=opt.num_segments, transform=transform_train)

val_dataset = UCF101(setting=opt.val_setting, root=opt.train_dir, train=False,
                             new_width=opt.new_width, new_height=opt.new_height, new_length=(opt.new_length if opt.new_length==opt.new_length_diff else opt.new_length_diff),
                             target_width=opt.input_size, target_height=opt.input_size,
                             num_segments=opt.num_segments, transform=transform_test)

train_data = gluon.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                                           prefetch=int(opt.prefetch_ratio * opt.num_workers), last_batch='rollover')

val_data = gluon.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                         prefetch=int(opt.prefetch_ratio * opt.num_workers), last_batch='discard')
print('Load %d training samples.' % len(train_dataset))

    

optimizer = 'sgd'
# Set parameters
optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

#num_batches = len(train_data)
#lr_scheduler = LRSequential([
#    LRScheduler('linear', base_lr=opt.warmup_lr, target_lr=opt.lr,
#                nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
#    LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
#                nepochs=opt.epochs - opt.warmup_epochs,
#                iters_per_epoch=num_batches,
#                step_epoch=opt.lr_decay_epoch,
#                step_factor=opt.lr_decay, power=2)
#])
#optimizer_params['lr_scheduler'] = lr_scheduler

if opt.partial_bn:
    train_patterns = None
    if 'inceptionv3' in opt.model:
        train_patterns = '.*weight|.*bias|inception30_batchnorm0_gamma|inception30_batchnorm0_beta|inception30_batchnorm0_running_mean|inception30_batchnorm0_running_var'
    else:
        print('Current model does not support partial batch normalization.')

if opt.use_amp:
    amp.init()
# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, optimizer_params,update_on_kvstore=False)

if opt.use_amp:
    amp.init_trainer(trainer)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

train_history = TrainingHistory(['training-acc','val-top1-acc','val-top5-acc'])

epochs = opt.epochs
batch_size = opt.per_device_batch_size * opt.num_gpus
lr_decay_count = 0

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

def test(ctx,val_data):
    acc_top1.reset()
    acc_top5.reset()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    num_test_iter = len(val_data)
    val_loss_epoch = 0
    for i, batch in enumerate(val_data):
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        val_outputs = []
        for _, X in enumerate(data):
#            X = X.reshape((-1,) + X.shape[2:])
#            X = X.reshape((-1,15)+X.shape[-2:])
            X = X.reshape((-3,-3,-2))
            pred = net(X)
            val_outputs.append(pred)
            
        loss = [L(yhat, y) for yhat, y in zip(val_outputs, label)]
        
        acc_top1.update(label, val_outputs)
        acc_top5.update(label, val_outputs)
        
        val_loss_epoch += sum([l.mean().asscalar() for l in loss]) / len(loss)
    
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    val_loss = val_loss_epoch / num_test_iter
    
    return (top1, top5, val_loss)

for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    train_metric.reset()
    train_loss_epoch = 0
    train_loss_iter = 0
    best_val_score = 0
    num_train_iter = len(train_data)

    # Learning rate decay
    if epoch == opt.lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*opt.lr_decay)
        lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        # AutoGrad
        with ag.record():
            output = []
            for _, X in enumerate(data):
#                X = X.reshape((-1,) + X.shape[2:])
#                X = X.reshape((-1,15)+X.shape[-2:])
                X = X.reshape((-3,-3,-2))                
                pred = net(X)
                output.append(pred)
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
        # Backpropagation
        for l in loss:
            l.backward()
#        if opt.use_amp:
#            with amp.scale_loss(loss, trainer) as scaled_loss:
#                ag.backward(scaled_loss)
#        else:
#            ag.backward(loss)
#        # Optimize
#        trainer.step(batch_size)
#        if opt.accumulate > 1 and (i + 1) % opt.accumulate == 0:
#            trainer.step(batch_size * opt.accumulate)
#            net.collect_params().zero_grad()
#        else:
#            trainer.step(batch_size)
        trainer.step(batch_size)


        # Update metrics
#        train_loss += sum([l.mean().asscalar() for l in loss])
        
        train_loss_iter = sum([l.mean().asscalar() for l in loss]) / len(loss)
        train_loss_epoch += train_loss_iter
        train_metric.update(label, output)
        
        train_metric_name, train_metric_score = train_metric.get()
        if opt.log_interval and not (i+1) % opt.log_interval:
            print('Epoch[%03d] Batch [%04d]/[%04d]\tSpeed: %f samples/sec\t %s=%f\t loss=%f\t lr=%f' % (
                        epoch, i, num_train_iter, opt.batch_size*opt.log_interval/(time.time()-btic),
                        train_metric_name, train_metric_score*100, train_loss_epoch/(i+1), trainer.learning_rate))
            btic = time.time()
        
    name, acc = train_metric.get()
    # test
    acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)

    # Update history and print metrics
    train_history.update([acc,acc_top1_val,acc_top5_val])
    print('[Epoch %d] train=%f loss=%f time: %f' %
        (epoch, acc, train_loss_epoch / (i+1), time.time()-tic))
    print('[Epoch %d] val top1 =%f top5=%f val loss=%f' %
        (epoch, acc_top1_val, acc_top5_val, loss_val ))
        
    if acc_top1_val > best_val_score:
        best_val_score = acc_top1_val
        net.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, opt.model, epoch))
        trainer.save_states('%s/%.4f-%s-%s-%03d-best.states'%(opt.save_dir, best_val_score, opt.dataset, opt.model, epoch))
    else:
        if opt.save_frequency and opt.save_dir and (epoch + 1) % opt.save_frequency == 0:
            net.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, opt.model, epoch))
            trainer.save_states('%s/%s-%s-%03d.states'%(opt.save_dir, opt.dataset, opt.model, epoch))

# We can plot the metric scores with:
train_history.plot(save_path=os.path.join(opt.save_dir,'train-acc.jpg'))
# save the last model
net.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, opt.model, opt.epochs-1))
trainer.save_states('%s/%s-%s-%03d.states'%(opt.save_dir, opt.dataset, opt.model, opt.epochs-1))