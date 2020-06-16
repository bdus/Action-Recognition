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
from mxboard import SummaryWriter
from mxnet.contrib import amp

from gluoncv.data.transforms import video
from gluoncv.data import UCF101, Kinetics400, SomethingSomethingV2, HMDB51
#from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory
from gluoncv.data.sampler import SplitSampler
from model_zoo import get_model 

class config:
    def __init__(self):
        self.new_length = 5
        self.model = 'inceptionv3_ucf101_sim'
        self.num_classes = 101
        self.new_length_diff = self.new_length  +1 
        self.train_dir = '/media/hp/data/mxnet/rawframes'
        self.train_setting = '/media/hp/data/mxnet/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
        self.val_setting = '/media/hp/data/mxnet/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
        self.save_dir = 'logs/param_FrameDiff_incep_seg7_ch15'
        self.logging_file = 'train.log'
        self.name_pattern='img_%05d.jpg'
        self.dataset = 'ucf101'
        self.input_size=299
        self.new_height=340
        self.new_width=450
        self.input_channel=15 
        self.num_segments=7
        self.num_workers = 10
        self.num_gpus = 2
        self.per_device_batch_size = 10
        self.lr = 0.001
        self.warmup_lr = 0
        self.warmup_epochs = 0
        self.momentum = 0.9
        self.wd = 0.0005
        self.lr_decay = 0.1 
        self.prefetch_ratio = 2.0
        self.use_amp = True
        self.num_epochs = 2
        self.lr_decay_epoch = '30,60,80'
        self.dtype = 'float32'
        self.use_pretrained = False
        self.partial_bn = True
        self.clip_grad = 40
        self.log_interval = 10


def get_data_loader(opt, batch_size, num_workers, logger):
    data_dir = opt.train_dir    
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    input_size = opt.input_size

    def batch_fn(batch, ctx):
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        return data, label

    transform_train = video.VideoGroupTrainTransform(size=(input_size, input_size), scale_ratios=scale_ratios, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = video.VideoGroupValTransform(size=input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if opt.dataset == 'ucf101':
        train_dataset = UCF101(setting=opt.train_setting, root=data_dir, train=True,
                               new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length_diff,
                               target_width=input_size, target_height=input_size,
                               num_segments=opt.num_segments, transform=transform_train)
        val_dataset = UCF101(setting=opt.val_setting, root=data_dir, train=False,
                             new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length_diff,
                             target_width=input_size, target_height=input_size,
                             num_segments=opt.num_segments, transform=transform_test)
    else:
#        logger.info('Dataset %s is not supported yet.' % (opt.dataset))
        print('Dataset %s is not supported yet.' % (opt.dataset))

        
    print('Load %d training samples and %d validation samples.' % (len(train_dataset), len(val_dataset)))
#    logger.info('Load %d training samples and %d validation samples.' % (len(train_dataset), len(val_dataset)))
    
    train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                       prefetch=int(opt.prefetch_ratio * num_workers), last_batch='rollover')
    val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     prefetch=int(opt.prefetch_ratio * num_workers), last_batch='discard')

    return train_data, val_data, batch_fn

#def main():

def get_diff(input_data,new_length=5):
#    print(type(input_data))
    assert input_data.shape[3] == new_length+1
    fron = input_data.slice_axis(axis=3,begin=1,end=new_length+1).copy()
    last = input_data.slice_axis(axis=3,begin=0,end=new_length)
    fron = fron-last
    return fron

def main():
    opt = config()

    makedirs(opt.save_dir)

    filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)

    sw = SummaryWriter(logdir=opt.save_dir, flush_secs=5, verbose=False)

    if opt.use_amp:
        amp.init()

    batch_size = opt.per_device_batch_size
    classes = opt.num_classes

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    logger.info('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]

    optimizer = 'sgd'

    if opt.clip_grad > 0:
        optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'clip_gradient': opt.clip_grad}
    else:
        optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

    if opt.dtype != 'float32':
        optimizer_params['multi_precision'] = True

    model_name = opt.model

    net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained,
                    use_tsn=opt.use_tsn, num_segments=opt.num_segments, partial_bn=opt.partial_bn, input_channel=opt.input_channel)
    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    logger.info(net)

    if opt.resume_params is not '':
        net.load_parameters(opt.resume_params, ctx=context)


    train_data, val_data, batch_fn = get_data_loader(opt, batch_size, num_workers, logger)

    num_batches = len(train_data)
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=opt.warmup_lr, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])
    optimizer_params['lr_scheduler'] = lr_scheduler

    train_metric = mx.metric.Accuracy()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    def test(ctx, val_data, kvstore=None):
        acc_top1.reset()
        acc_top5.reset()
        L = gluon.loss.SoftmaxCrossEntropyLoss()
        num_test_iter = len(val_data)
        val_loss_epoch = 0
        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = []
            for _, X in enumerate(data):
#                X = X.reshape((-1,) + X.shape[2:])
                X = X.reshape((-3,-3,-2))
                pred = net(X.astype(opt.dtype, copy=False))
                outputs.append(pred)

            loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]

            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

            val_loss_epoch += sum([l.mean().asscalar() for l in loss]) / len(loss)

            if opt.log_interval and not (i+1) % opt.log_interval:
                logger.info('Batch [%04d]/[%04d]: evaluated' % (i, num_test_iter))

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        val_loss = val_loss_epoch / num_test_iter
     
        return (top1, top5, val_loss)

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]

        if opt.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        if opt.partial_bn:
            train_patterns = None
            if 'inceptionv3' in opt.model:
                train_patterns = '.*weight|.*bias|inception30_batchnorm0_gamma|inception30_batchnorm0_beta|inception30_batchnorm0_running_mean|inception30_batchnorm0_running_var'
            else:
                logger.info('Current model does not support partial batch normalization.')
          
            trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, optimizer_params, update_on_kvstore=False)
        else:
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, update_on_kvstore=False)

        if opt.accumulate > 1:
            params = [p for p in net.collect_params().values() if p.grad_req != 'null']
            for p in params:
                p.grad_req = 'add'

        if opt.resume_states is not '':
            trainer.load_states(opt.resume_states)

        if opt.use_amp:
            amp.init_trainer(trainer)

        L = gluon.loss.SoftmaxCrossEntropyLoss()

        best_val_score = 0
        lr_decay_count = 0

        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()
            train_metric.reset()
            btic = time.time()
            num_train_iter = len(train_data)
            train_loss_epoch = 0
            train_loss_iter = 0

            for i, batch in enumerate(train_data):
                data, label = batch_fn(batch, ctx)

                with ag.record():
                    outputs = []
                    for _, X in enumerate(data):
#                        X = X.reshape((-1,) + X.shape[2:])
                        X = X.reshape((-3,-3,-2))
                        pred = net(X.astype(opt.dtype, copy=False))
                        outputs.append(pred)
                    loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]

                    if opt.use_amp:
                        with amp.scale_loss(loss, trainer) as scaled_loss:
                            ag.backward(scaled_loss)
                    else:
                        ag.backward(loss)

                if opt.accumulate > 1 and (i + 1) % opt.accumulate == 0:                    
                    trainer.step(batch_size * opt.accumulate)
                    net.collect_params().zero_grad()
                else:
                    trainer.step(batch_size)

                train_metric.update(label, outputs)
                train_loss_iter = sum([l.mean().asscalar() for l in loss]) / len(loss)
                train_loss_epoch += train_loss_iter

                train_metric_name, train_metric_score = train_metric.get()
                sw.add_scalar(tag='train_acc_top1_iter', value=train_metric_score*100, global_step=epoch * num_train_iter + i)
                sw.add_scalar(tag='train_loss_iter', value=train_loss_iter, global_step=epoch * num_train_iter + i)
                sw.add_scalar(tag='learning_rate_iter', value=trainer.learning_rate, global_step=epoch * num_train_iter + i)

                if opt.log_interval and not (i+1) % opt.log_interval:
                    logger.info('Epoch[%03d] Batch [%04d]/[%04d]\tSpeed: %f samples/sec\t %s=%f\t loss=%f\t lr=%f' % (
                                epoch, i, num_train_iter, batch_size*opt.log_interval/(time.time()-btic),
                                train_metric_name, train_metric_score*100, train_loss_epoch/(i+1), trainer.learning_rate))
                    btic = time.time()

            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(batch_size * i /(time.time() - tic))
            mx.ndarray.waitall()
            
            acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)

            logger.info('[Epoch %03d] training: %s=%f\t loss=%f' % (epoch, train_metric_name, train_metric_score*100, train_loss_epoch/num_train_iter))
            logger.info('[Epoch %03d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time()-tic))
            logger.info('[Epoch %03d] validation: acc-top1=%f acc-top5=%f loss=%f' % (epoch, acc_top1_val*100, acc_top5_val*100, loss_val))

            sw.add_scalar(tag='train_loss_epoch', value=train_loss_epoch/num_train_iter, global_step=epoch)
            sw.add_scalar(tag='val_loss_epoch', value=loss_val, global_step=epoch)
            sw.add_scalar(tag='val_acc_top1_epoch', value=acc_top1_val*100, global_step=epoch)

            if acc_top1_val > best_val_score:
                best_val_score = acc_top1_val
                net.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, model_name, epoch))
                trainer.save_states('%s/%.4f-%s-%s-%03d-best.states'%(opt.save_dir, best_val_score, opt.dataset, model_name, epoch))
            else:
                if opt.save_frequency and opt.save_dir and (epoch + 1) % opt.save_frequency == 0:
                    net.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, model_name, epoch))
                    trainer.save_states('%s/%s-%s-%03d.states'%(opt.save_dir, opt.dataset, model_name, epoch))

        # save the last model
        net.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, model_name, opt.num_epochs-1))
        trainer.save_states('%s/%s-%s-%03d.states'%(opt.save_dir, opt.dataset, model_name, opt.num_epochs-1))

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)

    train(context)
    sw.close()


opt = config()


batch_size = opt.per_device_batch_size
classes = opt.num_classes

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers


optimizer = 'sgd'

optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

model_name = opt.model

net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained,
                use_tsn=True, num_segments=opt.num_segments, partial_bn=opt.partial_bn, input_channel=opt.input_channel)
net.cast(opt.dtype)
net.collect_params().reset_ctx(ctx)
print(net)
net.hybridize(static_alloc=True, static_shape=True)
train_data, val_data, batch_fn = get_data_loader(opt, batch_size, num_workers, logger=None)
    

# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
if opt.clip_grad > 0:
    optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'clip_gradient': opt.clip_grad}
else:
    optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True
        
num_batches = len(train_data)
lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=opt.warmup_lr, target_lr=opt.lr,
                nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
    LRScheduler('step', base_lr=opt.lr, target_lr=0,
                nepochs=opt.num_epochs - opt.warmup_epochs,
                iters_per_epoch=num_batches,
                step_epoch=opt.lr_decay_epoch,
                step_factor=opt.lr_decay, power=2)
])
optimizer_params['lr_scheduler'] = lr_scheduler
        
# Define our trainer for net
if opt.partial_bn:
    train_patterns = None
    if 'inceptionv3' in opt.model:
        train_patterns = '.*weight|.*bias|inception30_batchnorm0_gamma|inception30_batchnorm0_beta|inception30_batchnorm0_running_mean|inception30_batchnorm0_running_var'
    else:
        logger.info('Current model does not support partial batch normalization.')
  
    trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, optimizer_params)
else:
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

if opt.use_amp:
    amp.init()
        
if opt.use_amp:
    amp.init_trainer(trainer)
    
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

train_history = TrainingHistory(['training-acc','val-top1-acc','val-top5-acc'])

epochs = opt.num_epochs
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
            X = get_diff(input_data=X,new_length=opt.new_length)
#            X = X.reshape((-1,) + X.shape[2:])
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
    train_metric.reset()
    btic = time.time()            
#    train_loss = 0
    num_train_iter = len(train_data)
    train_loss_epoch = 0
    train_loss_iter = 0
    
    # Learning rate decay
    if epoch == opt.lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        

        # AutoGrad
        with ag.record():
            output = []
            for X in data:
#                X = X.reshape((-1,) + X.shape[2:])
#                X = X.reshape((-1,15)+X.shape[-2:])
                X = get_diff(X,new_length=opt.new_length)
#                print(type(X))
                X = X.reshape((-3,-3,-2))
#                print(X.shape)
                pred = net(X.astype(opt.dtype, copy=False))
                output.append(pred)
            loss = [loss_fn(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()
#        if opt.use_amp:
#            with amp.scale_loss(loss, trainer) as scaled_loss:
#                ag.backward(scaled_loss)
#        else:
#            ag.backward(loss)
        # Optimize
        trainer.step(batch_size)

        # Update metrics
#        train_loss += sum([l.mean().asscalar() for l in loss])
        train_metric.update(label, output)
        train_loss_iter = sum([l.mean().asscalar() for l in loss]) / len(loss)
        train_loss_epoch += train_loss_iter
        
        train_metric_name, train_metric_score = train_metric.get()
        if opt.log_interval and not (i+1) % opt.log_interval:
            print('Epoch[%03d] Batch [%04d]/[%04d]\tSpeed: %f samples/sec\t %s=%f\t loss=%f\t lr=%f' % (
                        epoch, i, num_train_iter, batch_size*opt.log_interval/(time.time()-btic),
                        train_metric_name, train_metric_score*100, train_loss_epoch/(i+1), trainer.learning_rate))
            btic = time.time()        

    name, acc = train_metric.get()
    
    # test
    acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)

    # Update history and print metrics
    train_history.update([acc,acc_top1_val,acc_top5_val])
    print('[Epoch %d] train=%f loss=%f time: %f' %
        (epoch, acc, train_loss / (i+1), time.time()-tic))
    print('[Epoch %d] val top1 =%f top5=%f val loss=%f' %
        (epoch, acc_top1_val, acc_top5_val, loss_val ))

# We can plot the metric scores with:
train_history.plot()

#if __name__ == '__main__':
#main2()


