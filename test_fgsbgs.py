#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:54:16 2019

@author: bdus

take background and foreground as input


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
import ucf101_bgs as ucf101
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory

from model_zoo import get_model as myget

# number of GPUs to use
num_gpus = 1
#ctx = [mx.gpu(i) for i in range(num_gpus)]
ctx = [mx.gpu(1)]

# Get the model 
#net = get_model(name='vgg16_ucf101', nclass=101, num_segments=3)
#net = myget(name='simple', nclass=101, num_segments=3)
net = myget(name='dualnet_outavg', nclass=101, num_segments=3)

net.collect_params().reset_ctx(ctx)
#print(net)

transform_train = transforms.Compose([
    # Fix the input video frames size as 256×340 and randomly sample the cropping width and height from
    # {256,224,192,168}. After that, resize the cropped regions to 224 × 224.
    video.VideoMultiScaleCrop(size=(224, 224), scale_ratios=[1.0, 0.875, 0.75, 0.66]),
    # Randomly flip the video frames horizontally
    video.VideoRandomHorizontalFlip(),
    # Transpose the video frames from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    video.VideoToTensor(),
    # Normalize the video frames with mean and standard deviation calculated across all images
    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Batch Size for Each GPU
per_device_batch_size = 8
# Number of data loader workers
num_workers = 2
# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

# Set train=True for training data. Here we only use a subset of UCF101 for demonstration purpose.
# The subset has 101 training samples, one sample per class.

#train_dataset = ucf101.classification.UCF101(train=True, num_segments=3, transform=transform_train)

#train_dataset = ucf101.classification.UCF101(train=True, num_segments=3, 
#                transform=transform_train, root='/media/hp/dataset/UCF101/BGSDecom/FrameDifference/bgs',
#                setting='/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_2_rawframes.txt',
#                name_pattern='image_%05d.jpg')

train_dataset = ucf101.classification.UCF101(train=True, num_segments=3, transform=transform_train,
                                             root_bgs='/media/hp/data/BGSDecom/FrameDifference/bgs',
                                             root_fgs='/media/hp/data/BGSDecom/FrameDifference/fgs',
                                             setting='/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_2.txt',
                                             name_pattern='img_%05d.jpg'#'img_%05d.jpg'
                                             )

val_dataset = ucf101.classification.UCF101(train=False, num_segments=3, transform=transform_train,
                                             root_bgs='/media/hp/data/BGSDecom/FrameDifference/bgs',#'/home/hp/lixiaoyu/dataset/flow',
                                             root_fgs='/media/hp/data/BGSDecom/FrameDifference/fgs',
                                             setting='/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_2.txt',
                                             name_pattern='img_%05d.jpg'#'img_%05d.jpg'
                                             )

train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
print('Load %d training samples.' % len(train_dataset))
val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    

# Learning rate decay factor
lr_decay = 0.1
# Epochs where learning rate decays
lr_decay_epoch = [30, 60, np.inf]

# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
optimizer_params = {'learning_rate': 0.001, 'wd': 0.0001, 'momentum': 0.9}

# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

train_history = TrainingHistory(['training-acc','val-top1-acc','val-top5-acc'])

epochs = 80
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
        data_bgs = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        data_fgs = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        val_outputs = []
        for _, (X_bgs,X_fgs) in enumerate(zip(data_bgs,data_fgs)):
            X_bgs = X_bgs.reshape((-1,) + X_bgs.shape[2:])
            X_fgs = X_fgs.reshape((-1,) + X_fgs.shape[2:])
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

for epoch in range(epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0

    # Learning rate decay
    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

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
                X_bgs = X_bgs.reshape((-1,) + X_bgs.shape[2:])
                X_fgs = X_fgs.reshape((-1,) + X_fgs.shape[2:])
                pred = net(X_bgs,X_fgs)
                output.append(pred)
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        trainer.step(batch_size)

        # Update metrics
        train_loss += sum([l.mean().asscalar() for l in loss])
        train_metric.update(label, output)

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