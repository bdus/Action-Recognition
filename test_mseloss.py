#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:43:46 2019

@author: bdus

verify MSE loss

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
from gluoncv.data import ucf101
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
net = myget('mseloss_vgg16',nclass=101,num_segments=3)

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
per_device_batch_size = 12
# Number of data loader workers
num_workers = 1
# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

# Set train=True for training data. Here we only use a subset of UCF101 for demonstration purpose.
# The subset has 101 training samples, one sample per class.

#train_dataset = ucf101.classification.UCF101(train=True, num_segments=3, transform=transform_train)

#train_dataset = ucf101.classification.UCF101(train=True, num_segments=3, 
#                transform=transform_train, root='/media/hp/dataset/UCF101/BGSDecom/FrameDifference/bgs',
#                setting='/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_2_rawframes.txt',
#                name_pattern='image_%05d.jpg')

train_dataset = ucf101.classification.UCF101(train=True, num_segments=3, transform=transform_train,root='/home/hp/lixiaoyu/dataset/flow')

train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
print('Load %d training samples.' % len(train_dataset))


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
loss_mse = gluon.loss.L2Loss()

def myloss(yhat,y):    
    # yhat is the output of net, is a list
    #a,b,c = yhat#.split(axis=0,num_outputs=3)
    ans = 0.0
    for i in yhat:
        ans += loss_fn(i,y)
    while len(yhat) > 0:
        a = yhat.pop()
        for i in yhat:
            ans += loss_mse(a,i)
#    print('a.shape',a.shape)
#    print('y.shape',y.shape)
#    ans = loss_fn(a,y) + loss_fn(b,y) + loss_fn(c,y)  \
#        + loss_mse(a,b)  + loss_mse(a,c)  + loss_mse(b,c)    
    return ans

def mean_loss(yhat,y):
    ans = 0.0
    mean = nd.zeros_like(y)    
    for i in yhat:
        mean += i
    mean = mean / len(yhat)
    return loss_fn(mean,y)
        
    
def mymean(output):
    ans = list()
    for item in output:
        a,b,c = item#.split(axis=0,num_outputs=3)
        ans.append( nd.add_n(a,b,c)/3 )
    return ans

train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training-acc'])


epochs = 80
lr_decay_count = 0


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
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
#        print(data[0].shape) #(10, 3, 3, 224, 224)

        # AutoGrad
        with ag.record():
            output = []
            for _, X in enumerate(data):
#                print('X',X.shape) # X (10, 3, 3, 224, 224)
                X = X.reshape((-1,) + X.shape[2:])
#                print('reshape',X.shape) #reshape (30, 3, 224, 224)
                pred = net(X)
#                print('pred',pred.shape) #pred (30, 101)
                output.append(pred)
            loss = [myloss(yhat, y) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        trainer.step(batch_size)

        # Update metrics
        train_loss += sum([l.mean().asscalar() for l in loss])
        train_metric.update(label, mymean(output)) 

    name, acc = train_metric.get()

    # Update history and print metrics
    train_history.update([acc])
    print('[Epoch %d] train=%f loss=%f time: %f' %
        (epoch, acc, train_loss / (i+1), time.time()-tic))

# We can plot the metric scores with:
train_history.plot()