#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:14:03 2019

@author: bdus
"""
import os
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

class Simple(HybridBlock):
    def __init__(self,**kwargs):
        super(Simple, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
                nn.Conv2D(8,5),
                nn.Conv2D(16,5),
                nn.BatchNorm(momentum=0.8),
                nn.MaxPool2D(pool_size=(2, 2)),
                nn.Conv2D(32,5),
                nn.BatchNorm(momentum=0.8),
                nn.MaxPool2D(pool_size=(2, 2)),
                nn.Conv2D(64,3),
                nn.BatchNorm(momentum=0.8),
                nn.MaxPool2D(pool_size=(2, 2)),
                nn.Conv2D(128,3),
                nn.BatchNorm(momentum=0.8),
                nn.MaxPool2D(pool_size=(2, 2)),
                nn.Conv2D(32,1),
                nn.Flatten(),
                nn.Dense(256,activation='relu'),
                nn.Dropout(0.25),
                nn.Dense(101,activation='relu')
                )
    def hybrid_forward(self, F, x):            
        x = self.net(x)
        return x

    def __getitem__(self, key):
        return self.net[key]
    
    def __len__(self):
        return len(self.net)

def simple(**kwargs):
    return get_simple(**kwargs)

def get_simple(pretrained=False,
               root='para', **kwargs):
    net = Simple(**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
#        filepath = os.path.join(root,'simple%d.params'%(index))
#        print(filepath)
#        net.load_parameters(filepath)
        pass
    else:
        net.initialize()
    return net