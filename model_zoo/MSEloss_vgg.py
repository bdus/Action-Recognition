#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:40:38 2019

@author: bdus

VGG

"""


import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo import vgg16


__all__ = ['MSElossVGG16','mseloss_vgg16']

class MSElossVGG16(HybridBlock):
    def __init__(self,nclass,num_segments, num_crop=1,input_channel=3,dropout_ratio=0.9, init_std=0.001,feat_dim=4096,**kwargs):
        super(MSElossVGG16, self).__init__(**kwargs)
        self.nclass = nclass
        self.num_segments = num_segments
        self.feat_dim = feat_dim
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std
        self.num_crop=num_crop
        
        pretrained_model = vgg16(pretrained=True)
        vgg16_feature = pretrained_model.features
        if input_channel == 3:
            self.feature = vgg16_feature
        else:
            # change the input channel of first layer convnet                             
            self.feature = nn.HybridSequential()
            with pretrained_model.name_scope():
                self.feature.add(nn.Conv2D(in_channels=input_channel,channels=64,
                                           kernel_size=3,strides=(1,1),padding=(1,1),
                                           weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                        bias_initializer='zeros'))
            self.feature[0].initialize()
            for layer in vgg16_feature[1:]:
                self.feature.add(layer)
        
        def update_dropout_ratio(block):
            if isinstance(block, nn.basic_layers.Dropout):
                block._rate = self.dropout_ratio
        self.feature.apply(update_dropout_ratio)
        self.output = nn.Dense(units=self.nclass, in_units=self.feat_dim,
                               weight_initializer=init.Normal(sigma=self.init_std))
        self.output.initialize()
        
    def hybrid_forward(self, F, x):            
        x = self.feature(x)
        
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        #x = F.mean(x, axis=1)        
        x = x.split(axis=1,num_outputs=self.num_segments)
        
        x = [self.output(i) for i in x]
        return x

    def __getitem__(self, key):
        return self.net[key]
    
    def __len__(self):
        return len(self.net)

def mseloss_vgg16(**kwargs):
    return get_mseloss(**kwargs)

def get_mseloss(pretrained=False,
               root='para', **kwargs):
    net = MSElossVGG16(**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
#        filepath = os.path.join(root,'simple%d.params'%(index))
#        print(filepath)
#        net.load_parameters(filepath)
        pass
    else:
        #net.initialize()
        pass
    return net