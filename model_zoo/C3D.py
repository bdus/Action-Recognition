#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sat Dec 21 19:36:20 2019

@author: bdus

复现C3D

input shape : NCDHW Nx3x16x112x112
chop/resize shape 128x171

if you need pretrained model: 
    https://github.com/DavideA/c3d-pytorch
    https://github.com/axon-research/c3d-keras
    
"""
import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock


__all__ = ['c3d']

class C3D(HybridBlock):
    def __init__(self,nclass,dropout_ratio=0.5, batch_normal = True, init_std=0.001,**kwargs):
        super(C3D, self).__init__()
        self.nclass = nclass
        self.num_segments = 16
        self.feat_dim = 4096
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std        
        self.config_3d_conv = ([1,1,2,2,2],[64,128,256,512,512]) # (\#layer , channels)
        self.config_3d_pool = ([(1,2,2),2,2,2,2],[(1,2,2),2,2,2,2],[0,0,0,0,(0,1,1)]) # (pool_size,strides,padding)
        self.config_fc = [self.feat_dim, self.feat_dim]
       
        with self.name_scope():
            self.features = self._make_3d_feature(self.config_3d_conv,self.config_3d_pool, batch_normal)
            for num in self.config_fc:
                self.features.add(nn.Dense(num, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
                self.features.add(nn.Dropout(rate=self.dropout_ratio))

        self.output = nn.Dense(units=self.nclass, in_units=self.feat_dim,
                               weight_initializer=init.Normal(sigma=self.init_std))
        
    def _make_3d_feature(self, config_3d_conv,config_3d_pool,batch_normal):
        featurizer = nn.HybridSequential(prefix='')
        conv_layer, conv_channels = config_3d_conv
        pool_size, pool_stride, pool_padding = config_3d_pool
        assert len(conv_layer) == len(conv_channels) == len(pool_size) == len(pool_stride) == len(pool_padding)
        
        for i, num in enumerate(conv_layer):
            for _ in range(num):
                featurizer.add(nn.Conv3D(channels=conv_channels[i],
                                  kernel_size=(3,3,3),strides=(1,1,1),
                                  padding=(1,1,1),
                                  weight_initializer=init.Xavier(
                                          rnd_type='gaussian',factor_type='out',magnitude=2),
                                  bias_initializer='zero'))
                if batch_normal:                       
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add( nn.MaxPool3D(pool_size=pool_size[i],strides=pool_stride[i],padding=pool_padding[i]) )            
        # flatten to (N, 8192)
        featurizer.add(nn.Flatten())
        return featurizer 

        
    def hybrid_forward(self, F, x):            
        x = self.features(x)
        x = self.output(x)
        return x

def c3d(**kwargs):
    return get_simple(**kwargs)

def get_simple(pretrained=False,
               root='para', **kwargs):
    net = C3D(**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
#        filepath = os.path.join(root,'simple%d.params'%(index))
#        print(filepath)
#        net.load_parameters(filepath)
        pass
    else:
        net.initialize()        
    return net