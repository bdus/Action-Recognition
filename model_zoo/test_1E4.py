#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:29:06 2019

@author: bdus

this is the model for idea1 experiment 4
video frames are take part into foregrounds and backgrounds
then feed into a dual stream network respectively
the network will fusion the result in difference method

the input is bgs and fgs frame


"""


import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo import vgg16


__all__ = ['DualNet','dualnet_avg','dualnet_max','dualnet_outmax','dualnet_outavg']

class DualNet(HybridBlock):
    def __init__(self,nclass,num_segments, fusion_method='avg',num_crop=1,input_channel=3,dropout_ratio=0.9, init_std=0.001,feat_dim=4096,**kwargs):
        super(DualNet, self).__init__(**kwargs)
        self.nclass = nclass
        self.num_segments = num_segments
        self.feat_dim = feat_dim
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std
        self.num_crop=num_crop
        self.fusion_method = fusion_method
        
        pretrained_model_bgs = vgg16(pretrained=True)
        pretrained_model_fgs = vgg16(pretrained=True)
        
        vgg16_feature_bgs = pretrained_model_bgs.features
        vgg16_feature_fgs = pretrained_model_fgs.features
        if input_channel == 3:
            self.feature_bgs = vgg16_feature_bgs
            self.feature_fgs = vgg16_feature_fgs
        else:
            raise ValueError('not support input_channel not equal 3')
            # change the input channel of first layer convnet                             
#            self.feature = nn.HybridSequential()
#            with pretrained_model.name_scope():
#                self.feature.add(nn.Conv2D(in_channels=input_channel,channels=64,kernel_size=3,strides=(1,1),padding=(1,1)))
#            self.feature[0].initialize()
#            for layer in vgg16_feature[1:]:
#                self.feature.add(layer)
        
        def update_dropout_ratio(block):
            if isinstance(block, nn.basic_layers.Dropout):
                block._rate = self.dropout_ratio
        self.feature_bgs.apply(update_dropout_ratio)
        self.feature_fgs.apply(update_dropout_ratio)
        
        if self.fusion_method == 'avg' or self.fusion_method == 'max':
            self.output = nn.Dense(units=self.nclass, in_units=self.feat_dim, weight_initializer=init.Normal(sigma=self.init_std))            
            self.output.initialize()
        elif self.fusion_method == 'out_avg' or self.fusion_method == 'out_max':
            self.output_fgs = nn.Dense(units=self.nclass, in_units=self.feat_dim, weight_initializer=init.Normal(sigma=self.init_std))
            self.output_bgs = nn.Dense(units=self.nclass, in_units=self.feat_dim, weight_initializer=init.Normal(sigma=self.init_std))
            self.output_fgs.initialize()
            self.output_bgs.initialize()
        else:
            raise ValueError("not support fusion method")

        
    def hybrid_forward(self, F, x_bgs, x_fgs):            
        x_bgs = self.feature_bgs(x_bgs)
        x_fgs = self.feature_fgs(x_fgs)
        
        x_bgs = F.reshape(x_bgs, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x_fgs = F.reshape(x_fgs, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        
        if self.fusion_method == 'avg' or self.fusion_method == 'max':
            x = F.concat(x_bgs,x_fgs,dim=1)        
            if self.fusion_method == 'avg':
                x = F.mean(x, axis=1)
            elif self.fusion_method == 'max':
                x = F.max(x,axis=1)
            else:
                raise ValueError('fusion_method not supported')
            x = self.output(x)
            
        elif self.fusion_method == 'out_avg' or self.fusion_method == 'out_max':
            if self.fusion_method == 'out_avg':
                x_bgs = F.mean(x_bgs, axis=1)
                x_fgs = F.mean(x_fgs, axis=1)                
            elif self.fusion_method == 'out_max':
                x_bgs = F.max(x_bgs,axis=1)
                x_fgs = F.max(x_fgs,axis=1)
            else:
                raise ValueError('fusion_method not supported')
            x_bgs = self.output_bgs(x_bgs)
            x_fgs = self.output_fgs(x_fgs)
#            x = F.concat(x_bgs,x_fgs,dim=1)
#            x = F.mean(x,axis=1)
            x = (x_bgs+x_fgs)/2
        return x

    def __getitem__(self, key):
        return self.net[key]
    
    def __len__(self):
        return len(self.net)

def dualnet_avg(**kwargs):
    return get_dualnet(fusion_method='avg',**kwargs)

def dualnet_max(**kwargs):
    return get_dualnet(fusion_method='max',**kwargs)

def dualnet_outavg(**kwargs):
    return get_dualnet(fusion_method='out_avg',**kwargs)

def dualnet_outmax(**kwargs):
    return get_dualnet(fusion_method='out_max',**kwargs)

def get_dualnet(fusion_method,pretrained=False,
               root='para', **kwargs):
    net = DualNet(fusion_method=fusion_method,**kwargs)
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