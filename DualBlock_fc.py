#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2020-2-5 18:17:19

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
from model_zoo import get_model as myget

__all__ = ['DualBlock','get_dualnet']

class DualBlock(HybridBlock):
    def __init__(self,nclass,num_segments,fgs_model,bgs_model,fusion_method='avg',num_crop=1,input_channel=3,dropout_ratio=0.9, init_std=0.001,feat_dim=4096,**kwargs):
        super(DualBlock, self).__init__(**kwargs)
        self.nclass = nclass
        self.num_segments = num_segments
        self.feat_dim = feat_dim
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std
        self.num_crop=num_crop
        self.fusion_method = fusion_method
        self.fgs_model = fgs_model
        self.bgs_model = bgs_model
        print('fusion_method:',fusion_method)
        
        with self.name_scope():
            self.pretrained_model_bgs = myget(name=self.bgs_model, nclass=self.nclass, num_segments=self.num_segments,input_channel=input_channel,pretrained=True)
            self.dp = nn.Dropout(rate=self.dropout_ratio)
            self.fc = nn.HybridSequential(prefix='')
            self.fc.add( nn.Dense(units=512, in_units=512+1024,
                               weight_initializer=init.Normal(sigma=self.init_std)),
                             nn.Dense(units=self.nclass, in_units=512,
                               weight_initializer=init.Normal(sigma=self.init_std)) )
        self.pretrained_model_fgs = myget(name=self.fgs_model,nclass=self.nclass,num_segments=self.num_segments,input_channel=input_channel,pretrained=True)
        self.fc.initialize()
              
    def hybrid_forward(self, F, x_bgs, x_fgs):   
        #print(x_bgs.shape)#(80, 3, 224, 224)
        #print(x_fgs.shape)#(80, 3, 224, 224)
        #x_bgs = self.pretrained_model_bgs(x_bgs)
        #x_fgs = self.pretrained_model_fgs(x_fgs)
        if 'resnet18_v1b' in self.bgs_model:
            x_bgs = self.pretrained_model_bgs.conv1(x_bgs)
            x_bgs = self.pretrained_model_bgs.bn1(x_bgs)
            x_bgs = self.pretrained_model_bgs.relu(x_bgs)
            x_bgs = self.pretrained_model_bgs.maxpool(x_bgs)
            x_bgs = self.pretrained_model_bgs.layer1(x_bgs)
            x_bgs = self.pretrained_model_bgs.layer2(x_bgs)
            x_bgs = self.pretrained_model_bgs.layer3(x_bgs)
            x_bgs = self.pretrained_model_bgs.layer4(x_bgs)
            x_bgs = self.pretrained_model_bgs.avgpool(x_bgs)
            x_bgs = self.pretrained_model_bgs.flat(x_bgs)
            x_bgs = self.pretrained_model_bgs.drop(x_bgs)
            #print('res:',x_bgs.shape) (160, 512)
            x_bgs = F.reshape(x_bgs, shape=(-1, self.num_segments * self.num_crop, 512))
            x_bgs = F.mean(x_bgs, axis=1)
        else:
            raise ValueError('fusion_method not supported')
        if 'eco_resnet18_v1b' in self.fgs_model:
            x_fgs = self.pretrained_model_fgs.conv1(x_fgs)
            x_fgs = self.pretrained_model_fgs.bn1(x_fgs)
            x_fgs = self.pretrained_model_fgs.relu(x_fgs)
            x_fgs = self.pretrained_model_fgs.maxpool(x_fgs)
            x_fgs = self.pretrained_model_fgs.layer1(x_fgs)
            x_fgs = self.pretrained_model_fgs.layer2(x_fgs)
            x_fgs = x_fgs.reshape((-1,self.num_segments,128,28,28))
            x_fgs = x_fgs.transpose(axes=(0,2,1,3,4))
            x_fgs = self.pretrained_model_fgs.features_3d(x_fgs)
            x_fgs = F.flatten(x_fgs) 
        else:
            raise ValueError('fusion_method not supported')
        
        #print(x_bgs.shape) (20, 512)
        #print(x_fgs.shape) (20, 1024)
        x = F.concat(x_bgs,x_fgs,dim=1)      
        #print('x:',x.shape)
        #if self.fusion_method == 'avg':
            #x = F.mean(x, axis=1)                          
        #elif self.fusion_method == 'max':
            #x = F.max(x,axis=1)
        #else:
            #raise ValueError('fusion_method not supported') 
        #print('x:',x.shape)
        x = self.dp(x)
        x = self.fc(x)
        return x

def dualnet_avg(fgs_model,bgs_model,fgs_path,bgs_path,**kwargs):
    return get_dualnet(fgs_model,bgs_model,fgs_path,bgs_path,fusion_method='avg',**kwargs)

def dualnet_max(fgs_model,bgs_model,fgs_path,bgs_path,**kwargs):
    return get_dualnet(fgs_model,bgs_model,fgs_path,bgs_path,fusion_method='max',**kwargs)

def get_dualnet(fgs_model,bgs_model,fgs_path,bgs_path, **kwargs):
    net = DualBlock(fgs_model=fgs_model,bgs_model=bgs_model,**kwargs)
    print(bgs_path,',',os.path.exists(bgs_path))
    #net.pretrained_model_bgs.load_parameters(bgs_path)
    #net.pretrained_model_fgs.load_parameters(fgs_path)
 
    return net