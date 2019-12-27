#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 17:27:24 2019

@author: bdus

F_stCN ICCV2015

ori paper acc on ucf101 : 88.1

reference :
    https://github.com/MRzzm/action-recognition-models-pytorch
        
x = nd.zeros(shape=(N,16,3,204,204))
x_diff = nd.zeros(shape=(N,15,3,204,204))

y = net(x.transpose(axes=(0,2,1,3,4)))

"""
import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

__all__ = ['fstcn_3d']

class TCL(HybridBlock):
    def __init__(self, in_channel,**kwargs):
        super(TCL,self).__init__()
        self.branch1 = nn.HybridSequential()
        self.branch1.add(
                nn.Conv3D(in_channels=in_channel,channels=32,kernel_size=(3,1,1),
                          strides=(1,1,1),padding=(1,0,0),weight_initializer=init.Xavier(),bias_initializer='zero'),
                nn.Activation('relu'),
#                nn.BatchNorm(),
                nn.MaxPool3D(pool_size=(2,1,1),strides=(2,1,1)))
        self.branch2 = nn.HybridSequential()
        self.branch2.add(
                nn.Conv3D(in_channels=in_channel,channels=32,kernel_size=(5,1,1),
                          strides=(1,1,1),padding=(2,0,0),weight_initializer=init.Xavier(),bias_initializer='zero'),
                nn.Activation('relu'),
#                nn.BatchNorm(),
                nn.MaxPool3D(pool_size=(2,1,1),strides=(2,1,1)))
            
    def hybrid_forward(self, F, x):
        res1 = self.branch1(x)
        res2 = self.branch2(x)
        return F.concat(res1,res2,dim=1)
       

class FstCN(HybridBlock):
    def __init__(self,nclass,input_channel=3,batch_normal=True, dropout_ratio=0.8, init_std=0.001,**kwargs):
        super(FstCN, self).__init__()
        self.nclass = nclass
        self.new_length = 16 +1 
        #self.feat_dim = 4096
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std
#        self.config_3d_layer = [2,2,2,2]
#        self.config_3d_temporal_stride = [1,2,2,2]
        with self.name_scope():
            self.SCL1 = nn.HybridSequential()
            self.SCL1.add(
                        nn.Conv3D(in_channels=3, channels=96, kernel_size=(1,7,7),strides=(1,2,2),padding=(0,3,3),weight_initializer=init.Xavier(),bias_initializer='zero'),
                        nn.Activation('relu'),                        
                        nn.BatchNorm(),
                        nn.MaxPool3D(pool_size=(1,3,3),strides=(1,2,2)) )            
            self.SCL2 = nn.HybridSequential()
            self.SCL2.add(
                        nn.Conv3D(in_channels=96, channels=256, kernel_size=(1,5,5),strides=(1,2,2),padding=(0,2,2),weight_initializer=init.Xavier(),bias_initializer='zero'),
                        nn.Activation('relu'),
                        nn.BatchNorm(),
                        nn.MaxPool3D(pool_size=(1,3,3),strides=(1,2,2)) )
            self.SCL3 = nn.HybridSequential()
            self.SCL3.add(
                        nn.Conv3D(in_channels=256, channels=512, kernel_size=(1,3,3),strides=(1,1,1),padding=(0,1,1),weight_initializer=init.Xavier(),bias_initializer='zero'),
                        nn.Activation('relu') ,nn.BatchNorm())
            self.SCL4 = nn.HybridSequential()
            self.SCL4.add(
                        nn.Conv3D(in_channels=512, channels=512, kernel_size=(1,3,3),strides=(1,1,1),padding=(0,1,1),weight_initializer=init.Xavier(),bias_initializer='zero'),
                        nn.Activation('relu') ,nn.BatchNorm())
            self.Parallel_temporal = nn.HybridSequential()
            self.Parallel_temporal.add(
                    nn.Conv3D(in_channels=512,channels=128,kernel_size=(1,3,3),strides=(1,1,1),padding=(0,1,1),weight_initializer=init.Xavier(),bias_initializer='zero'),
                    nn.Activation('relu'),
                    nn.BatchNorm(),
                    nn.MaxPool3D(pool_size=(1,3,3),strides=(1,3,3)),
                    TCL(128)
                    )
            self.Parallel_spatial = nn.HybridSequential()
            self.Parallel_spatial.add(
                    nn.Conv2D(in_channels=512,channels=128,kernel_size=(3,3),strides=(1,1),padding=(1,1),weight_initializer=init.Xavier(),bias_initializer='zero'),
                    nn.Activation('relu'),                    
                    nn.MaxPool2D(pool_size=(3,3),strides=(3,3))
                    )
            self.tem_fc = nn.HybridSequential()
            self.tem_fc.add(
                    nn.Dense(in_units=8192,units=4096,weight_initializer=init.Normal(sigma=init_std)),
                    nn.Dropout(rate=dropout_ratio),
                    nn.Dense(in_units=4096,units=2048,weight_initializer=init.Normal(sigma=init_std)),
                    )
            self.spa_fc = nn.HybridSequential()
            self.spa_fc.add(
                    nn.Dense(in_units=2048,units=4096,weight_initializer=init.Normal(sigma=init_std)),
                    nn.Dropout(rate=dropout_ratio),
                    nn.Dense(in_units=4096,units=2048,weight_initializer=init.Normal(sigma=init_std)),
                    )
            self.fc = nn.Dense(in_units=4096,units=2048,weight_initializer=init.Normal(sigma=init_std))
            self.out = nn.Dense(in_units=2048,units=nclass,weight_initializer=init.Normal(sigma=init_std))
              
    def hybrid_forward(self, F, x):
        #x.shape=(N,3,16+1,204,204)
        x = self.SCL1(x)
        x = self.SCL2(x)
        x = self.SCL3(x)
        x = self.SCL4(x)
        xs = x[:,:,16//2,:,:]
        xt = F.slice_axis(data=x,axis=2,begin=1,end=17) - F.slice_axis(data=x,axis=2,begin=0,end=16)
        xs = self.Parallel_spatial(xs)
        xs = self.spa_fc(xs)
        xt = self.Parallel_temporal(xt)
        xt = F.reshape(data=xt,shape=(0,-1))
        xt = self.tem_fc(xt)
        x = F.concat(xs,xt,dim=1)
        x = self.fc(x)
        x = self.out(x)
        return x

def fstcn_3d(**kwargs):
    return get_simple(**kwargs)

def get_simple(pretrained=False,
               root='para', **kwargs):
    net = FstCN(**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
#        filepath = os.path.join(root,'simple%d.params'%(index))
#        print(filepath)
#        net.load_parameters(filepath)
        pass
    else:
        net.initialize()        
    return net