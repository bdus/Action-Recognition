#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:04:52 2019

@author: bdus

R(2+1) 

ori paper acc on ucf101 : 

reference :
    https://github.com/MRzzm/action-recognition-models-pytorch
        
x = nd.zeros(shape=(N,8,3,112,112))
y = net(x.transpose(axes=(0,2,1,3,4)))

"""
import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
#from mxnet.gluon.nn import HybridBlock

__all__ = ['res21d_34_3d']

class Res21D_Block(nn.HybridBlock):
    def __init__(self, in_channel,out_channel,spatial_stride=1,temporal_stride=1,**kwargs):
        super(Res21D_Block,self).__init__()
        self.MidChannel1 = int( (27*in_channel*out_channel) / (9*in_channel + 3*out_channel) )
        self.MidChannel2 = int( (27*out_channel*out_channel) / (12 * out_channel) )
        self.conv1_2D = nn.Conv3D(in_channels=in_channel, channels=self.MidChannel1, kernel_size=(1,3,3),
                                  strides=(1,spatial_stride,spatial_stride),padding=(0,1,1),weight_initializer=init.Xavier(),bias_initializer='zero')
        self.bn1_2D = nn.BatchNorm(in_channels=self.MidChannel1)
        self.conv1_1D = nn.Conv3D(in_channels=self.MidChannel1, channels=out_channel, kernel_size=(3,1,1),
                                  strides=(temporal_stride,1,1),padding=(1,0,0),weight_initializer=init.Xavier(),bias_initializer='zero')
        self.bn1_1D = nn.BatchNorm(in_channels=out_channel)
        self.conv2_2D = nn.Conv3D(in_channels=out_channel, channels=self.MidChannel2, kernel_size=(1,3,3),
                                  strides=(1,1,1),padding=(0,1,1),weight_initializer=init.Xavier(),bias_initializer='zero')
        self.bn2_2D = nn.BatchNorm(in_channels=self.MidChannel2)
        self.conv2_1D = nn.Conv3D(in_channels=self.MidChannel2, channels=out_channel, kernel_size=(3,1,1),strides=(1,1,1),
                                  padding=(1,0,0),weight_initializer=init.Xavier(),bias_initializer='zero')
        self.bn2_1D = nn.BatchNorm(in_channels=out_channel)
        self.relu = nn.Activation('relu')
        
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample = nn.HybridSequential()
            self.down_sample.add(
                    nn.Conv3D(in_channels=in_channel, channels=out_channel, kernel_size=(1,1,1),
                              strides=(temporal_stride,spatial_stride,spatial_stride),weight_initializer=init.Xavier(),use_bias=False),
                    nn.BatchNorm(in_channels=out_channel) )
        else:
            self.down_sample = None
            
    def hybrid_forward(self, F, x):
        x_branch = self.conv1_2D(x)
        x_branch = self.bn1_2D(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv1_1D(x_branch)
        x_branch = self.bn1_1D(x_branch)
        x_branch = self.relu(x_branch)
        
        x_branch = self.conv2_2D(x_branch)
        x_branch = self.bn2_2D(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2_1D(x_branch)
        x_branch = self.bn2_1D(x_branch)
        
        if self.down_sample is not None:
            x = self.down_sample(x)
            
        return self.relu(x_branch+x)
       

class Res21D_34(nn.HybridBlock):
    #
    def __init__(self,nclass,input_channel=3,batch_normal=True, dropout_ratio=0.8, init_std=0.001,**kwargs):
        super(Res21D_34, self).__init__()
        self.nclass = nclass
        self.new_length = 8
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std
#        self.config_3d_layer = [2,2,2,2]
#        self.config_3d_temporal_stride = [1,2,2,2]
        with self.name_scope():
            self.conv1 = nn.Conv3D(in_channels=input_channel, channels=64, kernel_size=(3,7,7),strides=(1,2,2),padding=(1,3,3),weight_initializer=init.Xavier(),bias_initializer='zero')
            self.conv2 = nn.HybridSequential()
            self.conv2.add(
                    Res21D_Block(in_channel=64,out_channel=64,spatial_stride=2),
                    Res21D_Block(64,64),
                    Res21D_Block(64,64) )
            self.conv3 = nn.HybridSequential()
            self.conv3.add(
                    Res21D_Block(in_channel=64,out_channel=128,spatial_stride=2,temporal_stride=2),
                    Res21D_Block(128,128),
                    Res21D_Block(128,128),
                    Res21D_Block(128,128))
            self.conv4 = nn.HybridSequential()
            self.conv4.add(
                    Res21D_Block(in_channel=128,out_channel=256,spatial_stride=2,temporal_stride=2),
                    Res21D_Block(256,256),
                    Res21D_Block(256,256),
                    Res21D_Block(256,256),
                    Res21D_Block(256,256),
                    Res21D_Block(256,256))
            self.conv5 = nn.HybridSequential()
            self.conv5.add(
                    Res21D_Block(in_channel=256,out_channel=512,spatial_stride=2,temporal_stride=2),
                    Res21D_Block(512,512),
                    Res21D_Block(512,512))
            self.avg_pool = nn.AvgPool3D(pool_size=(1,4,4))
            self.output = nn.Dense(in_units=512,units=nclass,weight_initializer=init.Normal(sigma=init_std))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = self.output(x)
        return x

def res21d_34_3d(**kwargs):
    return get_simple(**kwargs)

def get_simple(pretrained=False,
               root='para', **kwargs):
    net = Res21D_34(**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
#        filepath = os.path.join(root,'simple%d.params'%(index))
#        print(filepath)
#        net.load_parameters(filepath)
        pass
    else:
        net.initialize()        
    return net