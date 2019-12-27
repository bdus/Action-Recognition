#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sun Dec 22 20:49:26 2019

@author: bdus

1708.05038 3D resnet18 not CVPR2018 version

ori paper acc on ucf101 : 42.4%

reference :
    https://github.com/MRzzm/action-recognition-models-pytorch/blob/master/3DCNN/Res3D/Res3D.py
    https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
    
x = nd.zeros(shape=(7,8,3,112,112))
y = net(x.transpose(axes=(0,2,1,3,4)))


"""
import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

__all__ = ['resnet18_3d']

class BasicBlock(HybridBlock):
    def __init__(self, in_channel,out_channel, spatial_stride=1,temporal_stride=1,downsample=None,**kwargs):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv3D(in_channels=in_channel,channels=out_channel,
                               kernel_size=(3,3,3),strides=(temporal_stride,spatial_stride,spatial_stride),padding=(1,1,1),
                               weight_initializer=init.Xavier(rnd_type='gaussian',factor_type='out',magnitude=2),bias_initializer='zero')
        self.conv2 = nn.Conv3D(in_channels=out_channel,channels=out_channel,
                               kernel_size=(3,3,3),strides=(1,1,1),padding=(1,1,1),
                               weight_initializer=init.Xavier(rnd_type='gaussian',factor_type='out',magnitude=2),bias_initializer='zero')
        self.bn1 = nn.BatchNorm(in_channels=out_channel,epsilon=0.001)
        self.bn2 = nn.BatchNorm(in_channels=out_channel,epsilon=0.001)
        self.relu1 = nn.Activation('relu')
        self.relu2 = nn.Activation('relu')
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample = nn.HybridSequential()
            self.down_sample.add(
                    nn.Conv3D(in_channels=in_channel,channels=out_channel,
                               kernel_size=1,strides=(temporal_stride,spatial_stride,spatial_stride),
                               weight_initializer=init.Xavier(rnd_type='gaussian',factor_type='out',magnitude=2)
                               ,use_bias=False),
                    nn.BatchNorm(in_channels=out_channel,epsilon=0.001)
                    )
        else:
            self.down_sample = None
            
    def hybrid_forward(self, F, x):
        #residual  = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample is not None:
            x = self.down_sample(x)
        return self.relu2(x+out)      
        


class Res3D(HybridBlock):
    def __init__(self,nclass,input_channel=3,dropout_ratio=0.5, init_std=0.001,**kwargs):
        super(Res3D, self).__init__()
        self.nclass = nclass
        self.num_segments = 8
        #self.feat_dim = 4096
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std        
        self.config_3d_layer = [2,2,2,2]
        self.config_3d_temporal_stride = [1,2,2,2]
                       
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            # conv1
            self.features.add(
                    nn.Conv3D(in_channels=input_channel,channels=64,
                               kernel_size=(3,7,7),strides=(1,2,2),padding=(1,3,3),
                               weight_initializer=init.Xavier(rnd_type='gaussian',factor_type='out',magnitude=2),bias_initializer='zero')
                    )
            # conv2_x
            self.features.add(BasicBlock(in_channel=64,out_channel=64,spatial_stride=1,temporal_stride=1)) # input size = 112*112
            self.features.add(BasicBlock(in_channel=64,out_channel=64,spatial_stride=1,temporal_stride=1))
            # conv3_x
            self.features.add(BasicBlock(in_channel=64,out_channel=128,spatial_stride=2,temporal_stride=2))
            self.features.add(BasicBlock(in_channel=128,out_channel=128,spatial_stride=1,temporal_stride=1))
            # conv4_x
            self.features.add(BasicBlock(in_channel=128,out_channel=256,spatial_stride=2,temporal_stride=2))
            self.features.add(BasicBlock(in_channel=256,out_channel=256,spatial_stride=1,temporal_stride=1))
            # conv5_x
            self.features.add(BasicBlock(in_channel=256,out_channel=512,spatial_stride=2,temporal_stride=2))
            self.features.add(BasicBlock(in_channel=512,out_channel=512,spatial_stride=1,temporal_stride=1))
            # avg pool
            self.features.add(nn.AvgPool3D(pool_size=(1,7,7)))
            

        self.output = nn.Dense(units=self.nclass, in_units=512,
                               weight_initializer=init.Normal(sigma=self.init_std))
              
    def hybrid_forward(self, F, x):            
        x = self.features(x)
        x = self.output(x)
        return x

def resnet18_3d(**kwargs):
    return get_simple(**kwargs)

def get_simple(pretrained=False,
               root='para', **kwargs):
    net = Res3D(**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
#        filepath = os.path.join(root,'simple%d.params'%(index))
#        print(filepath)
#        net.load_parameters(filepath)
        pass
    else:
        net.initialize()        
    return net