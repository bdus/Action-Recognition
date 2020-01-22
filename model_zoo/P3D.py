#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:04:55 2019
@author: bdus

P3D

ori paper acc on ucf101 : 

reference :
    https://github.com/MRzzm/action-recognition-models-pytorch
    https://github.com/qijiezhao/pseudo-3d-pytorch
        
x = nd.zeros(shape=(N,16,3,160,160))
y = net(x.transpose(axes=(0,2,1,3,4)))

"""
import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
#from mxnet.gluon.nn import HybridBlock

__all__ = ['p3d']

class P3D_block(nn.HybridBlock):
    def __init__(self, blockType,inplanes,planes,stride=1):
        super(P3D_block,self).__init__()
        self.expansion = 4
        self.stride = stride
        self.blockType = blockType
        self.conv1 = nn.Conv3D(in_channels=inplanes, channels=planes, kernel_size=(1,1,1),use_bias=False)
        self.bn1 = nn.BatchNorm(in_channels=planes)
        if self.blockType == 'A':
            self.conv2D = nn.Conv3D(in_channels=planes, channels=planes, kernel_size=(1,3,3),strides=(1,stride,stride),padding=(0,1,1),use_bias=False)
            self.conv1D = nn.Conv3D(in_channels=planes, channels=planes, kernel_size=(3,1,1),strides=(stride,1,1),padding=(1,0,0),use_bias=False)
        elif self.blockType == 'B':
            self.conv2D = nn.Conv3D(in_channels=planes, channels=planes, kernel_size=(1,3,3),strides=stride,padding=(0,1,1),use_bias=False)
            self.conv1D = nn.Conv3D(in_channels=planes, channels=planes, kernel_size=(3,1,1),strides=stride,padding=(1,0,0),use_bias=False)
        else:
            self.conv2D = nn.Conv3D(in_channels=planes, channels=planes, kernel_size=(1,3,3),strides=stride,padding=(0,1,1),use_bias=False)
            self.conv1D = nn.Conv3D(in_channels=planes, channels=planes, kernel_size=(3,1,1),strides=1,padding=(1,0,0),use_bias=False)
        self.bn2D = nn.BatchNorm(in_channels=planes)
        self.bn1D = nn.BatchNorm(in_channels=planes)
        self.conv3 = nn.Conv3D(in_channels=planes, channels=planes*self.expansion, kernel_size=(1,1,1),use_bias=False)
        self.bn3 = nn.BatchNorm(in_channels=planes*self.expansion)
        self.relu = nn.Activation('relu')
        
        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.HybridSequential()
            self.downsample.add(
                    nn.Conv3D(in_channels=inplanes, channels=planes*self.expansion, kernel_size=(1,1,1),strides=stride, use_bias=False),
                    nn.BatchNorm(in_channels=planes * self.expansion) )
        else:
            self.downsample = None
            
            
    def hybrid_forward(self, F, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)
        
        if self.blockType == 'A':
            x_branch = self.conv2D(x_branch)
            x_branch = self.bn2D(x_branch)
            x_branch = self.relu(x_branch)
            x_branch = self.conv1D(x_branch)
            x_branch = self.bn1D(x_branch)
            x_branch = self.relu(x_branch)
        elif self.blockType == 'B':
            x_branch2D = self.conv2D(x_branch)
            x_branch2D = self.bn2D(x_branch2D)
            x_branch2D = self.relu(x_branch2D)
            
            x_branch1D = self.conv1D(x_branch)
            x_branch1D = self.bn1D(x_branch1D)
            x_branch = x_branch1D + x_branch2D
            x_branch = self.relu(x_branch)
        else:
            x_branch = self.conv2D(x_branch)
            x_branch = self.bn2D(x_branch)
            x_branch = self.relu(x_branch)
            x_branch1D = self.conv1D(x_branch)
            x_branch1D = self.bn1D(x_branch1D)            
            x_branch = x_branch1D + x_branch2D
            x_branch = self.relu(x_branch)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return self.relu(x_branch+x)
       

class P3D(nn.HybridBlock):
    #
    def __init__(self,nclass,input_channel=3,batch_normal=True, dropout_ratio=0.8, init_std=0.001,**kwargs):
        super(P3D, self).__init__()
        self.nclass = nclass        
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std
        self.expansion = 1

        with self.name_scope():
            self.conv1 = nn.Conv3D(in_channels=input_channel, channels=64, kernel_size=(1,7,7),strides=(1,2,2),padding=(0,3,3),use_bias=False)
            self.bn1 = nn.BatchNorm(in_channels=64)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool3D(pool_size=(1,3,3),strides=(1,2,2),padding=(0,1,1))
            self.conv2 = nn.HybridSequential()
            self.conv2.add(
                    P3D_block('A',64,64*self.expansion,2),
                    P3D_block('B',64*self.expansion,64*self.expansion),
                    P3D_block('C',64*self.expansion,64*self.expansion))
            self.conv3 = nn.HybridSequential()
            self.conv3.add(
                    P3D_block('A',64*self.expansion,128*self.expansion,2),
                    P3D_block('B',128*self.expansion,128*self.expansion),
                    P3D_block('C',128*self.expansion,128*self.expansion),
                    P3D_block('A',128*self.expansion,128*self.expansion))
            self.conv4 = nn.HybridSequential()
            self.conv4.add(
                    P3D_block('B',128*self.expansion,256*self.expansion,2),
                    P3D_block('C',256*self.expansion,256*self.expansion),
                    P3D_block('A',256*self.expansion,256*self.expansion),
                    P3D_block('B',256*self.expansion,256*self.expansion),
                    P3D_block('C',256*self.expansion,256*self.expansion),
                    P3D_block('A',256*self.expansion,256*self.expansion) )
            self.conv5 = nn.HybridSequential()
            self.conv5.add(
                    P3D_block('B',256*self.expansion,512*self.expansion,2),
                    P3D_block('C',512*self.expansion,512*self.expansion),
                    P3D_block('A',512*self.expansion,512))
            self.avg_pool = nn.AvgPool3D(pool_size=(1,3,3))
            self.output = nn.Dense(in_units=512,units=nclass,weight_initializer=init.Normal(sigma=init_std))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = self.output(x)
        return x

def p3d(**kwargs):
    return get_simple(**kwargs)

def get_simple(pretrained=False,
               root='para', **kwargs):
    net = P3D(**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
#        filepath = os.path.join(root,'simple%d.params'%(index))
#        print(filepath)
#        net.load_parameters(filepath)
        pass
    else:
        net.initialize()        
    return net