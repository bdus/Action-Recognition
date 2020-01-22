#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:48:26 2020

@author: bdus

eco


reference :
    https://github.com/jangho2001us/pytorch_eco/blob/master/resnet_3d.py
    
    https://data.lip6.fr/cadene/pretrainedmodels/
    https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/inception-bn.py
        
x = nd.zeros(shape=(N,16,3,224,224))

y = net(x.transpose(axes=(0,2,1,3,4)))

"""
import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
#from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo import get_model


__all__ = ['eco_resnet18_v2','eco_resnet18_v1b','eco_resnet18_v1b_k400']


class BasicBlock(nn.HybridBlock):
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

#class base_resnet18_v1b(nn.HybridBlock):
#    def __init__(self,pretrained=True,batch_normal=True, dropout_ratio=0.8, init_std=0.001,**kwargs):
#        super(base_resnet18_v1b, self).__init__()
#        self.net = get_model('resnet18_v1b',pretrained=pretrained)    
#    def hybrid_forward(self, F, x):
#        #x = nd.zeros(shape=(1,3,224,224))
#        t = self.net.conv1(x)
#        t = self.net.bn1(t)
#        t = self.net.relu(t)
#        t = self.net.maxpool(t)
#        t = self.net.layer1(t)
#        t = self.net.layer2(t)
#        # t.shape (1, 128, 28, 28)
#        return t
#
#class base_resnet18_v2(nn.HybridBlock):
#    def __init__(self,pretrained=True,batch_normal=True, dropout_ratio=0.8, init_std=0.001,**kwargs):
#        super(base_resnet18_v2, self).__init__()
#        self.net = get_model('resnet18_v2',pretrained=pretrained)    
#    def hybrid_forward(self, F, x):
#        for i in range(7):
#            x = self.net.features[i](x)        
#        return x
    
class ECO(nn.HybridBlock):   
    def __init__(self,nclass,base_model='resnet18_v1b',num_segments=8,input_channel=3,batch_normal=True, dropout_ratio=0.8, init_std=0.001,**kwargs):
        super(ECO, self).__init__()
        self.nclass = nclass        
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std
        self.num_segments = num_segments
        self.input_shape = 224
        self.base_model = base_model#['resnet18_v1b','resnet18_v2','resnet18_v1b_kinetics400'][1]
        
        pretrained_model = get_model(self.base_model,pretrained=True)
        
        with self.name_scope():
            # x = nd.zeros(shape=(7x8,3,224,224))
            #2D feature
            if self.base_model == 'resnet18_v2':
                self.feature2d = pretrained_model.features
            else: #'resnet18_v1b' in self.base_model:
                self.conv1 = pretrained_model.conv1
                self.bn1 = pretrained_model.bn1
                self.relu = pretrained_model.relu
                self.conv1 = pretrained_model.conv1
                self.maxpool = pretrained_model.maxpool
                self.layer1 = pretrained_model.layer1
                self.layer2 = pretrained_model.layer2

            #3D feature 
            self.features_3d = nn.HybridSequential(prefix='')
            # conv3_x
            self.features_3d.add(BasicBlock(in_channel=128,out_channel=128,spatial_stride=1,temporal_stride=1))
            self.features_3d.add(BasicBlock(in_channel=128,out_channel=128,spatial_stride=1,temporal_stride=1))
            # conv4_x
            self.features_3d.add(BasicBlock(in_channel=128,out_channel=256,spatial_stride=2,temporal_stride=2))
            self.features_3d.add(BasicBlock(in_channel=256,out_channel=256,spatial_stride=1,temporal_stride=1))
            # conv5_x
            self.features_3d.add(BasicBlock(in_channel=256,out_channel=512,spatial_stride=2,temporal_stride=2))
            self.features_3d.add(BasicBlock(in_channel=512,out_channel=512,spatial_stride=1,temporal_stride=1))
            self.features_3d.add(nn.AvgPool3D(pool_size=(1,7,7)))
            self.output = nn.HybridSequential(prefix='')
            self.output.add( nn.Dense(units=512, in_units=1024,
                               weight_initializer=init.Normal(sigma=self.init_std)),
                             nn.Dense(units=self.nclass, in_units=512,
                               weight_initializer=init.Normal(sigma=self.init_std)) )
            # init
            self.features_3d.initialize()
            self.output.initialize()

    def hybrid_forward(self, F, x):
        #2d
        if self.base_model == 'resnet18_v2':
            for i in range(7):
                x = self.feature2d[i](x)
        else: #resnet18_v1b
            #x = nd.zeros(shape=(1,3,224,224))
            t = self.conv1(x)
            t = self.bn1(t)
            t = self.relu(t)
            t = self.maxpool(t)
            t = self.layer1(t)
            x = self.layer2(t)
            # t.shape (1, 128, 28, 28)
        # reshape
        x = x.reshape((-1,self.num_segments,128,28,28))
        x = x.transpose(axes=(0,2,1,3,4))
        # 3d
        x = self.features_3d(x)
        x = F.flatten(x)       
        x = self.output(x)
        return x

def eco_resnet18_v1b_k400(base_model='resnet18_v1b_kinetics400',**kwargs):
    return get_simple(**kwargs)

    
def eco_resnet18_v1b(base_model='resnet18_v1b',**kwargs):
    return get_simple(**kwargs)

def eco_resnet18_v2(base_model='resnet18_v2',**kwargs):
    return get_simple(**kwargs)

def get_simple(pretrained=False,
               root='para', **kwargs):
    net = ECO(**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
#        filepath = os.path.join(root,'simple%d.params'%(index))
#        print(filepath)
#        net.load_parameters(filepath)
        pass
    return net