#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:48:26 2020

@author: bdus

eco
from  mxnet import nd
from model_zoo import get_model as myget

net = myget(name='eco_resnet18_v1b_k400',nclass=101,num_segments=32,input_channel=3,batch_normal=False)
X = nd.zeros(shape=(5,32,3,224,224))
X = X.reshape((-1,) + X.shape[2:])
net(X).shape == (5,101)

reference :
    https://github.com/jangho2001us/pytorch_eco/blob/master/resnet_3d.py
    
    https://data.lip6.fr/cadene/pretrainedmodels/
    https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/inception-bn.py


	t= nd.zeros(shape=(5,segment,3,224,224))
	t = t.reshape((-1,) + t.shape[2:])

N=8时候  [512,1024]
N=16时候 shape=(512,2048)

import mxnet as mx
from model_zoo import get_model as myget
from mxnet import nd ,init
from mxnet.gluon import nn


basemodel = 'resnet18_v1b'
basemodel = 'resnet34_v1b'
basemodel = 'resnet18_v1b_ucf101'
basemodel = 'resnet34_v1b_ucf101'
basemodel = 'resnet18_v1b_k400_ucf101'
basemodel = 'resnet34_v1b_k400_ucf101'


basemodel = 'resnet50_v1b'
basemodel = 'resnet101_v1b'
basemodel = 'resnet152_v1b'
basemodel = 'resnet50_v1b_ucf101'
basemodel = 'resnet101_v1b_ucf101'
basemodel = 'resnet152_v1b_ucf101'
basemodel = 'resnet50_v1b_k400_ucf101'
basemodel = 'resnet101_v1b_k400_ucf101'
basemodel = 'resnet152_v1b_k400_ucf101'

def printmodel(basemodel,segment=4,expo=1):
	t= nd.zeros(shape=(5,segment,3,224,224))
	t = t.reshape((-1,) + t.shape[2:])
	basenet = myget(name=basemodel,nclass=101,num_segments=1,input_channel=3,batch_normal=False)
	basenet.initialize()
	t = basenet.conv1(t)
	print("conv1:",t.shape)
	t = basenet.bn1(t)
	t = basenet.relu(t)
	t = basenet.maxpool(t)
	print("maxpool:",t.shape)
	t = basenet.layer1(t)
	print("layer1:",t.shape)
	t = basenet.layer2(t)
	print("layer2:",t.shape)
	t = t.reshape((-1,segment,128*expo,28,28))
	print("reshape:",t.shape)
	t = t.transpose(axes=(0,2,1,3,4))
	print("transpose:",t.shape)


printmodel('resnet50_v1b_ucf101',4,4)
printmodel('resnet18_v1b_ucf101',4,1)
printmodel('resnet50_v1b_ucf101',8,4)
printmodel('resnet18_v1b_ucf101',8,1)
printmodel('resnet50_v1b_ucf101',16,4)
printmodel('resnet18_v1b_ucf101',16,1)

printmodel('resnet50_v1b_ucf101',32,4)
printmodel('resnet18_v1b_ucf101',32,1)



def getf3d(exp=1,temp=1,avgtmp=1):
	f3d = nn.HybridSequential(prefix='')
	# conv3_x
	f3d.add(BasicBlock(in_channel=128*exp,out_channel=128,spatial_stride=1,temporal_stride=temp))
	f3d.add(BasicBlock(in_channel=128,out_channel=128,spatial_stride=1,temporal_stride=1))
	# conv4_x
	f3d.add(BasicBlock(in_channel=128,out_channel=256,spatial_stride=2,temporal_stride=2))
	f3d.add(BasicBlock(in_channel=256,out_channel=256,spatial_stride=1,temporal_stride=1))
	# conv5_x
	f3d.add(BasicBlock(in_channel=256,out_channel=512,spatial_stride=2,temporal_stride=2))
	f3d.add(BasicBlock(in_channel=512,out_channel=512,spatial_stride=1,temporal_stride=1))
	f3d.add(nn.AvgPool3D(pool_size=(avgtmp,7,7)))
	f3d.initialize()
	return f3d

f3d = getf3d(1,1)
f3d = getf3d(1,2)
f3d = getf3d(4,1)
f3d = getf3d(4,2)

f3d = getf3d(1,1,2)
f3d = getf3d(1,2,2)
f3d = getf3d(4,1,2)
f3d = getf3d(4,2,2)

print("features_3d:",f3d(nd.zeros(shape=(5,128,4,28,28))).shape)
print("features_3d:",f3d(nd.zeros(shape=(5,128,8,28,28))).shape)
print("features_3d:",f3d(nd.zeros(shape=(5,128,16,28,28))).shape)
print("features_3d:",f3d(nd.zeros(shape=(5,128,32,28,28))).shape)

print("features_3d:",f3d(nd.zeros(shape=(5,512,4,28,28))).shape)
print("features_3d:",f3d(nd.zeros(shape=(5,512,8,28,28))).shape)
print("features_3d:",f3d(nd.zeros(shape=(5,512,16,28,28))).shape)
print("features_3d:",f3d(nd.zeros(shape=(5,512,32,28,28))).shape)


"""
import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
#from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo import get_model

from .r2plus1d import conv3x1x1,Conv2Plus1D
from .r2plus1d import BasicBlock as BasicBlock_2Plus1D

__all__ = ['eco_resnet18_v2','eco_resnet18_v1b','eco_resnet18_v1b_k400','eco_resnet34_v1b','eco_resnet34_v1b_k400','eco_resnet50_v1b','eco_resnet50_v1b_k400','eco_resnet101_v1b','eco_resnet101_v1b_k400','eco_resnet152_v1b','eco_resnet152_v1b_k400','eco_resnet18_v1b_k400_ucf101']


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
    def __init__(self,nclass,base_model='resnet18_v1b',pretrained_base=True,num_segments=8,num_temporal=1,ifTSN=True,input_channel=3,batch_normal=True, dropout_ratio=0.8, init_std=0.001,**kwargs):
        super(ECO, self).__init__()
        self.nclass = nclass        
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std
        self.num_segments = num_segments
        self.ifTSN = ifTSN
        self.input_shape = 224
        self.base_model = base_model#['resnet18_v1b','resnet18_v2','resnet18_v1b_kinetics400','resnet18_v1b_k400_ucf101'][1]
        
        # resnet50 101 152 的 self.expansion == 4
        #self.expansion = 4 if ('resnet50_v1b' in self.base_model)or('resnet101_v1b' in self.base_model)or('resnet152_v1b' in self.base_model) else 1      
        
        if 'resnet18_v1b' in self.base_model:
            self.expansion = 1
        elif 'resnet34_v1b' in self.base_model:
            self.expansion = 1
        elif 'resnet50_v1b' in self.base_model:
            self.expansion = 4
        elif 'resnet101_v1b' in self.base_model:
            self.expansion = 4
        elif 'resnet152_v1b' in self.base_model:
            self.expansion = 4
        else:
            self.expansion = 1
        
        #2d 卷积的出来的维度
        self.feat_dim_2d = 128 * self.expansion
        
        # num_temporal 默认为1 论文中 一开始不减少时间维
        self.num_temporal = num_temporal
        if self.num_segments == 4:            
            self.num_temporal=1
        elif self.num_segments == 8:
            self.num_temporal=num_temporal
        elif self.num_segments == 16:
            self.num_temporal=num_temporal
        elif self.num_segments == 32:
            self.num_temporal=num_temporal
        else:
            self.num_temporal=1
        
        # 输入fc的维度
        if self.ifTSN == True:
            self.feat_dim_3d = 512
        else: # Flatten
            tmppara = self.num_segments // 4
            tmppara = tmppara // (self.num_temporal if tmppara > 1 else 1)
            self.feat_dim_3d = 512 * tmppara
            
        
        pretrained_model = get_model(self.base_model,pretrained=pretrained_base)
        
        
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
            self.features_3d.add(BasicBlock(in_channel=self.feat_dim_2d,out_channel=128,spatial_stride=1,temporal_stride=self.num_temporal))
            self.features_3d.add(BasicBlock(in_channel=128,out_channel=128,spatial_stride=1,temporal_stride=1))
            # conv4_x
            self.features_3d.add(BasicBlock(in_channel=128,out_channel=256,spatial_stride=2,temporal_stride=2))
            self.features_3d.add(BasicBlock(in_channel=256,out_channel=256,spatial_stride=1,temporal_stride=1))
            # conv5_x
            self.features_3d.add(BasicBlock(in_channel=256,out_channel=512,spatial_stride=2,temporal_stride=2))
            self.features_3d.add(BasicBlock(in_channel=512,out_channel=512,spatial_stride=1,temporal_stride=1))
            self.features_3d.add(nn.AvgPool3D(pool_size=(1,7,7)))
            self.dropout = nn.Dropout(rate=self.dropout_ratio)
            self.output = nn.HybridSequential(prefix='')
            if self.ifTSN == True:
                self.output.add( nn.Dense(units=self.nclass, in_units=512,
                               weight_initializer=init.Normal(sigma=self.init_std)) )
            else:
                self.output.add( nn.Dense(units=512, in_units=self.feat_dim_3d,
                               weight_initializer=init.Normal(sigma=self.init_std)),
                             nn.Dense(units=self.nclass, in_units=512,
                               weight_initializer=init.Normal(sigma=self.init_std)) ) 
            # init
            self.features_3d.initialize(init.MSRAPrelu())
            self.output.initialize(init.MSRAPrelu())

    def hybrid_forward(self, F, x):
        #2d
        if self.base_model == 'resnet18_v2':
            for i in range(7):
                x = self.feature2d[i](x)
        else: #resnet18_v1b
            #x = nd.zeros(shape=(N*numsegment,3,224,224)) N=5 numseg=8
            t = self.conv1(x) #conv1: (40, 64, 112, 112)
            t = self.bn1(t)
            t = self.relu(t)
            t = self.maxpool(t)#maxpool: (40, 64, 56, 56)
            t = self.layer1(t)#layer1: (40, 64, 56, 56)
            x = self.layer2(t)#layer2: (40, 64, 56, 56)
            # t.shape (1, 128, 28, 28)
        # reshape
        x = x.reshape((-1,self.num_segments,self.feat_dim_2d,28,28)) #reshape: (5, 8, 128 * self.expansion, 28, 28)
        x = x.transpose(axes=(0,2,1,3,4)) #transpose: (5, 128 * self.expansion, 8, 28, 28)
        # 3d
        x = self.features_3d(x)
        
                
        if self.ifTSN == True:
            # segmental consensus
            x = F.mean(x, axis=2)
        else:
            x = F.flatten(x)        
           
        x = self.output(self.dropout(x))
        return x



def eco_resnet18_v2(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet18_v2',**kwargs)
    if pretrained:
        pass
    return net

def eco_resnet18_v1b_k400_ucf101(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet18_v1b_kinetics400',**kwargs)
    if pretrained:
        filepath = '0.6349-ucf101-eco_resnet18_v1b_k400_ucf101-068-best.params'
        filepath = os.path.join(root,filepath)
        filepath = os.path.expanduser(filepath)
        net.load_parameters(filepath,allow_missing=True)
        print(filepath)
    return net

#

def eco_resnet18_v1b_k400(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet18_v1b_kinetics400',**kwargs)
    if pretrained:
        pass
    return net

def eco_resnet18_v1b(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet18_v1b',**kwargs)
    if pretrained:
        pass
    return net

def eco_resnet34_v1b_k400(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet34_v1b_kinetics400',**kwargs)
    if pretrained:
        pass
    return net

def eco_resnet34_v1b(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet34_v1b',**kwargs)
    if pretrained:
        pass
    return net

# 
def eco_resnet50_v1b_k400(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet50_v1b_kinetics400',**kwargs)
    if pretrained:
        pass
    return net

def eco_resnet50_v1b(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet50_v1b',**kwargs)
    if pretrained:
        pass
    return net

# 
def eco_resnet101_v1b_k400(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet101_v1b_kinetics400',**kwargs)
    if pretrained:
        pass
    return net

def eco_resnet101_v1b(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet101_v1b',**kwargs)
    if pretrained:
        pass
    return net

#
def eco_resnet152_v1b_k400(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet152_v1b_kinetics400',**kwargs)
    if pretrained:
        pass
    return net

def eco_resnet152_v1b(pretrained=False,
               root='~/.mxnet/models', **kwargs):
    net = ECO(base_model='resnet152_v1b',**kwargs)
    if pretrained:
        pass
    return net


