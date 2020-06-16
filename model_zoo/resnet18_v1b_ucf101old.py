#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-1-19 22:21:18
2020年03月02日17:44:20
@author: bdus

https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/action_recognition/actionrec_resnetv1b.py

x = nd.zeros(shape=(1,3,224,224))

"""
# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
import os
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo.resnetv1b import resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b, resnet152_v1b
from gluoncv.model_zoo import get_model
     
__all__ = ['resnet18_v1b_ucf101','resnet34_v1b_ucf101','resnet50_v1b_ucf101','resnet101_v1b_ucf101','resnet152_v1b_ucf101','resnet18_v1b_k400_ucf101','resnet34_v1b_k400_ucf101','resnet50_v1b_k400_ucf101','resnet101_v1b_k400_ucf101','resnet152_v1b_k400_ucf101']

class ActionRecResNetV1b(HybridBlock):
    r"""ResNet models for video action recognition
    Parameters
    ----------
    depth : int, number of layers in a ResNet model
    nclass : int, number of classes
    pretrained_base : bool, load pre-trained weights or not
    dropout_ratio : float, add a dropout layer to prevent overfitting on small datasets, such as UCF101
    init_std : float, standard deviation value when initialize the last classification layer
    feat_dim : int, feature dimension. Default is 4096 for VGG16 network
    num_segments : int, number of segments used
    num_crop : int, number of crops used during evaluation. Default choice is 1, 3 or 10
    Input: a single video frame or N images from N segments when num_segments > 1
    Output: a single predicted action label
    """
    def __init__(self, depth, nclass, pretrained_base=True,input_channel=3,
                 dropout_ratio=0.5, init_std=0.01,
                 feat_dim=2048, num_segments=1, num_crop=1,
                 partial_bn=False, **kwargs):
        super(ActionRecResNetV1b, self).__init__()

        if depth == 18:
            pretrained_model = resnet18_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 34:
            pretrained_model = resnet34_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 50:
            pretrained_model = resnet50_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 101:
            pretrained_model = resnet101_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 152:
            pretrained_model = resnet152_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 418:
            pretrained_model = get_model('resnet18_v1b_kinetics400',pretrained=True)
            self.expansion = 1
        elif depth == 434:
            pretrained_model = get_model('resnet34_v1b_kinetics400',pretrained=True)
            self.expansion = 1
        elif depth == 450:
            pretrained_model = get_model('resnet50_v1b_kinetics400',pretrained=True)
            self.expansion = 4
        elif depth == 501:
            pretrained_model = get_model('resnet101_v1b_kinetics400',pretrained=True)
            self.expansion = 4    
        elif depth == 552:
            pretrained_model = get_model('resnet152_v1b_kinetics400',pretrained=True)
            self.expansion = 4       
        else:
            print('No such ResNet configuration for depth=%d' % (depth))

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.feat_dim = 512 * self.expansion
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.input_channel = input_channel

        with self.name_scope():
            if self.input_channel == 3:
                self.conv1 = pretrained_model.conv1
            else:
                self.conv1 = nn.Conv2D(in_channels=input_channel,channels=64, kernel_size=7, strides=2,
                                       padding=3, use_bias=False,weight_initializer=mx.init.Xavier(magnitude=2))
                self.conv1.initialize()
            self.bn1 = pretrained_model.bn1
            self.relu = pretrained_model.relu
            self.maxpool = pretrained_model.maxpool
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
            self.avgpool = pretrained_model.avgpool
            self.flat = pretrained_model.flat
            self.drop = nn.Dropout(rate=self.dropout_ratio)
            self.output = nn.Dense(units=nclass, in_units=self.feat_dim,
                                   weight_initializer=init.Normal(sigma=self.init_std))
            self.output.initialize()

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.drop(x)

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.output(x)
        return x

def resnet18_v1b_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=18,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:        
        filepath = '0.7505-ucf101-resnet18_v1b_ucf101-085-best.params'
        filepath = os.path.join(root,filepath)
        filepath = os.path.expanduser(filepath)
        model.load_parameters(filepath,allow_missing=True)
        print(filepath)
    model.collect_params().reset_ctx(ctx)
    return model

def resnet34_v1b_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=34,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        pass
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=50,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        pass
    model.collect_params().reset_ctx(ctx)
    return model

def resnet101_v1b_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=101,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        pass
    model.collect_params().reset_ctx(ctx)
    return model

def resnet152_v1b_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=152,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        pass
    model.collect_params().reset_ctx(ctx)
    return model


def resnet18_v1b_k400_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=418,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        filepath = '0.8620-ucf101-resnet18_v1b_k400_ucf101-082-best.params'
        filepath = os.path.join(root,filepath)
        filepath = os.path.expanduser(filepath)
        model.load_parameters(filepath,allow_missing=True)
        print(filepath)
    model.collect_params().reset_ctx(ctx)
    return model

def resnet34_v1b_k400_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=434,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        pass
        #filepath = ''
        #filepath = os.path.join(root,filepath)
        #filepath = os.path.expanduser(filepath)
        #model.load_parameters(filepath,allow_missing=True)
        #print(filepath)
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_k400_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=450,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        pass
        #filepath = '0.8620-ucf101-resnet18_v1b_k400_ucf101-082-best.params'
        #filepath = os.path.join(root,filepath)
        #filepath = os.path.expanduser(filepath)
        #model.load_parameters(filepath,allow_missing=True)
        #print(filepath)
    model.collect_params().reset_ctx(ctx)
    return model

def resnet101_v1b_k400_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=501,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        pass
        #filepath = '0.8620-ucf101-resnet18_v1b_k400_ucf101-082-best.params'
        #filepath = os.path.join(root,filepath)
        #filepath = os.path.expanduser(filepath)
        #model.load_parameters(filepath,allow_missing=True)
        #print(filepath)
    model.collect_params().reset_ctx(ctx)
    return model

def resnet152_v1b_k400_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=552,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01,**kwargs)

    if pretrained:
        pass
        #filepath = '0.8620-ucf101-resnet18_v1b_k400_ucf101-082-best.params'
        #filepath = os.path.join(root,filepath)
        #filepath = os.path.expanduser(filepath)
        #model.load_parameters(filepath,allow_missing=True)
        #print(filepath)
    model.collect_params().reset_ctx(ctx)
    return model
