#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:10:42 2019

@author: bdus

see : https://github.com/bdus/hyperspectral/blob/master/indian_pines/indian_semi/symbols/symbols.py

"""
from .simple import *
from .MSEloss_vgg import *
from .test_1E4 import *
from .actionrec_inceptionv3 import *
from .inceptionv3_LSTM import *
from .inception_v3_k400ft import *
from .C3D import *
from .Res3D import *
from .F_stCN import *
from .R21D import *
from .ECO import *
from .resnet18_v1b_ucf101 import *
#from .DualBlock import *
from .mx_c3d import *
from .r2plus1d import *
from .mx_c3d_base import *
from .r2plus1d_base import *
from .mx_i3d_resnet import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'simple':simple,
    'mseloss_vgg16':mseloss_vgg16,
    'dualnet_max':dualnet_max,
    'dualnet_avg':dualnet_avg,
    'dualnet_outmax':dualnet_outmax,
    'dualnet_outavg':dualnet_outavg,
    'inceptionv3_ucf101_sim':inceptionv3_ucf101_sim,
    'inceptionv3_kinetics400_sim':inceptionv3_kinetics400_sim,
    'inceptionv3_ucf101_lstm':inceptionv3_ucf101_lstm,
    'inceptionv3_ucf101_k400ft':inceptionv3_ucf101_k400ft,
    'c3d':c3d,
    'resnet18_3d':resnet18_3d,
    'fstcn_3d':fstcn_3d,
    'res21d_34_3d':res21d_34_3d,
    'eco_resnet18_v2':eco_resnet18_v2,
    'eco_resnet18_v1b':eco_resnet18_v1b,
    'resnet18_v1b_ucf101':resnet18_v1b_ucf101,
    'eco_resnet18_v1b_k400':eco_resnet18_v1b_k400,
    'eco_resnet18_v1b_k400_ucf101':eco_resnet18_v1b_k400_ucf101,
    'resnet18_v1b_k400_ucf101':resnet18_v1b_k400_ucf101,
    'resnet34_v1b_ucf101':resnet34_v1b_ucf101,
    'resnet50_v1b_ucf101':resnet50_v1b_ucf101,
    'resnet101_v1b_ucf101':resnet101_v1b_ucf101,
    'resnet152_v1b_ucf101':resnet152_v1b_ucf101,
    'resnet34_v1b_k400_ucf101':resnet34_v1b_k400_ucf101,
    'resnet50_v1b_k400_ucf101':resnet50_v1b_k400_ucf101,
    'resnet101_v1b_k400_ucf101':resnet101_v1b_k400_ucf101,
    'resnet152_v1b_k400_ucf101':resnet152_v1b_k400_ucf101,
    'eco_resnet34_v1b':eco_resnet34_v1b,
    'eco_resnet34_v1b_k400':eco_resnet34_v1b_k400,
    'eco_resnet50_v1b':eco_resnet50_v1b,
    'eco_resnet50_v1b_k400':eco_resnet50_v1b_k400,
    'eco_resnet101_v1b':eco_resnet101_v1b,
    'eco_resnet101_v1b_k400':eco_resnet101_v1b_k400,
    'eco_resnet152_v1b':eco_resnet152_v1b,
    'eco_resnet152_v1b_k400':eco_resnet152_v1b_k400,
    'c3d_kinetics400':c3d_kinetics400, 
    'r2plus1d_resnet18_kinetics400':r2plus1d_resnet18_kinetics400, 
    'r2plus1d_resnet34_kinetics400':r2plus1d_resnet34_kinetics400,
    'r2plus1d_resnet50_kinetics400':r2plus1d_resnet50_kinetics400,
    'r2plus1d_resnet101_kinetics400':r2plus1d_resnet101_kinetics400,
    'r2plus1d_resnet152_kinetics400':r2plus1d_resnet152_kinetics400,
    'c3d_kinetics400_ucf101':c3d_kinetics400_ucf101,
    'r2plus1d_resnet18_kinetics400_custom':r2plus1d_resnet18_kinetics400_custom,
    'r2plus1d_resnet34_kinetics400_custom':r2plus1d_resnet34_kinetics400_custom,
    'c3d_kinetics400_custome':c3d_kinetics400_custome,
    'r2plus1d_resnet34_tranconv_lateral':r2plus1d_resnet34_tranconv_lateral,
    'r2plus1d_resnet34_tranconv_lateral_tanhbn':r2plus1d_resnet34_tranconv_lateral_tanhbn,
    'r2plus1d_resnet18_aetfc':r2plus1d_resnet18_aetfc,
    'i3d_resnet50_v1_kinetics400':i3d_resnet50_v1_kinetics400,
    'i3d_resnet101_v1_kinetics400':i3d_resnet101_v1_kinetics400,
    'i3d_resnet50_v1_ucf101':i3d_resnet50_v1_ucf101,
    #'i3d_resnet50_v1_custom':i3d_resnet50_v1_custom,
    'i3d_resnet50_v1_hmdb51':i3d_resnet50_v1_hmdb51
    }

def get_model(name, **kwargs):
    """Returns a pre-defined model by name
    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net

def get_model_list():
    """Get the entire list of model names in model_zoo.
    Returns
    -------
    list of str
        Entire list of model names in model_zoo.
    """
    return _models.keys()
