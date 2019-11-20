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

__all__ = ['get_model', 'get_model_list']

_models = {
    'simple':simple,
    'mseloss_vgg16':mseloss_vgg16,
    'dualnet_max':dualnet_max,
    'dualnet_avg':dualnet_avg
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