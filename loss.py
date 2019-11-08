#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:58:33 2019

@author: bdus

define a new loss

"""

import numpy as np
import mxnet as mx

from mxnet import gluon, nd

from mxnet.gluon import nn
from mxnet.gluon import loss


class MyLoss(loss.Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(MyLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.np.square(label - pred) if is_np_array() else F.square(label - pred)
        loss = _apply_weighting(F, loss, self._weight / 2, sample_weight)
        if is_np_array():
            if F is ndarray:
                return F.np.mean(loss, axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return F.mean(loss, axis=self._batch_axis, exclude=True)
