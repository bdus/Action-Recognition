# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
"""
2020年06月16日15:38:10
基于LSTM的没有训练起来 

"""
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo.inception import inception_v3, _make_basic_conv

__all__ = ['inceptionv3_ucf101_lstm']

class InceptionV3_LSTM(HybridBlock):
    r"""InceptionV3 model for video action recognition
    Christian Szegedy, etal, Rethinking the Inception Architecture for Computer Vision, CVPR 2016
    https://arxiv.org/abs/1512.00567
    Limin Wang, etal, Towards Good Practices for Very Deep Two-Stream ConvNets, arXiv 2015
    https://arxiv.org/abs/1507.02159
    Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016
    https://arxiv.org/abs/1608.00859

    Parameters
    ----------
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
    def __init__(self, nclass, pretrained_base=True,input_channel=3,num_layers=5,
                 partial_bn=True, dropout_ratio=0.8, init_std=0.001,
                 feat_dim=2048, num_segments=1, num_crop=1, **kwargs):
        super(InceptionV3_LSTM, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_dim = feat_dim        

        pretrained_model = inception_v3(pretrained=pretrained_base, partial_bn=partial_bn, **kwargs)
        inception_features = pretrained_model.features        
        if input_channel == 3:
            self.features = inception_features
        else:
            self.features = nn.HybridSequential(prefix='')
            with pretrained_model.name_scope():
                if 'norm_layer' not in dir():                    
                    norm_layer = nn.BatchNorm
                else:
                    if norm_layer is None:
                        norm_layer = nn.BatchNorm                        
                self.features.add(_make_basic_conv(in_channels=input_channel,channels=32, 
                                kernel_size=3, strides=2,norm_layer=norm_layer,norm_kwargs=None,
                                weight_initializer=mx.init.Xavier(magnitude=2)))
                self.features[0].initialize()
                for layer in inception_features[1:]:
                    self.features.add(layer)
                    
        def update_dropout_ratio(block):
            if isinstance(block, nn.basic_layers.Dropout):
                block._rate = self.dropout_ratio
        self.apply(update_dropout_ratio)
        self.output = nn.Dense(units=nclass, in_units=self.feat_dim,
                               weight_initializer=init.Normal(sigma=self.init_std))
        self.output.initialize()
        
        self.rnn = mx.gluon.rnn.LSTM(hidden_size=self.feat_dim, input_size=self.feat_dim,
                                        num_layers=num_layers, layout='NTC',
                                        h2h_weight_initializer=init.Normal(sigma=self.init_std),
                                        i2h_weight_initializer=init.Normal(sigma=self.init_std),
                                        h2r_weight_initializer=init.Normal(sigma=self.init_std)
                                        )
        self.rnn.initialize()
        

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = F.squeeze(x)
        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = self.rnn(x)
        x = F.mean(x, axis=1)
        x = self.output(x)
        return x

def inceptionv3_ucf101_lstm(nclass=101, pretrained=False, pretrained_base=True,
                       use_tsn=False, num_segments=1, num_crop=1, partial_bn=True,
                       ctx=mx.cpu(), root='~/.mxnet/models', **kwargs):
    model = InceptionV3_LSTM(nclass=nclass,
                                 partial_bn=partial_bn,
                                 pretrained_base=pretrained_base,
                                 num_segments=num_segments,
                                 num_crop=num_crop,
                                 dropout_ratio=0.8,
                                 init_std=0.001,**kwargs)

    if pretrained:
        pass
#        from ..model_store import get_model_file
#        model.load_parameters(get_model_file('inceptionv3_ucf101',
#                                             tag=pretrained, root=root))
#        from ...data import UCF101Attr
#        attrib = UCF101Attr()
#        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model
#
#def inceptionv3_kinetics400_sim(nclass=400, pretrained=False, pretrained_base=True,
#                            tsn=False, num_segments=1, num_crop=1, partial_bn=True,
#                            ctx=mx.cpu(), root='~/.mxnet/models', **kwargs):
#    model = InceptionV3_LSTM(nclass=nclass,
#                                 partial_bn=partial_bn,
#                                 pretrained_base=pretrained_base,
#                                 num_segments=num_segments,
#                                 num_crop=num_crop,
#                                 dropout_ratio=0.5,
#                                 init_std=0.01,**kwargs)
#
#    if pretrained:
#        from ..model_store import get_model_file
#        model.load_parameters(get_model_file('inceptionv3_kinetics400',
#                                             tag=pretrained, root=root))
#        from ...data import Kinetics400Attr
#        attrib = Kinetics400Attr()
#        model.classes = attrib.classes
#    model.collect_params().reset_ctx(ctx)
#    return model
