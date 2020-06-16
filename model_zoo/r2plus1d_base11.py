# pylint: disable=arguments-differ,unused-argument,line-too-long
"""R2Plus1D, implemented in Gluon. https://arxiv.org/abs/1711.11248.
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py.

https://github.com/dmlc/gluon-cv/blob/6e2cdd608bd4df0a1552d81371ca7f458de0db25/gluoncv/model_zoo/action_recognition/i3d_resnet.py

2020年3月7日22:40:13


>>> x = nd.zeros(shape=(5,3,16,112,112))
conv1 :  (5, 45, 16, 56, 56)
conv2 :  (5, 64, 16, 56, 56)
layer1 :  (5, 64, 16, 56, 56)
layer2 :  (5, 128, 8, 28, 28)
layer3 :  (5, 256, 4, 14, 14)
layer4 :  (5, 512, 2, 7, 7)

x = nd.zeros(shape=(5, 512, 2, 7, 7))
up_layer4 :  (5, 256, 2, 7, 7)
up_sample4 :  (5, 256, 4, 14, 14) 
up_layer3 :  (5, 128, 4, 14, 14)
up_sample3 :  (5, 128, 8, 28, 28)
up_layer2 :  (5, 64, 8, 28, 28)
up_sample2 :  (5, 64, 16, 56, 56)
up_layer1 :  (5, 64, 18, 56, 56)
up_sample1 :  (5, 64, 16, 112, 112)
up_conv2 :  (5, 45, 16, 112, 112)
up_conv1 :  (5, 3, 16, 112, 112)




up_layer4 = nn.Conv3DTranspose(in_channels=512, channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
up_layer4.initialize()

up_sample4 = nn.Conv3DTranspose(in_channels=256, channels=256, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
use_bias=False,  weight_initializer=mx.init.Bilinear()) 
up_sample4.initialize()

up_layer3 = nn.Conv3DTranspose(in_channels=256, channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
up_layer3.initialize()

up_sample3 = nn.Conv3DTranspose(in_channels=128, channels=128, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
use_bias=False,  weight_initializer=mx.init.Bilinear()) 
up_sample3.initialize()

up_layer2 = nn.Conv3DTranspose(in_channels=128, channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
up_layer2.initialize()

up_sample2 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
use_bias=False,  weight_initializer=mx.init.Bilinear()) 
up_sample2.initialize()

up_layer1 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1)) 
up_layer1.initialize()
up_sample1 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(1,2,2), strides=(1,2,2), padding=(1,0,0),
use_bias=False,  weight_initializer=mx.init.Bilinear()) 
up_sample1.initialize()

up_conv2 = nn.Conv3DTranspose(in_channels=64, channels=45, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
up_conv2.initialize()

up_conv1 = nn.Conv3DTranspose(in_channels=45, channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
up_conv1.initialize()





"""

from .r2plus1d import r2plus1d_resnet18_kinetics400,r2plus1d_resnet34_kinetics400,r2plus1d_resnet50_kinetics400,r2plus1d_resnet101_kinetics400,r2plus1d_resnet152_kinetics400

__all__ = ['r2plus1d_resnet18_kinetics400_custom','r2plus1d_resnet34_kinetics400_custom','r2plus1d_resnet34_tranconv_lateral','r2plus1d_resnet34_tranconv_lateral_tanhbn']

from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
import os


from .r2plus1d import BasicBlock,conv3x1x1,Conv2Plus1D

class R2Plus1D(HybridBlock):
    r"""The R2+1D network.
    A Closer Look at Spatiotemporal Convolutions for Action Recognition.
    CVPR, 2018. https://arxiv.org/abs/1711.11248

    Parameters
    ----------
    nclass : int
        Number of classes in the training dataset.
    block : Block, default is `Bottleneck`.
        Class for the residual block.
    layers : list of int
        Numbers of layers in each block
    dropout_ratio : float, default is 0.5.
        The dropout rate of a dropout layer.
        The larger the value, the more strength to prevent overfitting.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    init_std : float, default is 0.001.
        Standard deviation value when initialize the dense layers.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, nclass, block, layers, dropout_ratio=0.5,
                 num_segments=1, num_crop=1, feat_ext=False, use_lateral=False,
                 init_std=0.001, ctx=None, partial_bn=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(R2Plus1D, self).__init__()
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.use_lateral = use_lateral
        self.inplanes = 64
        self.feat_dim = 512 * block.expansion
               

        with self.name_scope():
            self.conv1 = nn.Conv3D(in_channels=3, channels=45, kernel_size=(1, 7, 7),
                                   strides=(1, 2, 2), padding=(0, 3, 3), use_bias=False)
            self.bn1 = norm_layer(in_channels=45, **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.conv2 = conv3x1x1(in_planes=45, out_planes=64)
            self.bn2 = norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs))

            if self.partial_bn:
                if norm_kwargs is not None:
                    norm_kwargs['use_global_stats'] = True
                else:
                    norm_kwargs = {}
                    norm_kwargs['use_global_stats'] = True

            self.layer1 = self._make_res_layer(block=block,
                                               planes=64,
                                               blocks=layers[0],
                                               layer_name='layer1_')
            self.layer2 = self._make_res_layer(block=block,
                                               planes=128,
                                               blocks=layers[1],
                                               stride=2,
                                               layer_name='layer2_')
            self.layer3 = self._make_res_layer(block=block,
                                               planes=256,
                                               blocks=layers[2],
                                               stride=2,
                                               layer_name='layer3_')
            self.layer4 = self._make_res_layer(block=block,
                                               planes=512,
                                               blocks=layers[3],
                                               stride=2,
                                               layer_name='layer4_')

            self.avgpool = nn.GlobalAvgPool3D()
            self.dropout = nn.Dropout(rate=self.dropout_ratio)
            self.fc = nn.Dense(in_units=self.feat_dim, units=nclass,
                               weight_initializer=init.Normal(sigma=self.init_std))

    def hybrid_forward(self, F, x):
        """Hybrid forward of R2+1D net"""
        if self.use_lateral:
            lateral = []
        x = self.conv1(x) #conv1 :  (5, 45, 16, 56, 56)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x) #conv2 :  (5, 64, 16, 56, 56)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x) #layer1 :  (5, 64, 16, 56, 56)
        
        x = self.layer2(x) #layer2 :  (5, 128, 8, 28, 28)
        if self.use_lateral:
            lateral.append(x)
        x = self.layer3(x) #layer3 :  (5, 256, 4, 14, 14)
        if self.use_lateral:
            lateral.append(x)
        x = self.layer4(x) #layer4 :  (5, 512, 2, 7, 7)
        if self.use_lateral:
            lateral.append(x)        
        x = self.avgpool(x)
        x = F.squeeze(x, axis=(2, 3, 4))

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)
        
        if self.use_lateral:
            lateral.append(x)

        if self.feat_ext:
            return x

        x = self.fc(self.dropout(x))
        if self.use_lateral:
            return x, lateral
        else:            
            return x

    def _make_res_layer(self,
                        block,
                        planes,
                        blocks,
                        stride=1,
                        norm_layer=BatchNorm,
                        norm_kwargs=None,
                        layer_name=''):
        """Build each stage of a ResNet"""
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix=layer_name + 'downsample_')
            with downsample.name_scope():
                downsample.add(nn.Conv3D(in_channels=self.inplanes,
                                         channels=planes * block.expansion,
                                         kernel_size=1,
                                         strides=(stride, stride, stride),
                                         use_bias=False))
                downsample.add(norm_layer(in_channels=planes * block.expansion,
                                          **({} if norm_kwargs is None else norm_kwargs)))

        layers = nn.HybridSequential(prefix=layer_name)
        with layers.name_scope():
            layers.add(block(inplanes=self.inplanes,
                             planes=planes,
                             stride=stride,
                             downsample=downsample))

            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.add(block(inplanes=self.inplanes, planes=planes))

        return layers
    
#===============================================================================  

class R2Plus1D_TranConv(HybridBlock):
    def __init__(self, dropout_ratio=0.5,expansion=1,
                 num_segments=1, num_crop=1, feat_ext=False,
                 init_std=0.001, ctx=None, partial_bn=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(R2Plus1D_TranConv, self).__init__()
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.inplanes = 64
        self.feat_dim = 512 * expansion
        with self.name_scope():
            self.bn1 = norm_layer(in_channels=45, **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')            
            self.bn2 = norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs))
            self.sigmoid = nn.Activation('sigmoid')
        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True
                
        self.up_layer4 = nn.Conv3DTranspose(in_channels=512, channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))             
        self.up_sample4 = nn.Conv3DTranspose(in_channels=256, channels=256, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False,  weight_initializer=init.Bilinear())             
        self.up_layer3 = nn.Conv3DTranspose(in_channels=256, channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))             
        self.up_sample3 = nn.Conv3DTranspose(in_channels=128, channels=128, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False,  weight_initializer=init.Bilinear())             
        self.up_layer2 = nn.Conv3DTranspose(in_channels=128, channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))           
        self.up_sample2 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False, weight_initializer=init.Bilinear()) 
        self.up_layer1 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1)) 
        self.up_sample1 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(1,2,2), strides=(1,2,2), padding=(1,0,0),
            use_bias=False,  weight_initializer=init.Bilinear())                 
        self.up_conv2 = nn.Conv3DTranspose(in_channels=64, channels=45, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
        self.up_conv1 = nn.Conv3DTranspose(in_channels=45, channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
    def hybrid_forward(self, F, x):
        x = self.up_layer4(x)        
        x = self.relu(x)
        x = self.up_sample4(x)
        x = self.up_layer3(x)
        x = self.relu(x)
        x = self.up_sample3(x)
        x = self.up_layer2(x)
        x = self.relu(x)
        x = self.up_sample2(x)
        x = self.up_layer1(x)
        x = self.relu(x)
        x = self.up_sample1(x)
        x = self.up_conv2(x)
        x = self.relu(x)
        x = self.up_conv1(x)            
        x = self.sigmoid(x)
        return x            

class R2Plus1D_TranConv_lateral_tanhbn(HybridBlock):
    def __init__(self, dropout_ratio=0.5,expansion=1,
                 num_segments=1, num_crop=1, feat_ext=False,
                 init_std=0.001, ctx=None, partial_bn=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(R2Plus1D_TranConv_lateral_tanhbn, self).__init__()
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.inplanes = 64
        self.feat_dim = 512 * expansion
        with self.name_scope():            
            self.relu = nn.Activation('relu')                        
            self.bn_l4 = norm_layer(in_channels=256, **({} if norm_kwargs is None else norm_kwargs))
            self.bn_l3 = norm_layer(in_channels=256, **({} if norm_kwargs is None else norm_kwargs))
            self.bn_l2 = norm_layer(in_channels=128, **({} if norm_kwargs is None else norm_kwargs))
            self.bn_l1 = norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs))
            self.bn2 = norm_layer(in_channels=45, **({} if norm_kwargs is None else norm_kwargs))
            self.bn1 = norm_layer(in_channels=3, **({} if norm_kwargs is None else norm_kwargs))
            self.tanh = nn.Activation('tanh')
        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True                
        self.up_layer4 = nn.Conv3DTranspose(in_channels=512, channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))             
        self.up_sample4 = nn.Conv3DTranspose(in_channels=256, channels=256, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False,  weight_initializer=init.Bilinear())         
        self.up_layer3 = nn.Conv3DTranspose(in_channels=512, channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))             
        self.up_sample3 = nn.Conv3DTranspose(in_channels=256, channels=256, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False,  weight_initializer=init.Bilinear())          
        self.up_layer2 = nn.Conv3DTranspose(in_channels=384, channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))           
        self.up_sample2 = nn.Conv3DTranspose(in_channels=128, channels=128, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False, weight_initializer=init.Bilinear())         
        self.up_layer1 = nn.Conv3DTranspose(in_channels=128, channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1)) 
        self.up_sample1 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(1,2,2), strides=(1,2,2), padding=(1,0,0),
            use_bias=False,  weight_initializer=init.Bilinear())           
        self.up_conv2 = nn.Conv3DTranspose(in_channels=64, channels=45, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
        self.up_conv1 = nn.Conv3DTranspose(in_channels=45, channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.up_conv0 = nn.Conv3DTranspose(in_channels=3, channels=3, kernel_size=(1,1,1), padding=0)
    def hybrid_forward(self, F, l0,l1,l2):
        x = l2#lateral[2]
        x = self.up_layer4(x) #(5, 256, 2, 7, 7)
        x = self.bn_l4(x)
        x = self.relu(x)
        x = self.up_sample4(x) #(5, 256, 4, 14, 14)         
        x = F.concat(x,l1)#lateral[1]) #(5, 256+256, 4, 14, 14)  
        
        x = self.up_layer3(x)
        x = self.bn_l3(x)
        x = self.relu(x)
        x = self.up_sample3(x) # (5, 256, 8, 28, 28)                
        x = F.concat(x,l0)#lateral[0]) # (5, 256 + 128, 8, 28, 28)        
        
        x = self.up_layer2(x)
        x = self.bn_l2(x)
        x = self.relu(x)
        x = self.up_sample2(x)
        
        x = self.up_layer1(x)
        x = self.bn_l1(x)
        x = self.relu(x)
        x = self.up_sample1(x)

        x = self.up_conv2(x)
        x = self.bn2(x)
        x = self.relu(x)        
        x = self.up_conv1(x)
        x = self.bn1(x)
        x = self.up_conv0(x)
        x = self.tanh(x)
        return x

class R2Plus1D_TranConv_lateral(HybridBlock):
    def __init__(self, dropout_ratio=0.5,expansion=1,
                 num_segments=1, num_crop=1, feat_ext=False,
                 init_std=0.001, ctx=None, partial_bn=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(R2Plus1D_TranConv_lateral, self).__init__()
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.inplanes = 64
        self.feat_dim = 512 * expansion
        with self.name_scope():
            self.bn1 = norm_layer(in_channels=45, **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')            
            self.bn2 = norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs))
            self.sigmoid = nn.Activation('sigmoid')
        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True                
        self.up_layer4 = nn.Conv3DTranspose(in_channels=512, channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))             
        self.up_sample4 = nn.Conv3DTranspose(in_channels=256, channels=256, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False,  weight_initializer=init.Bilinear())         
        self.up_layer3 = nn.Conv3DTranspose(in_channels=512, channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))             
        self.up_sample3 = nn.Conv3DTranspose(in_channels=256, channels=256, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False,  weight_initializer=init.Bilinear())          
        self.up_layer2 = nn.Conv3DTranspose(in_channels=384, channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))           
        self.up_sample2 = nn.Conv3DTranspose(in_channels=128, channels=128, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False, weight_initializer=init.Bilinear())         
        self.up_layer1 = nn.Conv3DTranspose(in_channels=128, channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1)) 
        self.up_sample1 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(1,2,2), strides=(1,2,2), padding=(1,0,0),
            use_bias=False,  weight_initializer=init.Bilinear())           
        self.up_conv2 = nn.Conv3DTranspose(in_channels=64, channels=45, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
        self.up_conv1 = nn.Conv3DTranspose(in_channels=45, channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1))        
    def hybrid_forward(self, F, l0,l1,l2):
        x = l2#lateral[2]
        x = self.up_layer4(x) #(5, 256, 2, 7, 7)
        x = self.relu(x)
        x = self.up_sample4(x) #(5, 256, 4, 14, 14)         
        x = F.concat(x,l1)#lateral[1]) #(5, 256+256, 4, 14, 14)         
        x = self.up_layer3(x)
        x = self.relu(x)
        x = self.up_sample3(x) # (5, 256, 8, 28, 28)        
        x = F.concat(x,l0)#lateral[0]) # (5, 256 + 128, 8, 28, 28)        
        x = self.up_layer2(x)
        x = self.relu(x)
        x = self.up_sample2(x)
        x = self.up_layer1(x)
        x = self.relu(x)
        x = self.up_sample1(x)
        x = self.up_conv2(x)
        x = self.relu(x)
        x = self.up_conv1(x)            
        x = self.sigmoid(x)
        return x        
    

class R2Plus1D_TranConv_lateral_tanhbn_tran(HybridBlock):
    def __init__(self, dropout_ratio=0.5,expansion=1,
                 num_segments=1, num_crop=1, feat_ext=False,
                 init_std=0.001, ctx=None, partial_bn=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(R2Plus1D_TranConv_lateral_tanhbn, self).__init__()
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.inplanes = 64
        self.feat_dim = 512 * expansion
        with self.name_scope():            
            self.relu = nn.Activation('relu')                        
            self.bn_l4 = norm_layer(in_channels=256, **({} if norm_kwargs is None else norm_kwargs))
            self.bn_l3 = norm_layer(in_channels=256, **({} if norm_kwargs is None else norm_kwargs))
            self.bn_l2 = norm_layer(in_channels=128, **({} if norm_kwargs is None else norm_kwargs))
            self.bn_l1 = norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs))
            self.bn2 = norm_layer(in_channels=45, **({} if norm_kwargs is None else norm_kwargs))
            self.bn1 = norm_layer(in_channels=3, **({} if norm_kwargs is None else norm_kwargs))
            self.tanh = nn.Activation('tanh')
        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True                
        self.up_layer4 = nn.Conv3DTranspose(in_channels=512, channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))             
        self.up_sample4 = nn.Conv3DTranspose(in_channels=256, channels=256, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False,  weight_initializer=init.Bilinear())         
        self.up_layer3 = nn.Conv3DTranspose(in_channels=512, channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))             
        self.up_sample3 = nn.Conv3DTranspose(in_channels=256, channels=256, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False,  weight_initializer=init.Bilinear())          
        self.up_layer2 = nn.Conv3DTranspose(in_channels=384, channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))           
        self.up_sample2 = nn.Conv3DTranspose(in_channels=128, channels=128, kernel_size=(2,2,2), strides=(2,2,2), padding=0,
            use_bias=False, weight_initializer=init.Bilinear())         
        self.up_layer1 = nn.Conv3DTranspose(in_channels=128, channels=64, kernel_size=(3, 3, 3), padding=(0, 1, 1)) 
        self.up_sample1 = nn.Conv3DTranspose(in_channels=64, channels=64, kernel_size=(1,2,2), strides=(1,2,2), padding=(1,0,0),
            use_bias=False,  weight_initializer=init.Bilinear())           
        self.up_conv2 = nn.Conv3DTranspose(in_channels=64, channels=45, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
        self.up_conv1 = nn.Conv3DTranspose(in_channels=45, channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.up_conv0 = nn.Conv3DTranspose(in_channels=3, channels=3, kernel_size=(1,1,1), padding=0)
    def hybrid_forward(self, F, l0,l1,l2):
        x = l2#lateral[2]
        x = self.up_layer4(x) #(5, 256, 2, 7, 7)
        x = self.bn_l4(x)
        x = self.relu(x)
        x = self.up_sample4(x) #(5, 256, 4, 14, 14)         
        x = F.concat(x,l1)#lateral[1]) #(5, 256+256, 4, 14, 14)  
        
        x = self.up_layer3(x)
        x = self.bn_l3(x)
        x = self.relu(x)
        x = self.up_sample3(x) # (5, 256, 8, 28, 28)                
        x = F.concat(x,l0)#lateral[0]) # (5, 256 + 128, 8, 28, 28)        
        
        x = self.up_layer2(x)
        x = self.bn_l2(x)
        x = self.relu(x)
        x = self.up_sample2(x)
        
        x = self.up_layer1(x)
        x = self.bn_l1(x)
        x = self.relu(x)
        x = self.up_sample1(x)

        x = self.up_conv2(x)
        x = self.bn2(x)
        x = self.relu(x)        
        x = self.up_conv1(x)
        x = self.bn1(x)
        x = self.up_conv0(x)
        x = self.tanh(x)
        return x


def r2plus1d_resnet34_tranconv_lateral(nelength=16, pretrained=False,ctx=cpu(),**kwargs):
    model = R2Plus1D_TranConv_lateral()
    if pretrained:
        modelfile = '0.9315-ucf101-r2plus1d_resnet34_tranconv_lateral-079-best.params'#'0.8567-ucf101-r2plus1d_resnet34_tranconv_lateral-079-best.params'
        root = '/home/hp/lcx/Action-Recognition/logs/param_rgb_r2plus1d_resnet18_kinetics400_custom_ucf101_nlength16_lateral_1'
        filepath = os.path.join(root,modelfile)
        filepath = os.path.expanduser(filepath)
        model.load_parameters(modelfile,ctx=ctx,allow_missing=True)
        print(filepath)
    else:
        model.initialize(init.MSRAPrelu(), ctx=ctx)
    #model.collect_params().reset_ctx(ctx)
    
    return model
    
def r2plus1d_resnet34_tranconv_lateral_tanhbn(nelength=16, pretrained=False,ctx=cpu(),**kwargs):
    model = R2Plus1D_TranConv_lateral_tanhbn()
    if pretrained:
        pass
        #modelfile = '0.9315-ucf101-r2plus1d_resnet34_tranconv_lateral-079-best.params'#'0.8567-ucf101-r2plus1d_resnet34_tranconv_lateral-079-best.params'
        #root = '/home/hp/lcx/Action-Recognition/logs/param_rgb_r2plus1d_resnet18_kinetics400_custom_ucf101_nlength16_lateral_1'
        #filepath = os.path.join(root,modelfile)
        #filepath = os.path.expanduser(filepath)
        #model.load_parameters(modelfile,ctx=ctx,allow_missing=True)
        #print(filepath)
    else:
        model.initialize(init.MSRAPrelu(), ctx=ctx)
    #model.collect_params().reset_ctx(ctx)
    
    return model

def r2plus1d_resnet18_kinetics400_custom(nclass=400, pretrained=False, pretrained_base=True,use_kinetics_pretrain=True,
                                  root='~/.mxnet/models', num_segments=1, num_crop=1,use_lateral=False,
                                  feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet18 backbone trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    """
    model = R2Plus1D(nclass=nclass,
                     block=BasicBlock,
                     layers=[2, 2, 2, 2],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     **kwargs)
    """
    from .model_zoo import get_model
    #model = get_model('r2plus1d_resnet18_kinetics400', nclass=nclass,num_crop=num_crop,
    #                  feat_ext=feat_ext,num_segments=num_segments,ctx=ctx,pretrained=False) 
    
    model = R2Plus1D(nclass=nclass,
                     block=BasicBlock,
                     layers=[2, 2, 2, 2],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     use_lateral=use_lateral,
                     **kwargs)    
    model.initialize(init.MSRAPrelu(), ctx=ctx)
    
    if use_kinetics_pretrain and not pretrained:
        print('use_kinetics_pretrain == True')
        from .model_store import get_model_file
        kinetics_model = get_model('r2plus1d_resnet18_kinetics400', nclass=400, pretrained=True)
        source_params = kinetics_model.collect_params()
        target_params = model.collect_params()        
        assert len(source_params.keys()) == len(target_params.keys())
        
        pretrained_weights = []
        for layer_name in source_params.keys():
            pretrained_weights.append(source_params[layer_name].data())
        
        for i, layer_name in enumerate(target_params.keys()):
            #print(i,',',layer_name)
            if i + 2 == len(source_params.keys()):
                # skip the last dense layer
                break
            target_params[layer_name].set_data(pretrained_weights[i])
            
        
        #from ...data import Kinetics400Attr
        #attrib = Kinetics400Attr()
        #model.classes = attrib.classes
    elif pretrained:
        #model.load_parameters(get_model_file('r2plus1d_resnet18_kinetics400',tag=pretrained, root=root), ctx=ctx)
        pass
    else:
        print('use_kinetics_pretrain == False')
        #model.initialize(init.MSRAPrelu(), ctx=ctx)
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet34_kinetics400_custom(nclass=400, pretrained=False, pretrained_base=True,use_kinetics_pretrain=True,
                                  root='~/.mxnet/models', num_segments=1, num_crop=1,
                                  feat_ext=False, ctx=cpu(), **kwargs):

    from .model_zoo import get_model
    #model = get_model('r2plus1d_resnet34_kinetics400', nclass=nclass,num_crop=num_crop,
     #                 feat_ext=feat_ext,num_segments=num_segments,ctx=ctx,pretrained=False) 
    model = R2Plus1D(nclass=nclass,
                 block=BasicBlock,
                 layers=[3, 4, 6, 3],
                 num_segments=num_segments,
                 num_crop=num_crop,
                 feat_ext=feat_ext,
                 ctx=ctx,
                 **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)
        
    
    if use_kinetics_pretrain and not pretrained:
        #from .model_store import get_model_file
        kinetics_model = get_model('r2plus1d_resnet34_kinetics400', nclass=400, pretrained=True)
        source_params = kinetics_model.collect_params()
        target_params = model.collect_params()        
        assert len(source_params.keys()) == len(target_params.keys())
        
        pretrained_weights = []
        for layer_name in source_params.keys():
            pretrained_weights.append(source_params[layer_name].data())
        
        for i, layer_name in enumerate(target_params.keys()):
            #print(i,',',layer_name)
            if i + 2 == len(source_params.keys()):
                # skip the last dense layer
                break
            target_params[layer_name].set_data(pretrained_weights[i])
            
        
        #from ...data import Kinetics400Attr
        #attrib = Kinetics400Attr()
        #model.classes = attrib.classes
    elif pretrained:
        #model.load_parameters(get_model_file('r2plus1d_resnet18_kinetics400',tag=pretrained, root=root), ctx=ctx)
        pass
    else:
        model.initialize(init.MSRAPrelu(), ctx=ctx)
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet34_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                  root='~/.mxnet/models', num_segments=1, num_crop=1,
                                  feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet34 backbone trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = R2Plus1D(nclass=nclass,
                     block=BasicBlock,
                     layers=[3, 4, 6, 3],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('r2plus1d_resnet34_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        #from ...data import Kinetics400Attr
        #attrib = Kinetics400Attr()
        #model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet50_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                  root='~/.mxnet/models', num_segments=1, num_crop=1,
                                  feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet50 backbone trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = R2Plus1D(nclass=nclass,
                     block=Bottleneck,
                     layers=[3, 4, 6, 3],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('r2plus1d_resnet50_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
       # from ...data import Kinetics400Attr
        #attrib = Kinetics400Attr()
        #model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet101_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                   root='~/.mxnet/models', num_segments=1, num_crop=1,
                                   feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet101 backbone trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = R2Plus1D(nclass=nclass,
                     block=Bottleneck,
                     layers=[3, 4, 23, 3],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('r2plus1d_resnet101_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        #from ...data import Kinetics400Attr
        #attrib = Kinetics400Attr()
        #model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet152_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                   root='~/.mxnet/models', num_segments=1, num_crop=1,
                                   feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet152 backbone trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = R2Plus1D(nclass=nclass,
                     block=Bottleneck,
                     layers=[3, 8, 36, 3],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('r2plus1d_resnet152_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        #from ...data import Kinetics400Attr
        #attrib = Kinetics400Attr()
        #model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
