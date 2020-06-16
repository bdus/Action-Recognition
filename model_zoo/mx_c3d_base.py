"""
C3D, implemented in Gluon. https://arxiv.org/abs/1412.0767

basenet = myget(name='c3d_kinetics400',nclass=400,input_channel=3,num_segments=16,batch_normal=False,pretrained=True)
x = nd.zeros(shape=(5,3,16,112,112))
y=basenet(x)

"""
# pylint: disable=arguments-differ,unused-argument

__all__ = ['c3d_kinetics400_ucf101','c3d_kinetics400_custome']

from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

from .mx_c3d import c3d_kinetics400

class mx_C3D_base(HybridBlock):
    r"""
    The Convolutional 3D network (C3D).
    Learning Spatiotemporal Features with 3D Convolutional Networks.
    ICCV, 2015. https://arxiv.org/abs/1412.0767

    Parameters
    ----------
    nclass : int
        Number of classes in the training dataset.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    dropout_ratio : float
        Dropout value used in the dropout layers after dense layers to avoid overfitting.
    init_std : float
        Default standard deviation value for initializing dense layers.
    ctx : str
        Context, default CPU. The context in which to load the pretrained weights.
    """

    def __init__(self, nclass, dropout_ratio=0.5, pretrained_base=False,
                 num_segments=1, num_crop=1, feat_ext=False,
                 init_std=0.001, ctx=None, **kwargs):
        super(mx_C3D_base, self).__init__()
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.feat_dim = 8192
        
        pretrained_model = c3d_kinetics400(pretrained = pretrained_base, num_segments = num_segments, **kwargs)
        
        with self.name_scope():
            self.conv1 = pretrained_model.conv1
            self.pool1 = pretrained_model.pool1

            self.conv2 = pretrained_model.conv2
            self.pool2 = pretrained_model.pool2

            self.conv3a = pretrained_model.conv3a
            self.conv3b = pretrained_model.conv3b
            self.pool3 = pretrained_model.pool3

            self.conv4a = pretrained_model.conv4a
            self.conv4b = pretrained_model.conv4b
            self.pool4 = pretrained_model.pool4

            self.conv5a = pretrained_model.conv5a
            self.conv5b = pretrained_model.conv5b
            self.pool5 = pretrained_model.pool5

            self.fc6 = pretrained_model.fc6
            self.fc7 = pretrained_model.fc7
            self.fc8 = nn.Dense(in_units=4096, units=nclass,
                                weight_initializer=init.Normal(sigma=init_std))
            self.dropout = nn.Dropout(rate=dropout_ratio)
            self.relu = nn.Activation('relu')
            self.fc8.initialize()

    def hybrid_forward(self, F, x):
        """Hybrid forward of C3D net"""
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)

        if self.feat_ext:
            return x

        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        return x

def c3d_kinetics400_custome(nclass=101, pretrained=False,use_kinetics_pretrain=True, ctx=cpu(),
                    root='~/.mxnet/models', num_segments=1, num_crop=1,
                    feat_ext=False, **kwargs):
    r"""The Convolutional 3D network (C3D) trained on Kinetics400 dataset.
    Learning Spatiotemporal Features with 3D Convolutional Networks.
    ICCV, 2015. https://arxiv.org/abs/1412.0767

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
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

    model = mx_C3D_base(nclass=nclass, ctx=ctx, num_segments=num_segments,
                num_crop=num_crop, feat_ext=feat_ext, **kwargs)
    #model.initialize(init.MSRAPrelu(), ctx=ctx)

    if use_kinetics_pretrain and not pretrained:
        #from .model_store import get_model_file
        from .model_zoo import get_model
        kinetics_model = get_model('c3d_kinetics400', nclass=400, pretrained=True)
        
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
        
    elif pretrained:    
        pass
        #from .model_store import get_model_file
        #model.load_parameters(get_model_file('c3d_kinetics400',
                                             #tag=pretrained, root=root), ctx=ctx)
        #from ...data import Kinetics400Attr
        #attrib = Kinetics400Attr()
        #model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model    
    
def c3d_kinetics400_ucf101(nclass=101, pretrained=False, ctx=cpu(),
                    root='~/.mxnet/models', num_segments=1, num_crop=1,
                    feat_ext=False, **kwargs):
    r"""The Convolutional 3D network (C3D) trained on Kinetics400 dataset.
    Learning Spatiotemporal Features with 3D Convolutional Networks.
    ICCV, 2015. https://arxiv.org/abs/1412.0767

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
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

    model = mx_C3D_base(nclass=nclass, ctx=ctx, num_segments=num_segments,
                num_crop=num_crop, feat_ext=feat_ext, **kwargs)
    #model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        pass
        #from .model_store import get_model_file
        #model.load_parameters(get_model_file('c3d_kinetics400',
                                             #tag=pretrained, root=root), ctx=ctx)
        #from ...data import Kinetics400Attr
        #attrib = Kinetics400Attr()
        #model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
