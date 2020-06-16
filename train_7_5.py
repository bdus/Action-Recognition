#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020年3月6日18:52:46
2020年3月15日 20:21:37

@author: bdus

(5,3,16,112,112)

loss是修改过的 predict + construction 修改之前的用train_6_1.py

train_7_1.py

尝试对抗训练
D
resnet34_v1b_k400_ucf101 class==2 seg=16
G  
R2+1D r2plus1d_resnet18_kinetics400_custom

UCF101上继续跑

"""
from __future__ import division

import argparse, time, logging, os, sys, math
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
os.environ['CUDA_VISIBLE_DEVICES']='1' #0,1

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter

from gluoncv.data.transforms import video
from gluoncv.data import UCF101, Kinetics400, SomethingSomethingV2, HMDB51
#from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory

from model_zoo import get_model as myget

class AttrDisplay:
  def gatherAttrs(self):
    return ",".join("{}={}"
            .format(k, getattr(self, k))
            for k in self.__dict__.keys())
    # attrs = []
    # for k in self.__dict__.keys():
    #   item = "{}={}".format(k, getattr(self, k))
    #   attrs.append(item)
    # return attrs
    # for k in self.__dict__.keys():
    #   attrs.append(str(k) + "=" + str(self.__dict__[k]))
    # return ",".join(attrs) if len(attrs) else 'no attr'
  def __str__(self):
    return "[{}:{}]".format(self.__class__.__name__, self.gatherAttrs())

class config(AttrDisplay):
    def __init__(self):
        self.predict_T = 3
        self.use_take = True
        self.new_length = 2*(16 + self.predict_T-1)
        self.new_step = 1
        self.model = 'r2plus1d_resnet34_kinetics400_custom'
        self.use_kinetics_pretrain = False
        self.TranConv_model = 'r2plus1d_resnet34_tranconv_lateral_tanhbn'
        self.use_lateral=True
        self.freeze_lateral=False #True
        self.discriminator = 'resnet34_v1b_k400_ucf101'
        #self.switch_epoch_g = 5
        #self.switch_epoch_d = 5
        self.save_dir = 'logs/param_rgb_r2plus1d_resnet34_kinetics400_custom_hmdb51_nlength16_lateral_scratch_gan_train75'
        self.num_classes = 51
        #self.new_length_diff = self.new_length +1
        self.dataset = 'hmdb51'#'hmdb51'#'ucf101'
        self.train_dir = os.path.expanduser('~/.mxnet/datasets/ucf101/rawframes')#'/media/hp/mypan/BGSDecom/cv_MOG2/fgs')#        
        self.train_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
        self.val_setting = '/home/hp/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
        self.train_dir_hmdb51 = os.path.expanduser('~/.mxnet/datasets/hmdb51/rawframes')
        self.train_setting_hmdb51 = '/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_train_split_1_rawframes.txt'
        self.val_setting_hmdb51 = '/home/hp/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt'
        self.logging_file = 'train.log'
        self.name_pattern='img_%05d.jpg'        
        self.input_size=112#204#112
        self.new_height=128#256#128
        self.new_width=171#340#171
        self.input_channel=3
        self.num_segments= 1
        self.num_workers =1
        self.num_gpus = 1
        self.per_device_batch_size = 5
        self.lr = 0.1
        self.lr_decay = 0.1
        self.warmup_lr = 0.001
        self.warmup_epochs = 25
        self.momentum = 0.9
        self.wd = 0.0001      
        self.prefetch_ratio = 1.0
        self.use_amp = False
        self.epochs = 500
        self.lr_decay_epoch = [90,160]
        self.lr_decay_period = 0
        self.scale_ratios = [1.0, 0.8]#[1.0, 0.875, 0.75, 0.66]
        self.dtype = 'float32'
        self.pretrained_lateral_path = 'logs/param_rgb_r2plus1d_resnet18_kinetics400_custom_hmdb51_nlength16_lateral_scratch_gan_train71'
        self.pretrained_lateral_file = '0.3941-hmdb51-r2plus1d_resnet34_tranconv_lateral_tanhbn-078-best.params' 
        self.use_pretrained = False
        self.partial_bn = False
        self.train_patterns = 'r2plus1d0_dense'#'r2plus1d1_dense'
        self.use_train_patterns = False#True
        self.freeze_patterns = '' #'net1'
        self.freeze_lr_mult = 10 #set freezed base layer lr = self.lr * self.freeze_lr_mult
        self.use_mult = False
        self.clip_grad = 40
        self.log_interval = 10
        self.lr_mode = 'cosine'        
        self.resume_epoch = 0 #32
        self.resume_path = ''#'logs/param_rgb_r2plus1d_resnet18_kinetics400_custom_hmdb51_nlength16_lateral_scratch_gan_train71'
        self.resume_params = ''#os.path.join(self.resume_path,'0.3941-hmdb51-r2plus1d_resnet18_kinetics400_custom-078-best.params')
        self.resume_states = ''#os.path.join(self.resume_path,'0.3941-hmdb51-r2plus1d_resnet18_kinetics400_custom-078-best.states')        
        self.reshape_type = 'tsn' #mxc3d c3d tsn tsn_newlength
      

opt = config()

makedirs(opt.save_dir)

filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)
logger.info(opt)
    
# number of GPUs to use
num_gpus = opt.num_gpus
ctx = [mx.gpu(i) for i in range(num_gpus)]
#ctx = [mx.gpu(1)]

# Get the model 
net = myget(name=opt.model, nclass=opt.num_classes, num_segments=opt.num_segments,input_channel=opt.input_channel,batch_normal=opt.partial_bn,use_lateral=opt.use_lateral,use_kinetics_pretrain=opt.use_kinetics_pretrain)
net.cast(opt.dtype)
net.collect_params().reset_ctx(ctx)

net1 = myget(name=opt.TranConv_model)
net1.cast(opt.dtype)
net1.collect_params().reset_ctx(ctx)

net2 = myget(name=opt.discriminator, nclass=2, num_segments=16,input_channel=opt.input_channel,batch_normal=opt.partial_bn)
net2.cast(opt.dtype)
net2.collect_params().reset_ctx(ctx)


#logger.info(net)
if opt.resume_params is not '':
    net.load_parameters(opt.resume_params, ctx=ctx)

#if opt.use_pretrained:
    #net.features_3d.load_parameters(opt.pretrained_ECOfeature3d,ctx=ctx,allow_missing=True)
    #net.output.load_parameters(opt.pretrained_ECOoutput,ctx=ctx,allow_missing=True)
    #logger.info('use pretrained model : %s , %s',opt.pretrained_ECOfeature3d,opt.pretrained_ECOoutput)
if opt.use_pretrained:
    modelpath = os.path.join(opt.pretrained_lateral_path,opt.pretrained_lateral_file)
    modelfile = os.path.expanduser(modelpath)
    net1.load_parameters(modelfile,ctx=ctx,allow_missing=True)
    logger.info('use pretrained model : %s',modelfile)
    
if opt.use_mult:
    #net.collect_params(opt.freeze_patterns).setattr('lr_mult',opt.freeze_lr_mult)
    net1.collect_params().setattr('lr_mult',opt.freeze_lr_mult)

logger.info(net)
net.collect_params().reset_ctx(ctx)
    
transform_train = video.VideoGroupTrainTransform(size=(opt.input_size, opt.input_size), scale_ratios=opt.scale_ratios, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = video.VideoGroupValTransform(size=opt.input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Batch Size for Each GPU
per_device_batch_size = opt.per_device_batch_size
# Number of data loader workers
num_workers = opt.num_workers
# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

# Set train=True for training data. Here we only use a subset of UCF101 for demonstration purpose.
# The subset has 101 training samples, one sample per class.

if opt.dataset == 'ucf101':
    train_dataset = UCF101(setting=opt.train_setting, root=opt.train_dir, train=True,
                       new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,new_step=opt.new_step,
                       target_width=opt.input_size, target_height=opt.input_size,
                       num_segments=opt.num_segments, transform=transform_train)
    val_dataset = UCF101(setting=opt.val_setting, root=opt.train_dir, train=False,
                     new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,new_step=opt.new_step,
                     target_width=opt.input_size, target_height=opt.input_size,
                     num_segments=opt.num_segments, transform=transform_test)
elif opt.dataset == 'hmdb51':
    train_dataset = HMDB51(setting=opt.train_setting_hmdb51, root=opt.train_dir_hmdb51, train=True,
           new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
           target_width=opt.input_size, target_height=opt.input_size, 
           num_segments=opt.num_segments, transform=transform_train)
    val_dataset = HMDB51(setting=opt.val_setting_hmdb51, root=opt.train_dir_hmdb51, train=False,
           new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
           target_width=opt.input_size, target_height=opt.input_size, 
           num_segments=opt.num_segments, transform=transform_test)
else:
    logger.info('Dataset %s is not supported yet.' % (opt.dataset))



train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   prefetch=int(opt.prefetch_ratio * num_workers), last_batch='rollover')
val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 prefetch=int(opt.prefetch_ratio * num_workers), last_batch='discard')


#train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                          #         shuffle=True, num_workers=num_workers)$
logger.info('Load %d training samples.' % len(train_dataset))
#val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size,
#                                   shuffle=False, num_workers=num_workers)
    
def batch_fn(batch, ctx):
    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label
# Learning rate decay factor
lr_decay = opt.lr_decay
# Epochs where learning rate decays
#lr_decay_epoch = opt.lr_decay_epoch
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = opt.lr_decay_epoch
lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]

# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

num_batches = len(train_data)
lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=opt.warmup_lr, target_lr=opt.lr,
                nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
    LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                nepochs=opt.epochs - opt.warmup_epochs,
                iters_per_epoch=num_batches,
                step_epoch=lr_decay_epoch,
                step_factor=lr_decay, power=2)
])
optimizer_params['lr_scheduler'] = lr_scheduler


if opt.use_lateral and not opt.freeze_lateral:
    print("============== use_lateral")
    lst = list(net.collect_params().values()) + list(net1.collect_params().values())
    trainer = gluon.Trainer(lst, optimizer, optimizer_params, update_on_kvstore=False)    
else:
    print("============== training net0. net1 is not included in training")
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, update_on_kvstore=False)

trainer_d = gluon.Trainer(net2.collect_params(), optimizer, optimizer_params, update_on_kvstore=False)

if opt.resume_states is not '':
    trainer.load_states(opt.resume_states)

# Define our trainer for net
#trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
loss_bec = gluon.loss.SigmoidBinaryCrossEntropyLoss()
loss_l2 = gluon.loss.L2Loss(weight=1.0)
loss_l2.initialize()

def takeT(X,T=0):
    #idx = nd.array(nd.arange(T,opt.new_length,2),ctx=ctx[0])
    idx = nd.array([2*n+T for n in range(16)],ctx=ctx[0])
    return nd.take(X.astype(opt.dtype, copy=False),idx,axis=3)

if opt.use_take:
    print('==============================================',opt.new_length)
    print([2*n+0 for n in range(16)])#nd.arange(0,opt.new_length,2))
    print([2*n+opt.predict_T for n in range(16)])#nd.arange(opt.predict_T,opt.new_length,2))


train_metric = mx.metric.Accuracy()

train_history = TrainingHistory(['training-acc','val-top1-acc','val-top5-acc','training-loss','cross-loss','mse-loss','pre-loss'])

lr_decay_count = 0
best_val_score = 0

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)


def test(ctx,val_data):
    acc_top1.reset()
    acc_top5.reset()    
    #L = gluon.loss.SoftmaxCrossEntropyLoss()    
    #L2 = gluon.loss.L2Loss(weight=1.0)
    #L2.initialize()
    num_test_iter = len(val_data)
    val_d_loss = 0
    val_epoch_loss = 0
    val_mse_loss = 0
    val_cheat_loss = 0
    val_pre_loss = 0    
    for i, batch in enumerate(val_data):
        val_output = []
        data, label = batch_fn(batch, ctx)
        for X, y in zip(data,label):            
            X1 = takeT(X)
            X2 = takeT(X,T=opt.predict_T)
            X1 = X1.reshape((-1,) + X1.shape[2:]) # for reconstraction feed g
            X2 = X2.reshape((-1,) + X2.shape[2:]) # for prodiction feed d

            pred, latel = net(X1.astype(opt.dtype, copy=False)) 
            val_output.append(pred)
            x_hat = net1(latel[0].astype(opt.dtype, copy=False),
                         latel[1].astype(opt.dtype, copy=False),latel[2].astype(opt.dtype, copy=False))
            # AutoGrad train d 
            #with ag.record():
            x_hat_reshape = nd.transpose(data=x_hat ,axes=(0,2,1,3,4))
            x_hat_reshape = x_hat_reshape.reshape((-1,) + x_hat_reshape.shape[2:]) #.reshape                
            x2_reshape = nd.transpose(data=X2,axes=(0,2,1,3,4))
            x2_reshape = x2_reshape.reshape((-1,) + x2_reshape.shape[2:]) #.reshape
            
            d_pred_real = net2(x2_reshape.astype(opt.dtype, copy=False))
            d_pred_fake = net2(x_hat_reshape.astype(opt.dtype, copy=False))                
            loss_d = loss_fn(d_pred_real,nd.ones(shape=(batch_size),ctx=ctx[0])) + loss_fn(d_pred_fake,nd.zeros(shape=(batch_size),ctx=ctx[0]))
                       
            # train g
            loss_g_l2 = loss_l2(x_hat, X1.astype(opt.dtype, copy=False) ) + loss_l2(x_hat, X2.astype(opt.dtype, copy=False) )
            loss_g_cheat = loss_fn(d_pred_fake, nd.ones(shape=(batch_size),ctx=ctx[0]) )#net2(x_hat_reshape))
            loss_g_ft = loss_fn(pred, y.astype(opt.dtype, copy=False))
            loss_g = loss_g_l2 + loss_g_cheat + loss_g_ft 

            val_epoch_loss += loss_g.mean().asscalar() / len(label)
            val_d_loss += loss_d.mean().asscalar() / len(label)            
            val_mse_loss += loss_g_l2.mean().asscalar() / len(label)
            val_cheat_loss += loss_g_cheat.mean().asscalar() / len(label)
            val_pre_loss += loss_g_ft.mean().asscalar() / len(label)                
            
        acc_top1.update(label, val_output)
        acc_top5.update(label, val_output)        
    
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    
    val_dloss = val_d_loss / num_test_iter
    val_loss = val_epoch_loss / num_test_iter
    loss_mse = val_mse_loss / num_test_iter
    loss_pre = val_pre_loss / num_test_iter
    loss_cheat = val_cheat_loss / num_test_iter
    
    return (top1, top5, val_loss, loss_mse,loss_pre,loss_cheat,val_dloss)

#acc_top1_val, acc_top5_val, loss_val, loss_mse, loss_pre, lossflow= test(ctx, val_data) 

# training 
for epoch in range(opt.resume_epoch, opt.epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0
    mse_loss = 0
    pre_loss = 0
    flow_loss = 0
    d_loss = 0
    btic = time.time()
    # Loop through each batch of training data
    for i, batch in enumerate(train_data): 
        """ 
            type(batch) == 'list' 
            len(batch) == 2
            type(batch[0]) == <class 'mxnet.ndarray.ndarray.NDArray'>
            batch[0].shape == (5, 1, 3, 36, 112, 112)
            batch[1].shape == (5,)
            
            type(data) == list
            len(data) == 1
            data[0].shape  == (5, 1, 3, 36, 112, 112)
            label[0].shape == (5,)                       
        """
        # Extract data and label
        data, label = batch_fn(batch, ctx)
        output = []
        #with ag.record():
        for X, y in zip(data,label):
            """
            X.shape == (5, 1, 3, 36, 112, 112)
            y.shape == (5,)
            data,label 里面只有一个mini-batch
            X1.shape == (5, 1, 3, 16, 112, 112) == X2.shape
            after reshape
            X1.shape == (5, 3, 16, 112, 112) 
            
            pred.shape == (5, 51)
            type(latel) == 'list'
            len(latel) == 4
            >>> latel[0].shape    (5, 128, 8, 28, 28)
            >>> latel[1].shape    (5, 256, 4, 14, 14)
            >>> latel[2].shape    (5, 512, 2, 7, 7)
            >>> latel[3].shape    (5, 512)
            
            >>> x_hat.shape                (5, 3, 16, 112, 112)
            >>> x_hat_reshape.shape                (5, 16, 3, 112, 112)
            >>> x_hat_reshape = x_hat_reshape.reshape((-1,) + x_hat_reshape.shape[2:])
            >>> x_hat_reshape.shape                (80, 3, 112, 112)
            
            X2.shape == (5, 3, 16, 112, 112) 
            >>> x2_reshape.shape    (5, 16, 3, 112, 112)                
            >>> x2_reshape = x2_reshape.reshape((-1,) + x2_reshape.shape[2:])
            >>> x2_reshape.shape     (80, 3, 112, 112)
            
            """
            # get data
            X1 = takeT(X)
            X2 = takeT(X,T=opt.predict_T)
            X1 = X1.reshape((-1,) + X1.shape[2:]) # for reconstraction feed g
            X2 = X2.reshape((-1,) + X2.shape[2:]) # for prodiction feed d
            # feeding train d            
            with ag.record():
                """ making x_hat : feeding X1 into net and net1 """
                _ , latel = net(X1.astype(opt.dtype, copy=False)) 
                #output.append(pred)                
                x_hat = net1(latel[0].astype(opt.dtype, copy=False),
                             latel[1].astype(opt.dtype, copy=False),latel[2].astype(opt.dtype, copy=False))
                # train d 

                x_hat_reshape = nd.transpose(data=x_hat ,axes=(0,2,1,3,4))
                x_hat_reshape = x_hat_reshape.reshape((-1,) + x_hat_reshape.shape[2:]) #.reshape                
                x2_reshape = nd.transpose(data=X2,axes=(0,2,1,3,4))
                x2_reshape = x2_reshape.reshape((-1,) + x2_reshape.shape[2:]) #.reshape
                """ train discriminator """
                d_pred_real = net2(x2_reshape.astype(opt.dtype, copy=False))        # feeding real X2
                d_pred_fake = net2(x_hat_reshape.astype(opt.dtype, copy=False))     # feeding fake x_hat               
                loss_d = loss_fn(d_pred_real,nd.ones(shape=(batch_size),ctx=ctx[0])) + loss_fn(d_pred_fake,nd.zeros(shape=(batch_size),ctx=ctx[0]))
                loss_d.backward()
                
            trainer_d.step(batch_size,ignore_stale_grad=True)
            # train g
            with ag.record():
                """ generation x_hat : feeding X1 into net  4 predicting X2"""
                pred, latel = net(X1.astype(opt.dtype, copy=False))
                output.append(pred)
                x_hat = net1(latel[0].astype(opt.dtype, copy=False),
                             latel[1].astype(opt.dtype, copy=False),latel[2].astype(opt.dtype, copy=False))
                """ reconstruction X1 and predicting X2 """
                loss_g_l2 = loss_l2(x_hat, X1.astype(opt.dtype, copy=False) ) + loss_l2(x_hat, X2.astype(opt.dtype, copy=False) )
                """ cheat discriminator """
                loss_g_cheat = loss_fn(d_pred_fake, nd.ones(shape=(batch_size),ctx=ctx[0]) )#net2(x_hat_reshape))
                """ finetuning btw """
                loss_g_ft = loss_fn(pred, y.astype(opt.dtype, copy=False))
                loss_g = loss_g_l2 + loss_g_cheat + loss_g_ft
                loss_g.backward()
            trainer.step(batch_size,ignore_stale_grad=True)
            #cal loss
            d_loss += loss_d.mean().asscalar() / len(label)
            train_loss += loss_g.mean().asscalar() / len(label)
            mse_loss += loss_g_l2.mean().asscalar() / len(label)
            flow_loss += loss_g_cheat.mean().asscalar() / len(label)
            pre_loss += loss_g_ft.mean().asscalar() / len(label)                
                    
        train_metric.update(label,output)
        
        if i % opt.log_interval == 0:
            name,acc = train_metric.get()
            logger.info('[Epoch %d] [%d | %d] train=%f loss=%f mseloss=%f  pre_loss %f cheat: %f d : %f time: %f' %
                  (epoch,i,len(train_data), acc, train_loss / (i+1),mse_loss/(i+1),pre_loss/(i+1),flow_loss/(i+1), d_loss/(i+1), time.time()-btic) )
            btic = time.time()
    
    # epoch loop for test and save parameters
    name, acc = train_metric.get()
    
    acc_top1_val, acc_top5_val, loss_val, loss_mse, loss_pre, lossflow ,val_dloss = test(ctx, val_data) 
    train_history.update([acc,acc_top1_val,acc_top5_val,train_loss/(i+1),loss_val/(i+1),loss_mse/(i+1),loss_pre/(i+1)])
    train_history.plot(save_path=os.path.join(opt.save_dir,'trainlog_wth.jpg'))
    # log 
    logger.info('[Epoch %d] train=%f loss=%f time: %f' %
        (epoch, acc, train_loss / (i+1), time.time()-tic))    
    logger.info('[Epoch %d] val top1 =%f top5=%f val loss=%f,mesloss=%f,loss_pre = %f,loss_cheat=%f, d:%f lr=%f' %
        (epoch, acc_top1_val, acc_top5_val, loss_val ,loss_mse,loss_pre,lossflow,val_dloss, trainer.learning_rate ))
    if acc_top1_val > best_val_score and epoch > 5:
        best_val_score = acc_top1_val
        net.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, opt.model, epoch))
        trainer.save_states('%s/%.4f-%s-%s-%03d-best.states'%(opt.save_dir, best_val_score, opt.dataset, opt.model, epoch))
        net1.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, opt.TranConv_model, epoch))
        net2.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, opt.discriminator, epoch))


# We can plot the metric scores with:
train_history.plot(save_path=os.path.join(opt.save_dir,'trainlog_final.jpg'))