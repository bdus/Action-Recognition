#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:35:29 2019

@author: hp

decomposition a video into forground and background

"""
import sys
import os
import glob
import argparse

import numpy as np
import cv2
import pybgs as bgs

def parse_args():
    parser = argparse.ArgumentParser(description='Use algorithms from bgslibrary decomposition a rawframe-video-folder into forground and background and save it.')
    parser.add_argument('rawframepath',type=str,help='Input path where video are stored as rawframes.')
    parser.add_argument('output_base',type=str,help='Output path where background and foreground images are stored.')
    parser.add_argument('algorithm',type=str,default='FrameDifference',choices=['all']+[str(x) for x in dir(bgs) if type(eval(''.join(['bgs.',x]))) is type(bgs.FrameDifference)])
    parser.add_argument('--prefix_fgd',type=str,default='foreground')
    parser.add_argument('--prefix_bgd',type=str,default='background')
    # TODO : processing video 
    #parser.add_argument('--Image')
    
class PreProcessor:
    def __init__(self,rawframepath,output_base,algorithms):
        self.rawframepath = rawframepath
        self.output_base = output_base
        #choices=['all']+[str(x) for x in dir(bgs) if type(eval(''.join(['bgs.',x]))) is type(bgs.FrameDifference)]
        
        self.acquire_listdir(rawframepath)
        if algorithms == 'all':
            self.algorithm = [str(x) for x in dir(bgs) if isinstance(getattr(bgs,x), bgs.FrameDifference)]
        else:            
            self.algorithms = [algorithms]
        
    def acquire_listdir(self,rawframepath):
        if not os.path.exists(rawframepath):
            raise ValueError('Input path not exist.')
        self.classes = os.listdir(rawframepath)
        
    def mkdir(self,path):
        if not os.path.exists(path):
            os.mkdir(path)
    
    def equal_item(self,path1,path2):
        if not os.path.exists(path1):
            return False
        if not os.path.exists(path2):
            return False
        list1 = os.listdir(path1)
        list2 = os.listdir(path2)
        return len(list1) == len(list2)
    
    def apply_algorithm(self):
        #save as : self.output_base/algorithms/bgs or fgs/classes/video/img.jpg
        assert type(self.algorithms) == list
        for algorithm in self.algorithms:
            print(algorithm)
            fun = eval(''.join(['bgs.',str(algorithm)]))
            method = fun()
            spath = os.path.join(self.output_base,str(algorithm))
            bgpath1 = os.path.join(spath,'bgs')
            fgpath1 = os.path.join(spath,'fgs')            
            self.mkdir(spath)
            self.mkdir(bgpath1)
            self.mkdir(fgpath1)
            for vclass in self.classes:
                cpath = os.path.join(self.rawframepath,vclass)
                bgpath2 = bgpath1 #os.path.join(bgpath1,vclass)
                fgpath2 = fgpath1 #os.path.join(fgpath1,vclass)
#                os.mkdir(bgpath2)
#                os.mkdir(fgpath2)                
                for video in os.listdir(cpath):
                    vpath = os.path.join(cpath,video)
                    bgpath3 = os.path.join(bgpath2,video)
                    fgpath3 = os.path.join(fgpath2,video)
                    if self.equal_item(vpath,fgpath3) and self.equal_item(bgpath3,vpath):
#                        print('sikp:',str(video))
                        continue
#                    method = fun()
                    print('process:',str(video))
                    self.mkdir(bgpath3)
                    self.mkdir(fgpath3)
                    image_array = os.listdir(vpath)
                    image_array = sorted(image_array)
                    for x in range(0,len(image_array)):
                        img_path = os.path.join(vpath,image_array[x])
                        # read file into open cv and apply to algorithm to generate background model
                        #print(img_path)                        
                        assert os.path.exists(img_path)
                        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
#                        print(img.shape)
                        img_output = method.apply(img)
                        img_bgmodel = method.getBackgroundModel()
#                        # show images in python imshow window
#                        cv2.imshow('image', img)
#                        cv2.imshow('img_output', img_output)
#                        cv2.imshow('img_bgmodel', img_bgmodel)
                        # we need waitKey otherwise it wont display the image
                        if 0xFF & cv2.waitKey(10) == 27:
                          break
                    
                        # Comment out to save images to bg and fg folder
                        img_bg = os.path.join(bgpath3,image_array[x])
                        img_fg = os.path.join(fgpath3,image_array[x])
                        cv2.imwrite(img_bg, img_bgmodel)
                        cv2.imwrite(img_fg, img_output)
                    
                        #print("Frames left: " + str(len(image_array)-x))
            break

def main():
    a = PreProcessor('/home/hp/lixiaoyu/dataset/jpg','/media/hp/dataset/UCF101/BGSDecom','ViBe')
    a.apply_algorithm()

    
if __name__ == '__main__':
    main()


#['AdaptiveBackgroundLearning',
# 'AdaptiveSelectiveBackgroundLearning',
# 'CodeBook',
# 'DPAdaptiveMedian',
# 'DPEigenbackground',
# 'DPGrimsonGMM',
# 'DPMean',
# 'DPPratiMediod',
# 'DPTexture',
# 'DPWrenGA',
# 'DPZivkovicAGMM',
# 'FrameDifference',
# 'FuzzyChoquetIntegral',
# 'FuzzySugenoIntegral',
# 'IndependentMultimodal',
# 'KDE',
# 'KNN',
# 'LBAdaptiveSOM',
# 'LBFuzzyAdaptiveSOM',
# 'LBFuzzyGaussian',
# 'LBMixtureOfGaussians',
# 'LBP_MRF',
# 'LBSimpleGaussian',
# 'LOBSTER',
# 'MixtureOfGaussianV2',
# 'MultiCue',
# 'MultiLayer',
# 'PAWCS',
# 'PixelBasedAdaptiveSegmenter',
# 'SigmaDelta',
# 'StaticFrameDifference',
# 'SuBSENSE',
# 'T2FGMM_UM',
# 'T2FGMM_UV',
# 'T2FMRF_UM',
# 'T2FMRF_UV',
# 'TwoPoints',
# 'ViBe',
# 'VuMeter',
# 'WeightedMovingMean',
# 'WeightedMovingVariance']