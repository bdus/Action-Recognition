# Action-Recognition

从零开始的Action-Recognition
——
基于mxnet gluon 和 gluoncv  的行为识别

- ./model_zoo		模型文件
	* model_store.py  卡太小 又下不完k400 只好蹭gluoncv的预训练数据这样子
	* model_zoo.py 		模型库的头文件 网络结构就是要整整齐齐
	* __init__.py 	import用 写了新模型记得填一下
	* <del>simple.py	 	简单分类 为了项目框架写的 模型文件示例 demo [drop]</del>
	* <del>test_1E4.py 	有了前背景想法以后的粗糙实现 已经[drop]</del>
	* <del>MSEloss_vgg.py 	验证学习一致性有没有提高 [drop]</del>
	* <del>inception_v3_k400ft.py	 使用gluoncv inceptionv3的参数 作为用分类做视频的baseline [drop]</del>
	* <del>actionrec_inceptionv3.py		使用gluoncv inceptionv3的参数分类 [drop]</del>
	* <del> inceptionv3_LSTM.py	[drop]</del>
	* resnet18_v1b_ucf101.py 	使用了gluoncv实现的reset 作为分类baseline、复现双流、TSN、前背景的backbone [2D-ResNet/2-stream/TSN/bfgs]
	* C3D.py 	调了一下 [C3D]
	* Res3D.py		调了一下 [3D-ResNet]
	* R21D.py		调了一下 [R(2+1)D-34]
	* F_stCN.py  	简单调了一下一篇比较早的[FstCN]
	* <del>P3D.py 	调P3D 没搞好 [drop]</del>
	*  mx_c3d.py	gluoncv实现的c3d [C3D]
	*  mx_c3d_base.py  基于 mx_c3d.py改的 本来是打算用来和其它无监督算法基于C3D对比的 [C3D&decoder]
	*  r2plus1d.py		gluoncv实现的 [R(2+1)D]
	*  r2plus1d_base.py  	基于 r2plus1d.py改的 用来作为无监督算法的backbone [R(2+1)D&decoder]
	*  ECO.py	调了一下[ECO] 太依赖预训练了
	*  mx_i3d_resnet.py		用了一下gluoncv的[I3D]

-  ./ucf101_bgs		数据集加载	
  
   主要是twostream.py  基于原本gluoncv.data.ucf101修改的 双流版本的UCF101和HMDB51的数据集加载脚本  

- pre_ucf101.py		预处理数据集用
- BgsDecomposer.py 	 放在bgslibrary目录下运行 用于从视频中分割出前景的脚本
- test_model.py  测试单个模型用的脚本
- test_twostream_skip.py 	测试双流模型用的脚本
- train_%d_%d%s.py 实验训练脚本
- <del>其它奇怪的脚本 无视 </del>

# 实验

- train_1_%d
	基于双流结构的实验 包括分类、TSN、前背景、I3D
	
- train_2_%d
   训练C3D的配置

- train_3_%d
   训练ECO

- train_4_1&train_4_2
   训练各种3D卷积、R(2+1)D

- train_4_3& train_4_4
  R(2+1)D作base+反卷积 实现3D自编码

- train_5_%d 
    R(2+1)D作base+反卷积 实现重构和预测

- train_6_%d
	尝试了一下变换自编码器AET 没收敛

- train_7_%d
    R(2+1)D作base+反卷积 +resnet34TSN 基于灰度一致性的对抗训练
      