# Action-Recognition
Action-Recognition

# 实验

## 1. RGB input VGG16 的train&val@UCF101
实验说明： 主要目的是固定代码框架


```
python3 train_recognizer.py \
    --dataset ucf101 \
	--model vgg16_ucf101 \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 25 \
	--num-gpus 2 \
	--num-data-workers 32 \
	--new-height 256 \
	--new-width 340 \
	--input-size 224 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--save-frequency 5 \
	--log-interval 10 \
	--save-dir 'param_FrameDiff_bgs' \
	--data-dir '/media/hp/data/BGSDecom/FrameDifference/bgs' \
	--val-data-dir '/media/hp/data/BGSDecom/FrameDifference/bgs' \
	--train-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_1.txt' \
	--val-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_1.txt' 	


python3 train_recognizer.py \
	--dataset ucf101 \
	--model vgg16_ucf101 \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 12 \
	--num-segments 3 \
	--use-tsn \
	--num-gpus 2 \
	--num-data-workers 32 \
	--new-height 256 \
	--new-width 340 \
	--input-size 224 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--save-frequency 5 \
	--log-interval 10 \
	--save-dir 'param_FrameDiff_fgs' \
	--data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--val-data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--train-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_1.txt' \
	--val-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_1.txt' \
	| tee param_FrameDiff_fgs.log

python3 train_recognizer.py \
	--dataset ucf101 \
	--model simple \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 12 \
	--num-segments 3 \
	--num-gpus 2 \
	--num-data-workers 32 \
	--new-height 256 \
	--new-width 340 \
	--input-size 224 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--save-frequency 5 \
	--log-interval 10 \
	--save-dir 'param_FrameDiff_fgs_simple' \
	--data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--val-data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--train-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_1.txt' \
	--val-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_1.txt' \
	| tee param_FrameDiff_fgs_simple.log

python3 train_recognizer.py \
	--dataset ucf101 \
	--model vgg16_ucf101 \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 12 \
	--num-segments 3 \
	--use-tsn \
	--num-gpus 2 \
	--num-data-workers 32 \
	--new-height 256 \
	--new-width 340 \
	--input-size 224 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--save-frequency 5 \
	--log-interval 10 \
	--save-dir 'param_ViBe_fgs' \
	--data-dir '/media/hp/data/BGSDecom/ViBe/fgs' \
	--val-data-dir '/media/hp/data/BGSDecom/ViBe/fgs' \
	--train-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_1.txt' \
	--val-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_1.txt' \
	| tee param_ViBe_fgs.log

CUDA_VISIBLE_DEVICES=0,1 python3 train_recognizer.py \
    --dataset ucf101 \
    --data-dir '/media/hp/data/BGSDecom/FrameDifference/bgs'  \
    --train-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_1.txt' \
    --val-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_1.txt' \
    --mode hybrid \
    --dtype float32 \
    --prefetch-ratio 1.0 \
    --model i3d_resnet50_v1_ucf101 \
    --num-classes 101 \
    --batch-size 8 \
    --num-gpus 2 \
    --num-data-workers 16 \
    --input-size 224 \
    --new-height 256 \
    --new-width 340 \
    --new-length 32 \
    --new-step 2 \
    --lr 0.001 \
    --lr-decay 0.1 \
    --lr-mode step \
    --lr-decay-epoch 20,40,50 \
    --momentum 0.9 \
    --wd 0.0001 \
    --num-epochs 50 \
    --scale-ratios 1.0,0.8 \
    --save-frequency 5 \
    --clip-grad 40 \
    --log-interval 20 \
    --logging-file i3d_resnet50_v1_ucf101.log \
    --save-dir ./logs/i3d/ 


python3 mytrain_recognizer.py \
	--dataset ucf101 \
	--model simple \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 5 \
	--num-segments 7 \
	--new-length 5 \
	--input-channel 15 \
	--num-gpus 2 \
	--num-data-workers 4 \
	--new-height 256 \
	--new-width 340 \
	--input-size 224 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--save-frequency 5 \
	--log-interval 10 \
	--save-dir 'param_FrameDiff_fgs_simple_seg7_ch15' \
	--data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--val-data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--train-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_1.txt' \
	--val-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_1.txt' 

```