#!/usr/bin/env bash

python3 train_recognizer.py \
	--dataset ucf101 \
	--model inceptionv3_ucf101 \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 25 \
	--num-segments 3 \
	--use-tsn \
	--num-gpus 2 \
	--num-data-workers 16 \
	--new-height 340 \
	--new-width 450 \
	--input-size 299 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--clip-grad 40 \
	--partial-bn \
	--save-frequency 5 \
	--log-interval 10 \
	--save-dir 'logs/param_FrameDiff_fgs_inception_seg3_ch3' \
	--data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--val-data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--train-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_1.txt' \
	--val-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_1.txt' 


python3 mytrain_recognizer.py \
	--dataset ucf101 \
	--model inceptionv3_ucf101_sim \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 8 \
	--num-segments 7 \
	--use-tsn \
	--num-gpus 2 \
	--num-data-workers 16 \
	--new-height 340 \
	--new-width 450 \
	--new-length 5 \
	--input-channel 15 \
	--input-size 299 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--clip-grad 40 \
	--partial-bn \
	--save-frequency 5 \
	--log-interval 10 \
	--save-dir 'logs/param_FrameDiff_fgs_incep_seg7_ch15' \
	--data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--val-data-dir '/media/hp/data/BGSDecom/FrameDifference/fgs' \
	--train-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_train_split_1.txt' \
	--val-list '/home/hp/lixiaoyu/dataset/data/ucf101_rgb_flow/ucf101_rgb_val_split_1.txt' 

