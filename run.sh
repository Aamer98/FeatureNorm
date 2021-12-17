#!/bin/bash
# bash script to train STARTUP representation with SimCLR self-supervision
export CUDA_VISIBLE_DEVICES=0
##############################################################################################
# Train student representation using MiniImageNet as the source 
##############################################################################################
# Before running the commands, please take care of the TODO appropriately
for target_testset in "EuroSAT" "ISIC" "CropDisease" "ChestX" "miniImageNet_test"
do 
 # TODO: Please set the following argument appropriately 
    # --teacher_path: filename for the teacher model
    # --base_path: path to find base dataset
    # --dir: directory to save the student representation. 
    # E.g. the following commands trains a STARTUP representation based on the teacher specified at
    #      ../teacher_miniImageNet/logs_deterministic/checkpoints/miniImageNet/ResNet10_baseline_256_aug/399.tar 
    #      The student representation is saved at miniImageNet_source/$target_testset\_unlabeled_20/checkpoint_best.pkl
    python STARTUP.py \
    --dir miniImageNet_source/$target_testset\_unlabeled_20 \
    --target_dataset $target_testset \
    --image_size 224 \
    --target_subset_split ../datasets/split_seed_1/$target_testset\_unlabeled_20.csv \
    --bsize 256 \
    --epochs 1000 \
    --save_freq 50 \
    --print_freq 10 \
    --seed 1 \
    --wd 1e-4 \
    --num_workers 4 \
    --model resnet10 \
    --teacher_path ../teacher_miniImageNet/logs_deterministic/checkpoints/miniImageNet/ResNet10_baseline_256_aug/399.tar \
    --teacher_path_version 0 \
    --base_dataset miniImageNet \
    --base_path ../dataset/miniImagenet
    --base_no_color_jitter \
    --base_val_ratio 0.05 \
    --eval_freq 2 \
    --batch_validate \
    --resume_latest 

    zip -r ~/scratch/$target_testset\.zip $SLURM_TMPDIR/STARTUP/student_STARTUP/miniImageNet_source/$target_testset\_unlabeled_20
done
