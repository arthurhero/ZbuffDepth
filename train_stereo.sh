#!/bin/bash

mkdir -p logs
mkdir -p runs
#rm -rf runs/*

KITTI_DIR="/PATH-TO-KITTI/"
CHECKPOINT_FOLDER="logs"
EXP_FOLDER=test_stereo
TENSORBOARD=runs/$EXP_FOLDER
EIGNE_VAL="eigen_val_files.txt"

# Generate validation set
python3 generate_valid.py \
      --kitti_path $KITTI_DIR \
      --eigen_val_path $EIGNE_VAL \
      --max_size 500 \
      --seed 42 \
      --image_path "data/image_sequence_valid.txt" \
      --depth_path "data/depth_sequence_valid.txt"

# Train egoDepth
python3 run_stereo.py \
      --model orig\
      --log_directory $CHECKPOINT_FOLDER \
      --description 'orig stereo' \
      --experiment_directory $EXP_FOLDER \
      --summary_path $TENSORBOARD \
      --kitti_path $KITTI_DIR \
      --shuffle_seed 76 \
      --batch_size 3 \
      --learning_rate 0.00001 \
      --num_epochs 20 \
      --log_rate 200 \
      --image_height 352 \
      --image_width 1216 \
      --smooth_lambda 0.0 \
      --ssim_lambda 2.0 \
      --nd_lambda 2.0 \
      --zbuffer \
      --skymask_distance 100.0 \
#      --gordon_zbuffer
#      --sky_only_smooth
#      --include_nd
