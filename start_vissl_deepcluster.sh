#!/bin/bash
NUM_TH=8

export OMP_NUM_THREADS=$NUM_TH
export OPENBLAS_NUM_THREADS=$NUM_TH
export MKL_NUM_THREADS=$NUM_TH
export VECLIB_MAXIMUM_THREADS=$NUM_TH
export NUMEXPR_NUM_THREADS=$NUM_TH
export NUMEXPR_MAX_THREADS=$NUM_TH

python run_distributed_engines.py \
 hydra.verbose=true \
 config=deepclusterv2_2crops_resnet \
 config.CHECKPOINT.DIR="./checkpoints/model_checkpoints/deepcluster_covid_pretrained" \
 config.MODEL.WEIGHTS_INIT.PARAMS_FILE="./checkpoints/model_checkpoints/resnet_torchvision/resnet50-19c8e357.pth" \
 config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." \
 config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \
 config.TENSORBOARD_SETUP.USE_TENSORBOARD=true
