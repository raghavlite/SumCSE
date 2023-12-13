#!/usr/bin/env bash
path=${1}
CUDA_VISIBLE_DEVICES=0 python evaluation_anisotropy.py \
    --model_name_or_path ${path} \
    --pooler cls 
