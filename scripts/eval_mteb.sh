#!/usr/bin/env bash

path=${1}

CUDA_VISIBLE_DEVICES=0,1 python evaluation_MTEB.py \
    --model_name_or_path ${path}