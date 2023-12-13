#!/bin/bash
#! IMP: changed a few parameters - batch size, eval_steps
#!  first argument should be number fo gpus. second should be input dataset

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=${2}
shift
shift
# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(($SLURM_JOBID%62536+2500))

# Allow multiple threads
export OMP_NUM_THREADS=1

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"



if [[ ${NUM_GPU} -eq 2 ]];
then
   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed 42 \
    --do_mlm \
    --hard_negative_weight 0 \
    "$@"
elif [[ ${NUM_GPU} -eq  4 ]];
then
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed 42 \
    --do_mlm \
    --hard_negative_weight 0 \
    "$@" 
else
   python train.py \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed 42 \
    --do_mlm \
    --hard_negative_weight 0 \
    "$@"
fi



RESULT_DIR=${2}
./scripts/eval.sh $RESULT_DIR


