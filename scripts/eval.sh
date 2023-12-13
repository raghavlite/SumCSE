#!/usr/bin/env bash
# path=sjtu-lit/SynCSE-partial-RoBERTa-base
path=${1}
# path=result/my-sup-simcse-roberta-base_sjtu-lit/SynCSE-partial-NLI
python simcse_to_huggingface.py --path ${path}
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
    --model_name_or_path ${path} \
    --pooler cls \
    --task_set full \
    --mode test