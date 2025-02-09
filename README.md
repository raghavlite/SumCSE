# SumCSE: Summary as a Transformation for Contrastive Learning  

This is the official code for **SumCSE**, a method that leverages summaries as transformations for contrastive learning.  

This repository is a fork from [SynCSE](https://github.com/hkust-nlp/SynCSE.git).  

## Updates  
- **[Feb 2024]**: Updated `requirements.txt`. The codebase uses older versions of `torch` and `transformers`. Ensure you install the correct versions before running the code.  

## Data  
### Downloading SumCSE Dataset  
Download the SumCSE dataset from:  
📥 [Google Drive](https://drive.google.com/file/d/1458du6o6-4ZRzCopbjt_-GsZprax4sZj/view?usp=sharing)  

### Directory Setup  
- Place the dataset in: `../Data/`  
- Create a results directory: `../result/`  

## Running & Evaluating SumCSE  
Use the following script to train and evaluate SumCSE:  

```sh
./scripts/simcse_train_test.sh --num_gpus 4 \
  --output_dir ../result/SumCSE/ \
  --model_name_or_path roberta-large \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 128 \
  --train_file ../Data/SumCSE.csv \
  --num_train_epochs 3
```

## Regenerating the SumCSE training dataset  
```vicuna_inference_transformation.py``` files can be used to create SumCSE transformation if you are interested in recreating SumCSE dataset.
