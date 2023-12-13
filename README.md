This is official code for SumCSE: Summary as a transformation for Contrastive Learning

This repository is a fork from https://github.com/hkust-nlp/SynCSE.git



Download SumCSE: https://drive.google.com/file/d/1458du6o6-4ZRzCopbjt_-GsZprax4sZj/view?usp=sharing

Place it in `../Data/`

Also create a `../result` directory


Run and Evaluate SumCSE:
```
./scripts/simcse_train_test.sh --num_gpus 4 --output_dir ../result/SumCSE/ --model_name_or_path roberta-large --learning_rate 1e-5 --per_device_train_batch_size 128 --train_file ../Data/SumCSE.csv --num_train_epochs 3
```
