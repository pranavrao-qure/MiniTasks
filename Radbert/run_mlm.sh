#!/bin/bash

python3 roberta_pretraining_mlm.py --output_dir /models/user_checkpoints/pranav.rao/RadBert/PreTrainModelDir/RadBert/20231007/ --model_name_or_path UCSD-VA-health/RadBERT-RoBERTa-4m --train_file ./data/pretrain_data/full_train_data.txt --validation_split_percentage 5 --do_train --do_eval --per_device_train_batch_size 16 --gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 5 --save_steps 625 --logging_dir ./runs/pretraining/radbert/ --seed 42
