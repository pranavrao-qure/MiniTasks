#!/bin/bash

python3 roberta_pretraining.py --output_dir /models/user_checkpoints/pranav.rao/RadBert/PreTrainModelDir/RadBert/ --model_name_or_path UCSD-VA-health/RadBERT-RoBERTa-4m --mlm --train_data_file ./data/pretrain_data/full_train_data.txt --whole_word_mask True --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 4 --learning_rate 1e-4 --num_train_epochs 5 --save_steps 625 --logging_dir ./runs/pretraining/radbert/ --seed 42
