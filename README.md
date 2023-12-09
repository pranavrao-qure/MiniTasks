# MiniTasks
Repo for misc scripts


RadBERT/
    1. Pretraining Task:
        a. Script run_mlm.sh :
            Dataset at /home/users/pranav.rao/MiniTasks/Radbert/data/pretrain_data/full_train_data.txt (@TODO: report cleaning script )
            Output model : /models/user_checkpoints/pranav.rao/RadBert/PreTrainModelDir/RadBert/20231007/
        b. roberta_pretraining_mlm.py

    2. Finetuning Task:
        a. radbert_finetuning.py
            Tags extracted from python script (), 2 Million reports. Dataset in format /home/users/pranav.rao/Downloads/report_tags_25k_train.csv
            To train on a subset of labels give the subset in labels_file argument (/home/users/pranav.rao/MiniTasks/Radbert/labels_top50_withloc.txt)
            Sample run command: python3 radbert_finetuning_lightning.py 0 labels_top50_withloc.txt 0.00003 /models/user_checkpoints/pranav.rao/RadBert/PreTrainModelDir/RadBert/20231007/(optional)
        b. radbert_finetuning_lightning.py
            Same script with Lightning module implemented and training loop replaced by pl.Trainer