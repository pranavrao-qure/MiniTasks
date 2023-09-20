import sys
import datetime

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from utils import seed_everything, ReportTagsDataset, return_collate_fn
from MultiClassMultiLabel import RadBERTMultiClassMultiLabel, MultiClassMultiLabelLoss

class RadBERTMultiClassMultiLabelTrainer(pl.LightningModule):
    def __init__(self, model, optimizer_lr):
        super().__init__()
        self.model = model
        self.optimizer_lr = optimizer_lr
        self.loss_fn = MultiClassMultiLabelLoss(-100)

    def compute_loss(self, inputs, labels):
        outputs = self.model(inputs)
        return self.loss_fn(outputs, labels, penalize_certainity=False)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.compute_loss(inputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.compute_loss(inputs, labels)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        beta1 = 0.9
        beta2 = 0.99
        l2_weight_decay = 0.01
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_lr, betas=(beta1, beta2), weight_decay=l2_weight_decay)
        return optimizer

if __name__ == '__main__':
    seed_everything(42)

    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.width', 1000)
    torch.set_printoptions(linewidth=200)

    if len(sys.argv) < 4:
        print("Usage: python3 radbert_finetuning.py device(list of cuda devices, for eg: 1,2) labels_file lr")
        exit(0)

    devices = list(map(int, sys.argv[1].split(',')))
    labels_file = sys.argv[2]
    lr = float(sys.argv[3])

    print(f"Using {devices} device")
    checkpoint = 'UCSD-VA-health/RadBERT-RoBERTa-4m'
    labels_subset = [e.strip() for e in open(labels_file, 'r').readlines()]
    num_classes = len(labels_subset)
    print("The labels being used for classification objective are:\n" + '\n'.join(list(map(lambda x:', '.join(x), [labels_subset[i:i+10] for i in range(0, num_classes, 10)]))) + '\n')

    radbert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    radbert_multi_model = RadBERTMultiClassMultiLabel(num_classes, checkpoint)

    train_reports_base_path = '/home/users/pranav.rao/MiniTasks/Radbert/data/train'
    test_reports_base_path = '/home/users/pranav.rao/MiniTasks/Radbert/data/test'
    train_tags_file = '/home/users/pranav.rao/Downloads/report_tags_25k_train.csv'
    test_tags_file = '/home/users/pranav.rao/Downloads/report_tags_25k_test.csv'

    train_data = ReportTagsDataset(train_tags_file, train_reports_base_path, labels_subset=labels_subset)
    test_data = ReportTagsDataset(test_tags_file, test_reports_base_path, labels_subset=labels_subset)

    tokenizer_collate_fn = return_collate_fn(radbert_tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=24, num_workers=2, collate_fn=tokenizer_collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=24, collate_fn=tokenizer_collate_fn, num_workers=2)
    radbert_multi_model_trainer = RadBERTMultiClassMultiLabelTrainer(radbert_multi_model, lr)


    total_epochs = 3
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = TensorBoardLogger(save_dir="/home/users/pranav.rao/MiniTasks/Radbert/runs", name=f"radbert_finetune_25k_{num_classes}_{lr}")

    trainer = pl.Trainer(accelerator='gpu', max_epochs=3, devices=devices, default_root_dir="/home/users/pranav.rao/MiniTasks/Radbert/LightningRootDir/", enable_checkpointing=True, logger=writer)
    trainer.fit(model=radbert_multi_model_trainer, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)