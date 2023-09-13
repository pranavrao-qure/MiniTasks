import os
import datetime
import codecs

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline

class RadBERTMultiClassMultiLabel(nn.Module):
    """
    RadBERTMultiClassMultiLabel: Model expects batches of natural language sentences, will
    classify reports with multiple label
    """
    def __init__(self, num_classes, checkpoint):
        super().__init__()
        self.num_classes = num_classes
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.transformer_encoder = AutoModel.from_pretrained(self.checkpoint)
        self.transformer_encoder_hidden_size = self.transformer_encoder.config.hidden_size
        self.linear_classifier = nn.Linear(self.transformer_encoder_hidden_size, self.num_classes)
    
    def forward(self, x):
        tokenized_inp = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')
        encoder_out = self.transformer_encoder(**tokenized_inp)
        logits = self.linear_classifier(encoder_out.last_hidden_state[:, 0, :])
        return logits

class MultiClassMultiLabel(nn.Module):
    def __init__(self, uncertain_label):
        super(MultiClassMultiLabel, self).__init__()
        self.uncertain_label = uncertain_label

    def forward(self, output, target):
        certain_mask = (target != self.uncertain_label)
        loss_func = nn.MultiLabelSoftMarginLoss(weight=certain_mask.type(torch.float))
        return loss_func(output, target)

class ReportTagsDataset(Dataset):
    def __init__(self, tags_csv_file, report_base_path, text_transform=None, target_transform=None):
        self.report_base_path = report_base_path
        self.tags_csv_file = tags_csv_file

        self.tags_df = pd.read_csv(self.tags_csv_file)
        self.column_names = list(self.tags_df.columns.values)
        self.column_names[0] = 'filename'
        self.tags_df.columns = self.column_names

        self.text_transform = text_transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.tags_df.shape[0]
    
    def __getitem__(self, index):
        report_path = os.path.join(self.report_base_path, self.tags_df.iloc[index, 0].split('/')[-1] + '.txt')
        #report_text = open(report_path).read()
        report_text = codecs.open(report_path, 'r', encoding='utf-8', errors='ignore').read()
        target_list = torch.Tensor(list(self.tags_df.iloc[index][1:]))
        return report_text, target_list

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = radbert_multi_model(inputs)
        # Compute the loss and its gradients
        loss = multiclass_multilabel_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

if __name__ == '__main__':
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    torch.set_printoptions(linewidth=200)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    checkpoint = 'UCSD-VA-health/RadBERT-RoBERTa-4m'
    radbert_multi_model = RadBERTMultiClassMultiLabel(322, checkpoint)
    multiclass_multilabel_loss = MultiClassMultiLabel(-100)

    train_reports_base_path = '/home/users/pranav.rao/MiniTasks/Radbert/data/train'
    test_reports_base_path = '/home/users/pranav.rao/MiniTasks/Radbert/data/test'
    train_tags_file = '/home/users/pranav.rao/Downloads/report_tags_25k_train.csv'
    test_tags_file = '/home/users/pranav.rao/Downloads/report_tags_25k_train.csv'

    train_data = ReportTagsDataset(train_tags_file, train_reports_base_path)
    test_data = ReportTagsDataset(test_tags_file, test_reports_base_path)

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

    lr = 3e-5
    beta1 = 0.9
    beta2 = 0.99
    l2_weight_decay = 0.01
    optimizer = torch.optim.Adam(radbert_multi_model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=l2_weight_decay)

    total_epochs = 5
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    for epoch in range(total_epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        print('EPOCH {}:'.format(epoch_number + 1))
        radbert_multi_model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)
        # Set the model to evaluation mode, disabling dropout and using population, statistics for batch normalization
        running_vloss = 0.0
        radbert_multi_model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(train_dataloader):
                vinputs, vlabels = vdata
                voutputs = radbert_multi_model(vinputs)
                vloss = multiclass_multilabel_loss(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()
        model_path = '/home/users/pranav.rao/MiniTasks/Radbert/ModelPool/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(radbert_multi_model.state_dict(), model_path)
        epoch_number += 1
