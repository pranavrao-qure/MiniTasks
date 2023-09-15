import os, sys
import time, datetime
import codecs
from itertools import product

import random
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

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RadBERTMultiClassMultiLabel(nn.Module):
    """
    RadBERTMultiClassMultiLabel: Model expects batches of natural language sentences, will
    classify reports with multiple label
    """
    def __init__(self, num_classes, checkpoint, device):
        super().__init__()
        self.num_classes = num_classes
        self.checkpoint = checkpoint
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.transformer_encoder = AutoModel.from_pretrained(self.checkpoint)
        self.transformer_encoder_hidden_size = self.transformer_encoder.config.hidden_size
        self.linear_classifier = nn.Linear(self.transformer_encoder_hidden_size, self.num_classes)
    
    def forward(self, x):
        tokenized_inp = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt').to(self.device)
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
    def __init__(self, tags_csv_file, report_base_path, labels_subset=None, text_transform=None, target_transform=None):
        self.report_base_path = report_base_path
        self.tags_csv_file = tags_csv_file

        self.tags_df = pd.read_csv(self.tags_csv_file)
        self.column_names = list(self.tags_df.columns.values)
        self.column_names[0] = 'filename'
        self.tags_df.columns = self.column_names

        self.labels_subset = labels_subset
        self.text_transform = text_transform
        self.target_transform = target_transform

    def __len__(self):
        return self.tags_df.shape[0]

    def __getitem__(self, index):
        report_path = os.path.join(self.report_base_path, self.tags_df.iloc[index, 0].split('/')[-1] + '.txt')
        #report_text = open(report_path).read()
        report_text = codecs.open(report_path, 'r', encoding='utf-8', errors='ignore').read()
        if self.labels_subset is None:
            target_list = torch.Tensor(list(self.tags_df.iloc[index][1:]))
        else:
            target_list = torch.Tensor(list(self.tags_df[self.labels_subset].iloc[index]))
        return report_text, target_list

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = radbert_multi_model(inputs)
        # Compute the loss and its gradients
        loss = multiclass_multilabel_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

if __name__ == '__main__':
    seed_everything(42)

    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.width', 1000)
    torch.set_printoptions(linewidth=200)

    #device = (
    #    "cuda"
    #    if torch.cuda.is_available()
    #    else "mps"
    #    if torch.backends.mps.is_available()
    #    else "cpu"
    #)
    #device='cuda:1'

    if len(sys.argv) < 4:
        print("Usage: python3 radbert_finetuning.py device(cuda:1) labels_file lr")
        exit(0)

    device = sys.argv[1]
    labels_file = sys.argv[2]
    lr = float(sys.argv[3])

    print(f"Using {device} device")
    checkpoint = 'UCSD-VA-health/RadBERT-RoBERTa-4m'
    #labels_subset = "normal tuberculosis opacity bronchialdilation density parenchymalopacity ett aorticenlargement mediastinalwidening mediastinalmass\
    #        copd prominentbronchovascularmarkings bronchitis markings vascularprominence interval interstitiallungdisease bluntedcp effusion cardiomegaly\
    #        consolidation subtle_normal peffusion lineandtube thickening haziness hilarprominence hilar inhomogenousopacity rotation\
    #        calcification unfoldedaorta bandlikeopacity aorticcalcification aorticknucklecalcification fibrosis suture cardiacshift degenspine nodule\
    #        pneumonia inspiration fracture pneumonitis justfibrosis lesion nonaorticcalcification tuberculosispure pleuralthickening feedingtube".split()
    labels_subset = [e.strip() for e in open(labels_file, 'r').readlines()]
    print("The labels being used for classification objective are: " + '\n'.join([', '.join(labels_subset[:10]), ', '.join(labels_subset[10:20]),\
                                                                                  ', '.join(labels_subset[20:30]), ', '.join(labels_subset[30:])]) + '\n')
    num_classes = len(labels_subset)
    radbert_multi_model = RadBERTMultiClassMultiLabel(num_classes, checkpoint, device).to(device)
    multiclass_multilabel_loss = MultiClassMultiLabel(-100)

    train_reports_base_path = '/home/users/pranav.rao/MiniTasks/Radbert/data/train'
    test_reports_base_path = '/home/users/pranav.rao/MiniTasks/Radbert/data/test'
    train_tags_file = '/home/users/pranav.rao/Downloads/report_tags_25k_train.csv'
    test_tags_file = '/home/users/pranav.rao/Downloads/report_tags_25k_test.csv'

    train_data = ReportTagsDataset(train_tags_file, train_reports_base_path, labels_subset=labels_subset)
    test_data = ReportTagsDataset(test_tags_file, test_reports_base_path, labels_subset=labels_subset)

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

    #lr = 3e-5
    beta1 = 0.9
    beta2 = 0.99
    l2_weight_decay = 0.01
    print(f'Hyper params for optimizer: {lr}, ({beta1}, {beta2}), {l2_weight_decay}')
    optimizer = torch.optim.Adam(radbert_multi_model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=l2_weight_decay)

    total_epochs = 3
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    for epoch in range(total_epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        print('EPOCH {}:'.format(epoch + 1))
        radbert_multi_model.train(True)
        avg_loss = train_one_epoch(epoch, writer)
        model_path = '/home/users/pranav.rao/MiniTasks/Radbert/ModelPool/model_{}_{}'.format(timestamp, epoch)
        torch.save(radbert_multi_model.state_dict(), model_path)

        # Set the model to evaluation mode, disabling dropout and using population, statistics for batch normalization
        running_vloss = 0.0
        radbert_multi_model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_dataloader):
                vinputs, vlabels = vdata
                vlabels = vlabels.to(device)
                voutputs = radbert_multi_model(vinputs)
                vloss = multiclass_multilabel_loss(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()
