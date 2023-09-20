import os, random
import codecs

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def return_collate_fn(tokenizer):
    def tokenizer_collate_fn(data):
        """
        data: List of tuples (input, label), where input is a list of strings and label is a 0-1 vector
        """
        x, labels = zip(*data)
        inputs = tokenizer(x, padding=True, truncation=True, return_tensors='pt')
        labels = torch.stack(labels)
        return inputs, labels
    return tokenizer_collate_fn