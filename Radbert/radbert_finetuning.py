import sys
import datetime
from itertools import product

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from utils import seed_everything, ReportTagsDataset, return_collate_fn
from MultiClassMultiLabel import RadBERTMultiClassMultiLabel, MultiClassMultiLabelLoss

def train_one_epoch(epoch_index, tb_writer, device, dataloader, loss_fn, optimizer_fn):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_fn.zero_grad()
        outputs = radbert_multi_model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels, penalize_certainity=True)
        loss.backward()
        optimizer_fn.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
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

    if len(sys.argv) < 4:
        print("Usage: python3 radbert_finetuning.py device(cuda:1) labels_file lr checkpoint(optional)")
        exit(0)

    device = sys.argv[1]
    labels_file = sys.argv[2]
    lr = float(sys.argv[3])
    if len(sys.argv) > 4:
        checkpoint = sys.argv[4]
    else:
        checkpoint = 'UCSD-VA-health/RadBERT-RoBERTa-4m'

    print(f"Using {device} device")
    labels_subset = [e.strip() for e in open(labels_file, 'r').readlines()]
    num_classes = len(labels_subset)
    print("The labels being used for classification objective are:\n" + '\n'.join(list(map(lambda x:', '.join(x), [labels_subset[i:i+10] for i in range(0, num_classes, 10)]))) + '\n')

    radbert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    radbert_multi_model = RadBERTMultiClassMultiLabel(num_classes, checkpoint).to(device)
    multiclass_multilabel_loss = MultiClassMultiLabelLoss(-100, device)

    train_reports_base_path = '/home/users/pranav.rao/MiniTasks/Radbert/data/train'
    test_reports_base_path = '/home/users/pranav.rao/MiniTasks/Radbert/data/test'
    train_tags_file = '/home/users/pranav.rao/Downloads/report_tags_25k_train.csv'
    test_tags_file = '/home/users/pranav.rao/Downloads/report_tags_25k_test.csv'

    train_data = ReportTagsDataset(train_tags_file, train_reports_base_path, labels_subset=labels_subset)
    test_data = ReportTagsDataset(test_tags_file, test_reports_base_path, labels_subset=labels_subset)

    tokenizer_collate_fn = return_collate_fn(radbert_tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=32, num_workers=2, collate_fn=tokenizer_collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=tokenizer_collate_fn, num_workers=2)

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
        avg_loss = train_one_epoch(epoch, writer, device, train_dataloader, multiclass_multilabel_loss, optimizer)
        model_path = '/home/users/pranav.rao/MiniTasks/Radbert/ModelPool/model_{}_{}'.format(timestamp, epoch)
        torch.save(radbert_multi_model.state_dict(), model_path)

        # Set the model to evaluation mode, disabling dropout and using population, statistics for batch normalization
        running_vloss = 0.0
        radbert_multi_model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_dataloader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = radbert_multi_model(vinputs)
                vloss = multiclass_multilabel_loss(voutputs, vlabels, penalize_certainity=True)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()
