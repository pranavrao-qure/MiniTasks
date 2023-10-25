import sys
from npy_append_array import NpyAppendArray
import codecs

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import seed_everything
from MultiClassMultiLabel import RadBERTMultiClassMultiLabel

def get_sentence_embeddings(tokenizer, model, input):
    with torch.no_grad():
        tokenized_inp = tokenizer(input, padding=True, truncation=True, return_tensors='pt').to(device)
        encoder_out = model.transformer_encoder(**tokenized_inp)
    return encoder_out.last_hidden_state[:, 0, :].cpu().numpy()

if __name__ == '__main__':
    seed_everything(42)
    torch.set_printoptions(linewidth=200)

    if len(sys.argv) < 6:
        print("Usage: python3 radbert_finetuning.py device(cuda:1) labels_file checkpoint(optional) reports_list embeddings_output_file")
        exit(0)

    device = sys.argv[1]
    labels_file = sys.argv[2]
    checkpoint = sys.argv[3]
    reports_list = sys.argv[4]
    embeddings_output_file = sys.argv[5]

    print(f"Using {device} device")
    labels_subset = [e.strip() for e in open(labels_file, 'r').readlines()]
    num_classes = len(labels_subset)
    print("The labels being used for classification objective are:\n" + '\n'.join(list(map(lambda x:', '.join(x), [labels_subset[i:i+10] for i in range(0, num_classes, 10)]))) + '\n')

    base_model = 'UCSD-VA-health/RadBERT-RoBERTa-4m'
    radbert_tokenizer = AutoTokenizer.from_pretrained(base_model)
    radbert_multi_model = RadBERTMultiClassMultiLabel(num_classes, base_model).to(device)
    radbert_multi_model.load_state_dict(torch.load(checkpoint))

    batch_size = 32
    with open(reports_list, 'r') as reports:
        #with open(embeddings_output_file, 'w') as outfile:
        with NpyAppendArray(embeddings_output_file, delete_if_exists=True) as outfile:
            report_names, report_batch = list(), list()
            for idx, report_path in enumerate(reports):
                report_path = report_path.strip()
                report_names.append(report_path.split('/')[-1])
                report_batch.append(codecs.open(report_path + '.txt', 'r', encoding='utf-8', errors='ignore').read())
                if len(report_batch) == batch_size:
                    embeddings = get_sentence_embeddings(radbert_tokenizer, radbert_multi_model, report_batch)
                    #outfile.append('\n'.join([f'{report_name},{embedding}' for report_name, embedding in zip(report_names, embeddings)]) + '\n')
                    outfile.append(embeddings)
                    report_names, report_batch = list(), list()
                if idx % (batch_size * 50) == 0:
                    print(f"Done with {idx/batch_size} batches")
            if len(report_batch) > 0:
                embeddings = get_sentence_embeddings(radbert_tokenizer, radbert_multi_model, report_batch)
                #outfile.write('\n'.join([f'{report_name},{embedding.tobytes()}' for report_name, embedding in zip(report_names, embeddings)]) + '\n')
                outfile.append(embeddings)
                report_names, report_batch = list(), list()
