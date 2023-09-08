import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import RobertaTokenizer, AutoTokenizer
import numpy as np
import pandas as pd
import re
from utils import load_data

class CustomDataset(Dataset):
    def __init__(self, ids, added_code_list, removed_code_list, message_list, pad_token_id, features, labels, max_seq_length):
        self.added_code_list = added_code_list
        self.removed_code_list = removed_code_list
        self.message_list = message_list
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.labels = labels
        self.ids = ids
        self.features = features
    
    def __len__(self):
        return len(self.added_code_list)
    
    def __getitem__(self, idx):
        # truncate the code sequence if it exceeds max_seq_length
        added_code = self.added_code_list[idx][:self.max_seq_length]
        
        # pad the code sequence if it is shorter than max_seq_length
        num_padding = self.max_seq_length - len(added_code)
        added_code += [self.pad_token_id] * num_padding

        # truncate the code sequence if it exceeds max_seq_lengthadded
        removed_code = self.removed_code_list[idx][:self.max_seq_length]
        
        # pad the code sequence if it is shorter than max_seq_length
        num_padding = self.max_seq_length - len(removed_code)
        removed_code += [self.pad_token_id] * num_padding

        message = self.message_list[idx]

        feature = self.features[idx]
        feature = np.vstack(feature).astype(np.float32)
        feature = torch.tensor(feature, dtype=torch.float32).squeeze()

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        added_code = torch.tensor(added_code)
        removed_code = torch.tensor(removed_code)
        message = torch.tensor(message)

        return {
            'added_code': added_code,
            'removed_code': removed_code,
            'feature': feature,
            'message': message,
            'labels': labels
        }
    
def str_to_dict(string: str) -> dict:  # sourcery skip: avoid-builtin-shadow
    pattern = r'added_code:\s*(?P<added_code>.*?)\s*removed_code:\s*(?P<removed_code>.*?)\s*$'
    if match := re.match(pattern, string, re.DOTALL):
        dict = match.groupdict()
        dict['added_code'] = dict['added_code'].strip()
        dict['removed_code'] = dict['removed_code'].strip()
    else:
        dict = {"added_code": "", "removed_code": ""}
    return dict
    
def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return ' '.join([line_split[i] for i in range(max_length)])
    else:
        return line
    
def convert_msg_to_label(pad_msg, dict_msg):
    nrows, ncols = pad_msg.shape
    labels = []
    for i in range(nrows):
        column = list(set(list(pad_msg[i, :])))
        label = np.zeros(len(dict_msg))
        for c in column:
            label[c] = 1
        labels.append(label)
    return np.array(labels)

def mapping_dict_msg(pad_msg, dict_msg):
    return np.array(
        [np.array([dict_msg[w.lower()] if w.lower() in dict_msg.keys() else dict_msg['<NULL>'] for w in line.split(' ')]) for line in pad_msg])

def padding_message(data, max_length):
    return [padding_length(line=d, max_length=max_length) for d in data]

def preprocess_data(params, max_seq_length: int = 512):
    if params.train is True:
        # Load train data
        train_data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, messages, codes = train_data
        
        features = load_data(params)

    elif params.predict is True:
        # Load predict data
        predict_data = pickle.load(open(params.predict_data, 'rb'))
        ids, labels, messages, codes = predict_data

        features = load_data(params)

    # Combine train data and test data into data
    labels = list(labels)

    # CodeBERT tokenizer
    codeBERT_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # BERT tokenizer
    BERT_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Preprocessing messages
    messages_list = []

    for message in messages:
        message = BERT_tokenizer(message, padding='max_length', max_length=max_seq_length)
        messages_list.append(message["input_ids"])

    # Preprocessing codes
    added_code_list = []
    removed_code_list = []

    for commit in codes:
        added_code_tokens = [codeBERT_tokenizer.cls_token]
        removed_code_tokens = [codeBERT_tokenizer.cls_token]
        for hunk in commit:
            hunk = str_to_dict(hunk)
            added_code = " ".join(hunk["added_code"])
            removed_code = " ".join(hunk["removed_code"])
            added_code_tokens += codeBERT_tokenizer.tokenize(added_code) + [codeBERT_tokenizer.sep_token]
            removed_code_tokens += codeBERT_tokenizer.tokenize(removed_code) + [codeBERT_tokenizer.sep_token]
        added_code_tokens += [codeBERT_tokenizer.eos_token]
        removed_code_tokens += [codeBERT_tokenizer.eos_token]
        added_tokens_ids = codeBERT_tokenizer.convert_tokens_to_ids(added_code_tokens)
        removed_tokens_ids = codeBERT_tokenizer.convert_tokens_to_ids(removed_code_tokens)
        added_code_list.append(added_tokens_ids)
        removed_code_list.append(removed_tokens_ids)

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(ids, added_code_list, removed_code_list, messages_list, codeBERT_tokenizer.pad_token_id, features, labels, max_seq_length)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size)

    return (code_dataloader, dict_msg, dict_code)