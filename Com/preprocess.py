import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import re
from padding import padding_data

class CustomDataset(Dataset):
    def __init__(self, ids, code, message, labels):
        self.ids = ids
        self.code = code
        self.message = message
        self.labels = labels
    
    def __len__(self):
        return len(self.code)
    
    def __getitem__(self, idx):
        commit_hash = self.ids[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        code = self.code[idx]
        message = self.message[idx]
        code = torch.tensor(code)
        message = torch.tensor(message)

        return {
            'commit_hash': commit_hash,
            'code': code,
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

def preprocess_data(params):
    if params.train is True:
        # Load train data
        train_data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, messages, codes = train_data
    
    elif params.predict is True:
        # Load predict data
        predict_data = pickle.load(open(params.predict_data, 'rb'))
        ids, labels, messages, codes = predict_data

    if params.do_valid is True and params.train is True:
        val_data = pickle.load(open(params.test_data, 'rb'))
        val_ids, val_labels, val_messages, val_codes = val_data

    dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
    dict_msg, dict_code = dictionary

    pad_msg = padding_data(data=messages, dictionary=dict_msg, params=params, type='msg')        
    pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')

    if params.do_valid is True and params.train is True:
        val_pad_msg = padding_data(data=val_messages, dictionary=dict_msg, params=params, type='msg')        
        val_pad_code = padding_data(data=val_codes, dictionary=dict_code, params=params, type='code')

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(ids, pad_code, pad_msg, labels)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size)

    if params.do_valid is True and params.train is True:
        val_code_dataset = CustomDataset(val_ids, val_pad_code, val_pad_msg, val_labels)
        val_code_dataloader = DataLoader(val_code_dataset, batch_size=params.batch_size)

    if params.do_valid is True and params.train is True:
        return (code_dataloader, val_code_dataloader, dict_msg, dict_code)
    else:
        return (code_dataloader, dict_msg, dict_code)
