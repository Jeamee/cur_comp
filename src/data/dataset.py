from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


import numpy as np
import torch


def prepare_input(tokenizer, text, feature_text):
    inputs = tokenizer(text, feature_text,
                           add_special_tokens=True,
                           return_offsets_mapping=False,
                           return_tensors="pt"
                      )
    for key, val in inputs.items():
        inputs[key] = torch.squeeze(val)
    
    if "token_type_ids" in inputs:
        inputs["token_type_ids"] = torch.logical_not(inputs["token_type_ids"].byte()).long()
        
    inputs["attention_mask"] = inputs["attention_mask"]
    
    return inputs


def create_label(tokenizer, text, feature_text, annotation_length, location_list):
    encoded = tokenizer(text, feature_text,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) == None)[0]
    question_idxes = np.where(np.array(encoded.sequence_ids()) == 1)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -100
    label[question_idxes[1:]] = 1
    label[question_idxes[0]] = 2
    sequence_mask = torch.tensor(np.array(encoded.sequence_ids()) == 0, dtype=torch.bool)

    if annotation_length != 0:
        for location in location_list:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx] = 2
                    label[start_idx + 1:end_idx] = 1
    return torch.tensor(label, dtype=torch.long), sequence_mask


class TrainDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.tokenizer, 
                self.pn_historys[item], 
                self.feature_texts[item])
        label, sequence_mask = create_label(self.tokenizer, 
                self.pn_historys[item], 
                self.feature_texts[item],
                self.annotation_lengths[item], 
                self.locations[item])
        inputs["targets"] = label
        inputs["sequence_mask"] = sequence_mask
        assert len(inputs["input_ids"]) == len(sequence_mask)
        return inputs


def collate_fn(batch):
    output = dict()
    for key, val in batch[0].items():
        if key == "targets":
            padding_value = -100
        elif key == "sequence_mask":
            padding_value = False
        else:
            padding_value = 0
        output[key] = pad_sequence([torch.tensor(sample[key]) for sample in batch], batch_first=True, padding_value=padding_value)
    
    return output