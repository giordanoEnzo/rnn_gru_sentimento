import random
import torch
from torch.utils.data import DataLoader, TensorDataset

vocabulario = {}


def tokenize(text):
    return text.lower().split()


def construir_vocabulario(data):
    idx = 0
    for _, text in data:
        for token in text:
            if token not in vocabulario:
                vocabulario[token] = idx
                idx += 1
    return vocabulario


def tokens_to_ids(data, vocabulario):
    id_data = []
    for label, text in data:
        token_ids = [vocabulario[token] for token in text if token in vocabulario]
        label_id = 1 if label == 'positive' else 0
        id_data.append((label_id, token_ids))
    return id_data


def pad_sequences(data, max_len):
    padded_data = []
    for label, token_ids in data:
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        padded_data.append((label, token_ids))
    return padded_data


def dividir_treino_validacao(data, split_ratio=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]


def preparar_dataloader(data, batch_size, drop_last=False):
    labels = torch.tensor([item[0] for item in data], dtype=torch.long)
    sequences = torch.tensor([item[1] for item in data], dtype=torch.long)

    dataset = TensorDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
