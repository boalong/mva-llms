import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

def tokenize_function(tokenizer, batch):
    inputs = tokenizer(
        batch["text"],
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    return {
        "input_ids": inputs.input_ids.squeeze(0).tolist(),
        "attention_mask": inputs.attention_mask.squeeze(0).tolist(),
    }

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"]),
            "attention_mask": torch.tensor(self.data[idx]["attention_mask"]),
        }