import pandas as pd
import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data.reset_index(drop=True)  # Reset index for consistency
        self.max_length = max_length
        self.tokenizer = self.build_tokenizer()
    
    def build_tokenizer(self):
        vocab = set()
        for content in self.data['newscontents']:
            vocab.update(content.split())
        return {word: idx for idx, word in enumerate(vocab, start=1)}  # Start token indices from 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = self.data.iloc[idx]['newscontents']
        tokens = self.tokenize(content)

        # Padding or truncating tokens to max_length
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]  # Truncate to max_length
        
        return torch.tensor(tokens, dtype=torch.long)

    def tokenize(self, text):
        tokens = text.split()
        token_ids = [self.tokenizer.get(token, 0) for token in tokens]
        return token_ids  # No padding here, handled in __getitem__