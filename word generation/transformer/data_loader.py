import pandas as pd
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, csv_file, max_length):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length
        self.tokenizer = self.build_tokenizer()
    
    def build_tokenizer(self):
        # Build a simple tokenizer from the dataset
        vocab = set()
        for content in self.data['newscontents']:
            vocab.update(content.split())
        return {word: idx for idx, word in enumerate(vocab, start=1)}  # Start from 1, 0 is reserved for padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = self.data.iloc[idx]['newscontents']
        tokens = self.tokenize(content)
        return torch.tensor(tokens, dtype=torch.long), self.data.iloc[idx]['label']

    def tokenize(self, text):
        tokens = text.split()
        token_ids = [self.tokenizer.get(token, 0) for token in tokens]  # Use 0 for unknown tokens
        return token_ids[:self.max_length] + [0] * (self.max_length - len(token_ids))  # Padding