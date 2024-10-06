import json
import re
import torch
from torch.utils.data import Dataset
from collections import Counter


def load_text(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return text


def load_mappings(token_id_path, id_token_path):
    with open(token_id_path, "r") as json_file:
        token_to_id_mapping = json.load(json_file)
        token_to_id_mapping = json.loads(token_to_id_mapping)

    with open(id_token_path, "r") as json_file:
        id_to_token_mapping = json.load(json_file)
        id_to_token_mapping = json.loads(id_to_token_mapping)

    id_to_token_mapping = {int(t[0]): t[1] for t in id_to_token_mapping.items()}

    return token_to_id_mapping, id_to_token_mapping


def save_mappings(token_to_id, id_to_token, token_id_path, id_token_path):
    with open(token_id_path, "w") as f:
        json.dump(token_to_id, f, ensure_ascii=False)

    with open(id_token_path, "w") as f:
        json.dump(id_to_token, f, ensure_ascii=False)


def build_vocab(text):
    tokens = tokenize(text)
    token_counts = Counter(tokens)
    token_to_id = {token: idx for idx, (token, _) in enumerate(token_counts.items())}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    return token_to_id, id_to_token


def tokenize(text):
    text = text.lower()
    tokens = re.split("(\n|,|\s|\.|\"|!|\?|-|“|'|:|’|”|;|‘)", text)
    return tokens


def tokenize_to_id(tokens, token_to_id_mapping):
    ids = [token_to_id_mapping[t] for t in tokens]
    return ids


def detokenize_to_text(ids, id_to_token_mapping):
    tokens = [id_to_token_mapping[t] for t in ids]
    return "".join(tokens)


def format_as_escaped_json(data):
    json_str = json.dumps(data)
    return json_str.replace('"', '"')


def add_missing_words(input_words, token_id_path, id_token_path):
    token_to_id, id_to_token = load_mappings(token_id_path, id_token_path)
    current_max_id = len(token_to_id)
    for word in input_words:
        if word not in id_to_token:
            id_to_token[str(current_max_id)] = word
            token_to_id[word] = str(current_max_id)
            current_max_id += 1

    token_to_id = format_as_escaped_json(token_to_id)
    id_to_token = format_as_escaped_json(id_to_token)
    save_mappings(token_to_id, id_to_token, token_id_path, id_token_path)

    return current_max_id + 1


class TinyStoryDataset(Dataset):
    def __init__(self, text, token_to_id_mapping, config):
        self.seq_len = config["SEQ_LEN"]
        self.text = text
        self.tokens = tokenize(text)
        self.token_to_id_mapping = token_to_id_mapping

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        x = tokenize_to_id(x, self.token_to_id_mapping)
        y = tokenize_to_id(y, self.token_to_id_mapping)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return {"x": x, "y": y}
