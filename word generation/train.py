import yaml
import torch
from torch.utils.data import DataLoader
from transformer.data_loader import (
    load_text,
    save_mappings,
    load_mappings,
    format_as_escaped_json,
    TinyStoryDataset,
    build_vocab,
    add_missing_words,
)
from transformer.trainer import Trainer
from transformer.model import (
    Transformer,
)

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

TEXTPATH = config["TEXTPATH"]
TOKEN_ID_PATH = config["TOKEN_ID_PATH"]
ID_TOKEN_PATH = config["ID_TOKEN_PATH"]
SEQ_LEN = config["SEQ_LEN"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
VOCAB_SIZE = config["VOCAB_SIZE"]
EMBED_DIM = config["EMBED_DIM"]
TRANSFORMER_HEADS = config["TRANSFORMER_HEADS"]
TRANSFORMER_LAYERS = config["TRANSFORMER_LAYERS"]
LR = config["LR"]
DEVICE = config["DEVICE"]
MODEL_DIR = config["MODEL_DIR"]
PLOT_DIR = config["PLOT_DIR"]

# Load data
main_text = load_text(TEXTPATH)
train_index = int(0.9 * len(main_text.split(" ")))
train_text = " ".join(main_text.split()[:train_index])
valid_text = " ".join(main_text.split()[train_index:])

# Load mappings
"""token_to_id_mapping, id_to_token_mapping = build_vocab(main_text)
escaped_token_to_id = format_as_escaped_json(token_to_id_mapping)
escaped_id_to_token = format_as_escaped_json(id_to_token_mapping)
save_mappings(escaped_token_to_id, escaped_id_to_token, "t2i.json", "i2t.json")"""
token_to_id_mapping, id_to_token_mapping = load_mappings(TOKEN_ID_PATH, ID_TOKEN_PATH)

# Datasets and DataLoaders
train_data = TinyStoryDataset(train_text, token_to_id_mapping, config)
valid_data = TinyStoryDataset(valid_text, token_to_id_mapping, config)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

# Model and optimizer
model = Transformer(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBED_DIM,
    seq_length=SEQ_LEN,
    num_heads=TRANSFORMER_HEADS,
    num_layers=TRANSFORMER_LAYERS,
)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Trainer
trainer = Trainer(
    model, optimizer, train_loader, valid_loader, DEVICE, MODEL_DIR, PLOT_DIR
)
trainer.train(EPOCHS)

# Evaluation
evaluate_sample = ["Hi, my name is John."]
# add_missing_words(evaluate_sample, "t2i.json", "i2t.json")
token_to_id_mapping, id_to_token_mapping = load_mappings("t2i.json", "i2t.json")

results = trainer.evaluate(
    evaluate_sample, token_to_id_mapping, id_to_token_mapping, new_tokens=10
)
for result in results:
    print(result)
