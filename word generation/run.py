import yaml
import torch
from transformer.data_loader import (
    load_text,
    load_mappings,
    TinyStoryDataset,
)
from transformer.model import Transformer
from transformer.trainer import Trainer


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
TOKEN_ID_PATH = config["TOKEN_ID_PATH"]
ID_TOKEN_PATH = config["ID_TOKEN_PATH"]
DEVICE = config["DEVICE"]
MODEL_DIR = config["MODEL_DIR"]
SEQ_LEN = config["SEQ_LEN"]
BATCH_SIZE = config["BATCH_SIZE"]


token_to_id_mapping, id_to_token_mapping = load_mappings(TOKEN_ID_PATH, ID_TOKEN_PATH)
model = Transformer(
    vocab_size=config["VOCAB_SIZE"],
    embedding_dim=config["EMBED_DIM"],
    seq_length=SEQ_LEN,
    num_heads=config["TRANSFORMER_HEADS"],
    num_layers=config["TRANSFORMER_LAYERS"],
)

model.load_state_dict(torch.load(f"{MODEL_DIR}/init_model.pt"))
model.to(DEVICE)
model.eval()

evaluate_sample = ["Hi, my name is John."]

evaluate_data = TinyStoryDataset(" ".join(evaluate_sample), token_to_id_mapping, config)
evaluate_loader = torch.utils.data.DataLoader(
    evaluate_data, batch_size=BATCH_SIZE, shuffle=False
)

trainer = Trainer(model, None, evaluate_loader, None, DEVICE, MODEL_DIR)

results = trainer.evaluate(
    evaluate_sample, token_to_id_mapping, id_to_token_mapping, new_tokens=10
)

print("\nGenerated text:")
for result in results:
    print(result)
