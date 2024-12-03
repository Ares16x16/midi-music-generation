import torch
import torch.nn.functional as F
from transformer import Transformer
from dataset import MidiDataset
from config import *
import pandas as pd


def calculate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch["x"], batch["y"]
            inputs, targets = inputs.to(device), targets.to(device)
            logits, loss = model(inputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


def calculate_metrics(model, dataloader, device):
    perplexity = calculate_perplexity(model, dataloader, device)
    return {"Perplexity": perplexity}


def main():
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBED_DIM,
        seq_length=SEQ_LEN,
        num_heads=TRANSFORMER_HEADS,
        num_layers=TRANSFORMER_LAYERS,
    ).to(DEVICE)

    model.load_state_dict(torch.load("output\model_adl_remi_plus\model_epoch_5.pt"))
    model.eval()

    df = pd.read_csv(ENCODING_DIR + "/df.csv")
    val_df = df.iloc[int(len(df) * TRAIN_TEST_RATIO) :, :]
    eval_data = MidiDataset(val_df)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_data, batch_size=BATCH_SIZE, shuffle=False
    )

    metrics = calculate_metrics(model, eval_dataloader, DEVICE)
    print(f"Perplexity: {metrics['Perplexity']}")


if __name__ == "__main__":
    main()
