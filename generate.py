import torch
import numpy as np
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile, Note, Instrument
from pathlib import Path
from transformer import Transformer
from config import *


# sample_file = None
sample_file = "test/carribean.mid"

# Load tokenizer
config = TokenizerConfig(
    use_programs=True, one_token_stream_for_programs=True, use_time_signatures=True
)
tokenizer = REMI(config)

midi = MidiFile(sample_file)
tokens = list(tokenizer(midi))  # Convert to tokens
tokens = np.array(tokens)


seed_sequence = tokens.tolist()
idx = (
    torch.tensor(seed_sequence, dtype=torch.long).unsqueeze(0).to(DEVICE)
)  # Shape (1, T)

model = Transformer(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBED_DIM,
    seq_length=SEQ_LEN,
    num_heads=TRANSFORMER_HEADS,
    num_layers=TRANSFORMER_LAYERS,
).to(DEVICE)

# Load the trained model weights
model.load_state_dict(torch.load("output/model_adl_remi_plus/model_epoch_5.pt"))
model.eval()

max_new_tokens = 500  # Adjust the number of tokens to generate

print(f"Shape of input tensor idx: {idx.shape}")
generated_idx = model.generate(
    idx,
    max_new_tokens=max_new_tokens,
)
print(f"Shape of generated tensor: {generated_idx.shape}")

generated_tokens = generated_idx.squeeze().cpu().numpy()
decoded_tokens = tokenizer.decode(generated_tokens)

decoded_tokens.dump_midi("sample/epoch5/carribean_5_3.mid")
