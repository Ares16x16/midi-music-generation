import streamlit as st
import torch
import numpy as np
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from transformer import Transformer
from config import *
from pathlib import Path

# Streamlit app
st.title("MIDI Music Generation")

# Mode selector
mode = st.radio("Select mode", ("Upload MIDI file", "Generate from scratch"))

# File uploader
if mode == "Upload MIDI file":
    uploaded_file = st.file_uploader("Upload a MIDI file", type=["mid"])
else:
    uploaded_file = None

# Input for number of new tokens
max_new_tokens = st.number_input(
    "Number of new tokens to generate", min_value=1, value=500
)

if mode == "Upload MIDI file" and uploaded_file is not None:
    # Save uploaded file to a temporary location
    temp_file_path = Path("temp.mid")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    midi = MidiFile(temp_file_path)
else:
    midi = MidiFile(None)

# Load tokenizer
config = TokenizerConfig(
    use_programs=True, one_token_stream_for_programs=True, use_time_signatures=True
)
tokenizer = REMI(config)

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

if st.button("Generate"):
    with st.spinner("Generating new MIDI file..."):
        generated_idx = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
        )
        generated_tokens = generated_idx.squeeze().cpu().numpy()
        decoded_tokens = tokenizer.decode(generated_tokens)

        output_path = Path("generated.mid")
        decoded_tokens.dump_midi(output_path)

    st.success("MIDI file generated!")
    st.download_button(
        label="Download MIDI file",
        data=open(output_path, "rb").read(),
        file_name="generated.mid",
        mime="audio/midi",
    )
