import torch
import numpy as np
import pandas as pd
import time
import gc
from tqdm import tqdm
import copy
from collections import defaultdict
import os
import logging
from config import *
from trainer import Trainer
from dataset import MidiDataset
from transformer import Transformer
import matplotlib.pyplot as plt


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_epoch_loss = checkpoint["best_epoch_loss"]
        history = checkpoint["history"]
        logging.info(f"Loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch, best_epoch_loss, history
    else:
        logging.info(f"No checkpoint found at '{filename}'")
        return 0, np.inf, defaultdict(list)


def save_loss_graph(history, output_dir):
    plt.figure()
    plt.plot(history["TrainLoss"], label="Train Loss")
    plt.plot(history["ValLoss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(output_dir, "loss_graph.png"))
    plt.close()

    # Save step losses
    np.save(os.path.join(output_dir, "train_step_losses.npy"), history["TrainStepLoss"])
    np.save(os.path.join(output_dir, "val_step_losses.npy"), history["ValStepLoss"])


def main():
    # Prepping directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR + "/app_{}.log".format(SUFFIX), mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Loading files...")
    df = pd.read_csv(ENCODING_DIR + "/df.csv")
    logging.info("Dataset size: {}".format(df.shape[0]))

    # Data preparation
    split_index = int(len(df) * TRAIN_TEST_RATIO)
    train_df = df.iloc[:split_index, :]
    val_df = df.iloc[split_index:, :]

    train_data = MidiDataset(train_df)
    eval_data = MidiDataset(val_df)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_data, batch_size=BATCH_SIZE, shuffle=False
    )

    # Model preparation
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBED_DIM,
        seq_length=SEQ_LEN,
        num_heads=TRANSFORMER_HEADS,
        num_layers=TRANSFORMER_LAYERS,
    )
    model.to(DEVICE)  # Ensure model is on GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Load checkpoint if exists
    start_epoch, best_epoch_loss, history = load_checkpoint(
        model,
        optimizer,
        os.path.join(MODEL_DIR, "checkpoint.pt"),
    )

    # Initialize step loss history if not present
    if "TrainStepLoss" not in history:
        history["TrainStepLoss"] = []
    if "ValStepLoss" not in history:
        history["ValStepLoss"] = []

    trainer = Trainer(model, optimizer, DEVICE)

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        gc.collect()
        torch.cuda.empty_cache()  # Clear unused memory
        train_loss, train_step_losses = trainer.train_one_epoch(train_dataloader, epoch)
        val_loss, val_step_losses = trainer.valid_one_epoch(eval_dataloader, epoch)

        history["TrainLoss"].append(train_loss)
        history["ValLoss"].append(val_loss)
        history["TrainStepLoss"].extend(train_step_losses)  # Record step losses
        history["ValStepLoss"].extend(val_step_losses)  # Record step losses

        logging.info("Epoch: {} TL: {} VL: {}".format(epoch, train_loss, val_loss))
        if val_loss < best_epoch_loss:
            logging.info(
                f"Validation Loss Improved ({best_epoch_loss} ---> {val_loss})"
            )
            best_epoch_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                model.state_dict(), os.path.join(MODEL_DIR, "model_best_loss.pt")
            )
            logging.info("Model Saved")

        # Save the model for each epoch
        torch.save(
            model.state_dict(), os.path.join(MODEL_DIR, f"model_epoch_{epoch}.pt")
        )
        logging.info(f"Model for epoch {epoch} saved")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_epoch_loss": best_epoch_loss,
                "history": history,
            },
            os.path.join(MODEL_DIR, f"checkpoint.pt"),
        )
        save_loss_graph(history, MODEL_DIR)

        # Save intermediate checkpoints during the epoch
        if epoch % (EPOCHS // 10) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_epoch_loss": best_epoch_loss,
                    "history": history,
                },
                os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch}.pt"),
            )
            logging.info(f"Intermediate checkpoint for epoch {epoch} saved")

    end = time.time()
    time_elapsed = end - start
    logging.info(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    logging.info("Best Loss: {:.4f}".format(best_epoch_loss))
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_model.pt"))


if __name__ == "__main__":
    main()
