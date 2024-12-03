from tqdm import tqdm
import gc
import torch
import os
import logging
from config import MODEL_DIR


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader, epoch):
        self.model.train()
        dataset_size = 0
        running_loss = 0.0
        step_losses = []  # Record loss at each step
        bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for step, data in bar:
            x = data["x"].to(self.device)
            y = data["y"].to(self.device)
            batch_size = x.size(0)
            logits, loss = self.model.forward(x, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()
            step_losses.append(loss.item())  # Record step loss
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            bar.set_postfix(
                Epoch=epoch,
                Train_Loss=epoch_loss,
                LR=self.optimizer.param_groups[0]["lr"],
            )

            # Save intermediate checkpoint
            if step % (len(dataloader) // 10) == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "running_loss": running_loss,
                        "dataset_size": dataset_size,
                    },
                    os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch}_step_{step}.pt"),
                )
                logging.info(
                    f"Intermediate checkpoint for epoch {epoch}, step {step} saved"
                )

        gc.collect()
        torch.cuda.empty_cache()  # Clear unused memory
        return epoch_loss, step_losses  # Return step losses

    def valid_one_epoch(self, dataloader, epoch):
        self.model.eval()
        dataset_size = 0
        running_loss = 0.0
        step_losses = []  # Record loss at each step
        bar = tqdm(enumerate(dataloader), total=len(dataloader))

        with torch.no_grad():
            for step, data in bar:
                x = data["x"].to(self.device)
                y = data["y"].to(self.device)
                batch_size = x.size(0)
                logits, loss = self.model.forward(x, y)
                running_loss += loss.item()
                step_losses.append(loss.item())  # Record step loss
                dataset_size += batch_size
                epoch_loss = running_loss / dataset_size
                bar.set_postfix(
                    Epoch=epoch,
                    Valid_Loss=epoch_loss,
                    LR=self.optimizer.param_groups[0]["lr"],
                )

        gc.collect()
        torch.cuda.empty_cache()  # Clear unused memory
        return epoch_loss, step_losses  # Return step losses
