import torch
import gc
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from transformer.data_loader import tokenize, tokenize_to_id, detokenize_to_text


class Trainer:
    def __init__(
        self, model, optimizer, train_loader, valid_loader, device, model_dir, plot_dir
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.model_dir = model_dir
        self.plot_dir = plot_dir
        self.history = defaultdict(list)
        self.best_epoch_loss = np.inf
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0
        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for step, data in bar:
            x = data["x"].to(self.device)
            y = data["y"].to(self.device)
            batch_size = x.size(0)

            logits, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            bar.set_postfix(
                Epoch=epoch,
                Train_Loss=epoch_loss,
                LR=self.optimizer.param_groups[0]["lr"],
            )
        return epoch_loss

    def valid_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0
        bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for step, data in bar:
                x = data["x"].to(self.device)
                y = data["y"].to(self.device)
                batch_size = x.size(0)

                logits, loss = self.model(x, y)
                running_loss += loss.item()
                dataset_size += batch_size
                epoch_loss = running_loss / dataset_size
                bar.set_postfix(
                    Epoch=epoch,
                    Valid_Loss=epoch_loss,
                    LR=self.optimizer.param_groups[0]["lr"],
                )
        return epoch_loss

    def train(self, epochs):
        best_model_wts = torch.save(
            self.model.state_dict(), os.path.join(self.model_dir, "init_model.pt")
        )
        start = time.time()

        for epoch in range(1, epochs + 1):
            gc.collect()
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.valid_one_epoch(epoch)

            self.history["TrainLoss"].append(train_loss)
            self.history["ValLoss"].append(val_loss)
            print(f"EPOCH: {epoch}, Train Loss: {train_loss}, Valid Loss: {val_loss}")

            if val_loss < self.best_epoch_loss:
                diff_loss = val_loss - self.best_epoch_loss
                print(
                    f"Validation Loss Improved from {self.best_epoch_loss:.4f} to {val_loss:.4f} "
                    f"(Difference: {diff_loss:.4f})"
                )
                self.best_epoch_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_dir, f"best_model_{epoch}.pt"),
                )
                print("Model Saved")

        end = time.time()
        time_elapsed = end - start
        print(
            "Training completed in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600,
                (time_elapsed % 3600) // 60,
                (time_elapsed % 3600) % 60,
            )
        )
        print("Best Loss: {:.4f}".format(self.best_epoch_loss))
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, f"last_model_{epoch}.pt"),
        )

        self.plot_loss()

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.history["TrainLoss"], label="Training Loss", color="blue", marker="o"
        )
        plt.plot(
            self.history["ValLoss"], label="Validation Loss", color="orange", marker="o"
        )
        plt.title("Training and Validation Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.plot_dir, "loss_plot.png"))
        plt.close()

    def evaluate(
        self, sample_texts, token_to_id_mapping, id_to_token_mapping, new_tokens
    ):
        self.model.eval()
        results = []
        with torch.no_grad():
            for sample in sample_texts:
                tokens = tokenize(sample)
                x = tokenize_to_id(tokens, token_to_id_mapping)
                x = torch.tensor(x, dtype=torch.long).reshape(1, -1).to(self.device)
                gen_seq = self.model.generate(idx=x, max_new_tokens=new_tokens)
                output = detokenize_to_text(
                    list(gen_seq.cpu().detach().numpy()[0]), id_to_token_mapping
                )
                results.append(output)
        return results
