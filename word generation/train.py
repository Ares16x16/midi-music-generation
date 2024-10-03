import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from transformer.model import Transformer
from transformer.data_loader import NewsDataset

# Define the Trainer class
class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.epochs = 1  # Set the number of epochs

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                logits, loss = self.model(inputs, targets=labels)
                if loss is not None:
                    total_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}')
            self.validate()

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits, loss = self.model(inputs, targets=labels)
                if loss is not None:
                    total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f'Validation Loss: {avg_loss:.4f}')

# Main training function
def main():
    # Parameters
    csv_file = r'C:\Users\ed700\workspace\midi-music-generation\word generation\english_financial_news_v2.csv'
    max_length = 50  # Maximum length of tokens
    batch_size = 800
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load and preprocess the data
    # Load the entire dataset into a DataFrame
    full_dataset = pd.read_csv(csv_file)

    # Split the dataset into training and validation sets
    train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)

    train_dataset = NewsDataset(train_data, max_length)
    val_dataset = NewsDataset(val_data, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model
    model = Transformer(vocab_size=len(train_dataset.tokenizer) + 1, 
                              embedding_dim=32, 
                              seq_length=max_length, 
                              num_heads=4, 
                              num_layers=4)

    # Start training
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train()

if __name__ == "__main__":
    main()