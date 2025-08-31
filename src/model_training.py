from collections import Counter
import copy
from datasets import load_dataset
import datetime
import nltk
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


### Dataset and DataLoader ###
# Create a custom PyTorch dataset for batching
class WikiText2Dataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data        # List of tokenised and numericalised documents
        self.seq_len = seq_len  # Fixed sequence length for training

    # Defines the total number of samples in the dataset (number of input-output pairs across all documents)
    def __len__(self):
        # If length of document is not longer than sequence length, the number of samples is 0
        return sum(max(0, len(doc) - self.seq_len) for doc in self.data)

    # Retrieves a single sample (input-output pair) from the dataset
    def __getitem__(self, idx):     # idx: The index of the sample to retrieve
        # Loop through each document and decrement idx to find the input-output pair corresponding to that index
        for doc in self.data:
            if idx < len(doc) - self.seq_len:
                input_seq = doc[idx: idx + self.seq_len]
                output_seq = doc[idx + 1: idx + self.seq_len + 1]
                return torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_seq, dtype=torch.long)
            idx -= len(doc) - self.seq_len

        # Index provided is out of range of the dataset
        raise IndexError
    

def make_dataloaders(converted_tokenised_docs, seq_len, batch_size):
    # Create datasets
    train_dataset = WikiText2Dataset(converted_tokenised_docs["train"], seq_len)
    val_dataset = WikiText2Dataset(converted_tokenised_docs["validation"], seq_len)
    test_dataset = WikiText2Dataset(converted_tokenised_docs["test"], seq_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


### RNN Architecture ###
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)                 # Shape: (batch_size, seq_len, embed_size)
        output, hidden = self.rnn(embedded, hidden)  # RNN output and hidden state
        output = self.fc(output)                     # Shape: (batch_size, seq_len, vocab_size)
        return output, hidden


### Model Training ###
### TRAIN STEP (SINGLE EPOCH) ###
def train_step(model, dataloader, criterion, optimizer, device, vocab_size):    
    # Set model to training mode
    model.train()

    # Tracks total loss
    total_loss = 0.0

    # Iterate over data
    # inputs and labels: (batch_size, seq_len)
    for inputs, labels in dataloader:
        # Send data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(True):
            # Get outputs of model: (batch_size, seq_len, vocab_size)
            outputs, _ = model(inputs)
                
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            
            # Backward + optimise
            loss.backward()
            optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
            
    # Find training loss
    train_loss = total_loss / len(dataloader)

    return train_loss


### VALIDATION STEP (SINGLE EPOCH) ###
def val_step(model, dataloader, criterion, device, vocab_size):
    # Set model to evaluation mode
    model.eval()

    # Tracks total loss
    total_loss = 0.0

    # Iterate over data
    # inputs and labels: (batch_size, seq_len)
    for inputs, labels in dataloader:
        # Send data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass, but inference only
        with torch.no_grad():
            # Get outputs of model: (batch_size, seq_len, vocab_size)
            outputs, _ = model(inputs)

            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            
        # Update statistics
        total_loss += loss.item()
    
    # Find validation loss
    val_loss = total_loss / len(dataloader)

    return val_loss


def train(model, converted_tokenised_docs, train_vocab, seq_len, batch_size, num_epochs, lr, patience, device, save_name):
    # Create weights folder if it does not exist
    weight_folder_path = os.path.normpath(os.path.join("..", "weights"))
    os.makedirs(weight_folder_path, exist_ok=True)

    # Get the dataloaders for the respective phases of the dataset
    train_loader, val_loader, _ = make_dataloaders(converted_tokenised_docs=converted_tokenised_docs, seq_len=seq_len, batch_size=batch_size)

    # Initialise criterion and optimiser
    criterion = nn.CrossEntropyLoss(ignore_index=train_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lists that will record how loss changes throughout the training process
    train_loss_history = []
    val_loss_history = []
    
    # Details of best performing model, to be updated during training
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    
    # Will track the number of consecutive epochs where the validation loss does not set a new best
    counter = 0
    
    # Initial time
    since = time.time()

    # Training loop
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Train step
        train_loss = train_step(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, device=device, vocab_size=len(train_vocab))
        print("Train Loss: {:.4f}".format(train_loss))

        # Validation step
        val_loss = val_step(model=model, dataloader=val_loader, criterion=criterion, device=device, vocab_size=len(train_vocab))
        print("Val Loss: {:.4f}".format(val_loss))

        # Update history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        # Check if the current validation loss is the best
        if val_loss < best_loss:
            # There is an improvement in best validation loss
            # Update best statistics
            best_loss = val_loss

            # Update best model weights
            best_model_wts = copy.deepcopy(model.state_dict())

            # Reset counter
            counter = 0
            
            # Print details
            print(f"Best val loss has improved. Counter: {counter} | Best val loss: {best_loss}")

            # Save model weights temporarily
            tmp_path = os.path.join(weight_folder_path, f"temp_model_weights_{save_name}.pth")
            torch.save(best_model_wts, tmp_path)

        else:
            # There is no improvement in best validation loss
            # Increment counter
            counter += 1

            # Print details
            print(f"Best val loss did not improve. Counter: {counter} | Best val loss: {best_loss}")

            if counter >= patience:
                # Training has gone too long without improvement in best validation loss - stop training early
                print(f"Early stopping triggered. Best val loss did not improve for {patience} consecutive epochs.")
                break

        print()

    time_elapsed = int(time.time() - since)

    # Print results of training
    print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # Save best model weights
    model_weights_file_path = os.path.join(weight_folder_path, f"best_model_weights_{save_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pth")
    torch.save(best_model_wts, model_weights_file_path)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_loss_history, val_loss_history