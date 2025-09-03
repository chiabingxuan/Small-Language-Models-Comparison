import copy
import datetime
import logging
import math
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


### Dataset and DataLoader ###
# Create a custom PyTorch dataset for batching
class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data        # List of tokenised and numericalised documents
        self.seq_len = seq_len  # Fixed sequence length for training
        self.input_output_pairs = list()

        # Loop through each document and get the corresponding input-output pairs
        for doc in self.data:
            for i in range(len(doc) - self.seq_len):
                input_seq = doc[i: i + self.seq_len]
                output_seq = doc[i + 1: i + self.seq_len + 1]
                self.input_output_pairs.append((torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_seq, dtype=torch.long)))

    # Defines the total number of samples in the dataset (number of input-output pairs across all documents)
    def __len__(self):
        return len(self.input_output_pairs)

    # Retrieves a single sample (input-output pair) from the dataset
    def __getitem__(self, idx):     # idx: The index of the sample to retrieve
        return self.input_output_pairs[idx]


def make_dataloaders(converted_tokenised_docs, seq_len, batch_size):
    # Create datasets
    train_dataset = TextDataset(converted_tokenised_docs["train"], seq_len)
    val_dataset = TextDataset(converted_tokenised_docs["validation"], seq_len)
    test_dataset = TextDataset(converted_tokenised_docs["test"], seq_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


### RNN Architecture ###
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)                 # Shape: (batch_size, seq_len, embed_size)
        output, hidden = self.rnn(embedded, hidden)  # RNN output and hidden state
        output = self.fc(output)                     # Shape: (batch_size, seq_len, vocab_size)
        return output, hidden


### LSTM Architecture ###
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)  # Character embedding layer
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # Output layer
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)                    # Convert character indices to embeddings
        output, hidden = self.lstm(embedded, hidden)    # Pass through LSTM
        output = self.fc(output)                        # Map LSTM outputs to vocab probabilities
        return output, hidden
    
    
### Model Training ###
def setup_logger(save_name, current_date, logs_folder_path) -> logging.Logger:
    logging.basicConfig(
        filename=os.path.join(logs_folder_path, f"logs_{save_name}_{current_date}.txt"),
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode="w"
    )

    return logging.getLogger()


def train_step(model, dataloader, criterion, optimizer, device, vocab_size, grad_clipping_max_norm):    
    # Set model to training mode
    model.train()

    # Tracks total loss
    total_loss = 0.0

    # Iterate over data
    # inputs and labels: (batch_size, seq_len)
    for inputs, labels in tqdm(dataloader):
        # Send data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(True):
            # Get outputs of model: (batch_size, seq_len, vocab_size)
            outputs, _ = model(inputs)
                
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping_max_norm)

            optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
            
    # Find training loss
    train_loss = total_loss / len(dataloader)

    return train_loss


def val_step(model, dataloader, criterion, device, vocab_size):
    # Set model to evaluation mode
    model.eval()

    # Tracks total loss
    total_loss = 0.0

    # Iterate over data
    # inputs and labels: (batch_size, seq_len)
    for inputs, labels in tqdm(dataloader):
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


def train(model, converted_tokenised_docs, train_vocab, seq_len, batch_size, num_epochs, lr, grad_clipping_max_norm, patience, device, save_name):
    # Get current datetime, to be used in file names
    current_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # Create weights folder if it does not exist
    weights_folder_path = "weights"
    os.makedirs(weights_folder_path, exist_ok=True)

    # Create logs folder if it does not exist, and get logger
    logs_folder_path = "logs"
    os.makedirs(logs_folder_path, exist_ok=True)
    logger = setup_logger(save_name=save_name, current_date=current_date, logs_folder_path=logs_folder_path)

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
        train_loss = train_step(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, device=device, vocab_size=len(train_vocab), grad_clipping_max_norm=grad_clipping_max_norm)
        train_perplexity = math.exp(train_loss)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Train Perplexity: {:.4f}".format(train_perplexity))

        # Validation step
        val_loss = val_step(model=model, dataloader=val_loader, criterion=criterion, device=device, vocab_size=len(train_vocab))
        val_perplexity = math.exp(val_loss)
        print("Val Loss: {:.4f}".format(val_loss))
        print("Val Perplexity: {:.4f}".format(val_perplexity))

        logger.info(f"Epoch {epoch}/{num_epochs - 1} | Train Loss: {train_loss} | Train Perplexity: {train_perplexity} | Val Loss: {val_loss} | Val Perplexity: {val_perplexity}")

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
            logger.info(f"Epoch {epoch}/{num_epochs - 1} | Best val loss has improved | Counter: {counter} | Best val loss: {best_loss}")

            # Save model weights temporarily
            tmp_path = os.path.join(weights_folder_path, f"temp_model_weights_{save_name}.pth")
            torch.save(best_model_wts, tmp_path)

        else:
            # There is no improvement in best validation loss
            # Increment counter
            counter += 1

            # Print details
            print(f"Best val loss did not improve. Counter: {counter} | Best val loss: {best_loss}")
            logger.info(f"Epoch {epoch}/{num_epochs - 1} | Best val loss did not improve | Counter: {counter} | Best val loss: {best_loss}")

            if counter >= patience:
                # Training has gone too long without improvement in best validation loss - stop training early
                print(f"Early stopping triggered. Best val loss did not improve for {patience} consecutive epochs.")
                logger.info(f"Early stopping triggered. Best val loss did not improve for {patience} consecutive epochs.")
                break

        print()

    time_elapsed = int(time.time() - since)

    # Print results of training
    print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    logger.info("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # Save best model weights
    model_weights_file_path = os.path.join(weights_folder_path, f"best_model_weights_{save_name}_{current_date}.pth")
    torch.save(best_model_wts, model_weights_file_path)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_loss_history, val_loss_history


def plot_and_save_training_metrics(train_loss_history, val_loss_history, save_name):
    # Create plots folder if it does not exist
    plots_folder_path = "plots"
    os.makedirs(plots_folder_path, exist_ok=True)

    fig = plt.figure()
    
    # Plot loss curves
    ax1 = fig.add_subplot(121)
    ax1.title.set_text("Loss")
    ax1.plot(train_loss_history, label="train_loss")
    ax1.plot(val_loss_history, label="val_loss")
    ax1.set_xlabel("No. of epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.tight_layout()

    # Save the plot
    plt.savefig(os.path.normpath(os.path.join("plots", f"loss_curves_{save_name}.png")))

    plt.show()