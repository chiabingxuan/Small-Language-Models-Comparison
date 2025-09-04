from src.model_training import make_dataloaders
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_model(model_initialised, filename, device):
    # Get path to .pth file containing existing weights
    weights_folder_path = "weights"
    weights_file_path = os.path.join(weights_folder_path, f"{filename}.pth")

    # Load .pth file into model
    state_dict = torch.load(weights_file_path, map_location=torch.device(device))

    model_initialised.load_state_dict(state_dict)

    # Send model to device
    model_initialised = model_initialised.to(device)

    return model_initialised


def evaluate_model(model, converted_tokenised_docs, seq_len, batch_size, criterion, device, vocab_size):
    # Get the dataloader for the test dataset
    _, _, test_loader = make_dataloaders(converted_tokenised_docs=converted_tokenised_docs, seq_len=seq_len, batch_size=batch_size)

    # Set model to evaluation mode
    model.eval()

    # Tracks total loss
    total_loss = 0.0

    # Iterate over data
    # inputs and labels: (batch_size, seq_len)
    for inputs, labels in tqdm(test_loader):
        # Send data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass, but inference only
        with torch.no_grad():
            # Get outputs of model: (batch_size, seq_len, vocab_size)
            outputs, _ = model(inputs)

            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            
        # Update statistics
        total_loss += loss.item()
    
    # Find average loss
    loss = total_loss / len(test_loader)

    return loss


def generate_text(model, train_vocab, start_seq, temperature, max_len=300):
    # Set model to evaluation mode
    model.eval()

    # Create a mapping of ids to words in training vocab
    ids_to_words_in_train_vocab = {id: word for word, id in train_vocab.items()}

    # Get sequence of ids in starting sequence
    start_seq_ids = [train_vocab.get(word.lower(), train_vocab["<unk>"]) for word in start_seq]

    # Initialise inputs and hidden state
    x, hidden = torch.tensor([start_seq_ids], dtype=torch.long), None

    # Final generated sequence of words
    generated_text = start_seq.copy()

    for _ in tqdm(range(max_len)):
        # Get outputs and update hidden state of model
        outputs, hidden = model(x, hidden)
        outputs = outputs[:, -1, :]                                             # Take last step's output (last time step's logits, for each sample in the "batch")
        probabilities = F.softmax(outputs / temperature, dim=-1)                # Convert to probabilities with temperature (higher temperature => logits will be flattened more by the division, more random selection from distribution)
        next_token_id = torch.multinomial(probabilities, num_samples=1).item()  # Sample from distribution

        # If next token is end token, terminate generation
        if next_token_id == train_vocab["<eos>"]:
            break

        generated_text.append(ids_to_words_in_train_vocab[next_token_id])

        # Update input to be fed into model
        x = torch.tensor([[next_token_id]], dtype=torch.long)

    return " ".join(generated_text)