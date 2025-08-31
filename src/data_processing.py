from collections import Counter
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer


### Getting Data (WikiText2) ###
def get_data():
    wikitext2_raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    docs_splits = dict()
    total_dataset_size = 0
    for phase in ["train", "validation", "test"]:
        # Get the documents for the given dataset phase
        docs_in_phase = list(wikitext2_raw[phase]["text"])
        docs_splits[phase] = docs_in_phase

        phase_size = len(docs_in_phase)
        print(f"Size of {phase}: {phase_size}")
        total_dataset_size += phase_size
    
    print(f"Total dataset size: {total_dataset_size}")

    return docs_splits


def split_data(docs_splits, full_dataset_size, train_split, val_split):
    train_count, val_count = int(train_split * full_dataset_size), int(val_split * full_dataset_size)
    test_count = full_dataset_size - train_count - val_count

    # Select the first few data according to the adjusted counts
    corrected_docs_splits = dict()
    corrected_docs_splits["train"] = docs_splits["train"][:train_count]
    corrected_docs_splits["validation"] = docs_splits["validation"][:val_count]
    corrected_docs_splits["test"] = docs_splits["test"][:test_count]

    for phase in ["train", "validation", "test"]:
        print(f"New {phase} size: {len(corrected_docs_splits[phase])}")

    return corrected_docs_splits


### Data Processing ###
def process_data(corrected_docs_splits):
    # Initialize the tokeniser
    tokeniser = get_tokenizer("basic_english")

    tokenised_docs = dict()
    for phase, docs in corrected_docs_splits.items():
        # For each document in the phase, convert to lowercase, and then tokenise the document
        tokens_across_docs = [tokeniser(doc.lower()) for doc in docs]
        tokenised_docs[phase] = tokens_across_docs

    return tokenised_docs


### Numericalisation of Tokens ###
# Make vocabulary of tokens from the train set, where each unique token is mapped to a unique id
def make_train_vocab(train_docs):
    counter = Counter(token for doc in train_docs for token in doc)
    train_vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(), start=4)}
    train_vocab.update({"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3})
    print(f"Training vocabulary size: {len(train_vocab)}")
    
    return train_vocab


def convert_tokens_to_ids(tokenised_docs, train_vocab):
    converted_tokenised_docs = dict()
    for phase in ["train", "validation", "test"]:
        # For this dataset phase, get a list of lists of tokens (one small list = one document)
        tokens_across_docs = tokenised_docs[phase]
        
        # For each document, convert each token to its corresponding id
        converted_tokenised_docs_in_phase = list()
        for tokens_in_doc in tokens_across_docs:
            converted_tokenised_docs_in_phase.append([train_vocab.get("<bos>")] + [train_vocab.get(token, train_vocab["<unk>"]) for token in tokens_in_doc] + [train_vocab.get("<eos>")])
        
        converted_tokenised_docs[phase] = converted_tokenised_docs_in_phase
    
    return converted_tokenised_docs