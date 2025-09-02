from collections import Counter
from datasets import load_dataset
import nltk
from nltk.corpus import reuters
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from transformers import AutoTokenizer


nltk.download("reuters")
nltk.download("punkt")
nltk.download("punkt_tab")


### Getting Data ###
def get_and_split_data(full_dataset_size, train_split, val_split, seed):
    # Get reuters data from NLTK
    doc_ids = reuters.fileids()
    print(f"Total number of documents: {len(doc_ids)}")

    # Cap off at dataset size chosen
    doc_ids = doc_ids[:full_dataset_size]

    # Do train-val-test split
    train_and_val_ids, test_ids = train_test_split(doc_ids, train_size=train_split + val_split, random_state=seed)
    train_ids, val_ids = train_test_split(train_and_val_ids, train_size=train_split / (train_split + val_split), random_state=seed)

    # Select the data according to the ids in the train-val-test split
    docs_splits = dict()
    docs_splits["train"] = [reuters.raw(doc_id) for doc_id in train_ids]
    docs_splits["validation"] = [reuters.raw(doc_id) for doc_id in val_ids]
    docs_splits["test"] = [reuters.raw(doc_id) for doc_id in test_ids]

    for phase in ["train", "validation", "test"]:
        print(f"New {phase} size: {len(docs_splits[phase])}")

    return docs_splits


### Data Processing ###
def process_data(docs_splits, using_subword):
    # Initialize the tokeniser
    if using_subword:
        tokeniser = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
        tokenise_fn = lambda text: tokeniser.tokenize(text)
    else:
        tokeniser = get_tokenizer("basic_english")
        tokenise_fn = lambda text: tokeniser(text)

    tokenised_docs = dict()
    for phase, docs in docs_splits.items():
        # For each document in the phase, convert to lowercase, and then tokenise the document
        tokens_across_docs = [tokenise_fn(doc.lower()) for doc in docs]
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


def process_and_format_docs_to_ids(docs_splits, using_subword):
    # Data processing
    tokenised_docs = process_data(docs_splits=docs_splits, using_subword=using_subword)

    # Make vocabulary from train set
    train_vocab = make_train_vocab(train_docs=tokenised_docs["train"])

    # Use vocabulary to numericalise train, val and test datasets
    converted_tokenised_docs = convert_tokens_to_ids(tokenised_docs=tokenised_docs, train_vocab=train_vocab)

    return converted_tokenised_docs, train_vocab