"""
Capitolo 2 - Implementazione compatta del DataLoader
(dal notebook dataloader.ipynb: pipeline tokenizzazione+sampling)
"""

# --- Import necessari per eseguire questo file singolarmente ---
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# The Main Data Loading Pipeline Summarized
# Data loading pipeline without the intermediate steps.

# ---------------------------- OLD ----------------------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
# ---------------------------- NEW ----------------------------

tokenizer = tiktoken.get_encoding("gpt2")       # get tokenizer
encoded_text = tokenizer.encode(raw_text)       # tokenize the raw text

vocab_size = 50257          
output_dim = 256            # embedding dimension
context_length = 1024       # text dimension

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)      
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

max_length = 4              # window size
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length) # create the batches

print("Creation of the batches: ")
for batch in dataloader:
    x, y = batch

    token_embeddings = token_embedding_layer(x)
    # Create a one-dimensional tensor containing a sequence of integers starting from 0 (inclusive) up to max_length (exclusive).
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings = token_embeddings + pos_embeddings            # combine the embedding
    
    print("token_embeddings: ", token_embeddings)
    print("pos_embeddings: ", pos_embedding_layer)
    print("input_embeddings: ", input_embeddings)

    break

print("Size of the input embedding: ", input_embeddings.shape)
