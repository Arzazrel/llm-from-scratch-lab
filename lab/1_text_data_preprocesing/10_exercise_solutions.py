"""
Capitolo 2 - Soluzioni degli esercizi
"""
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Chapter 2 Exercise solutions

# Exercise 2.1 - BPE with unknown word
print("Exercise 2.1 - BPE with unknown word")
tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode("Akwirw ier")

print("Try to encode an unknown token: 'Akwirw ier'. Related token ID: ", integers)
print("The different token: ")
for i in integers:
    print(f"{i} -> {tokenizer.decode([i])}")

tokenizer.encode("Ak")
tokenizer.encode("w")
tokenizer.encode("ir")
tokenizer.encode("w")
tokenizer.encode(" ")
tokenizer.encode("ier")

tokenizer.decode([33901, 86, 343, 86, 220, 959])

# Exercise 2.2 - batch creation
print("\nExercise 2.2 - batch creation")

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


def create_dataloader(txt, batch_size=4, max_length=256, stride=128):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)

vocab_size = 50257
output_dim = 256
max_len = 4
context_length = max_len

token_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("Creation of the batches. window_size = 2 , stride = 2 ")
dataloader = create_dataloader(raw_text, batch_size=4, max_length=2, stride=2)

for batch in dataloader:
    x, y = batch
    break

print("Input: ", x)
print("Label: ", y)

print("Creation of the batches. window_size = 8 , stride = 2 ")
dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=2)

for batch in dataloader:
    x, y = batch
    break

print("Input: ", x)
print("Label: ", y)
