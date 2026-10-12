"""
Capitolo 2 - Working with text data
Sezione: 2.8 Encoding word positions
"""

import importlib
import tiktoken
import torch
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ---------------------------- OLD ----------------------------
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# - We use a sliding window approach, changing the position by +1:
# - Create dataset and dataloader that extract chunks from the input text dataset
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length. Stride is the step between the windows
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# function to obtain the data from dataset in window
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")                   # Initialize the tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # Create dataset

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# ---------------------------- NEW ----------------------------

# 2.8 Encoding word positions
# - Embedding layer convert IDs into identical vector representations regardless of where they are located in the input sequence:
# - Positional embeddings are combined with the token embedding vector to form the input embeddings for a large language model:
# - The BytePair encoder has a vocabulary size of 50,257:
# - Suppose we want to encode the input tokens into a 256-dimensional vector representation:
vocab_size = 50257
output_dim = 256
print("Use embedding with 256 dimension and use BPE encoder. Vocab size: ", vocab_size)

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# - If we sample data from the dataloader, we embed the tokens in each batch into a 256-dimensional vector
# - If we have a batch size of 8 with 4 tokens each, this results in a 8 x 4 x 256 tensor:

max_length = 4                      # window size
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("Embedding layer shape: ", token_embeddings.shape)

# - GPT-2 uses absolute position embeddings, so we just create another embedding layer:
print("Create new embedding layer for absolute position embeddings.")
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)    # create NN layer

pos_embeddings = pos_embedding_layer(torch.arange(max_length))          # create value for position embedding
print("Shape for absolute position embeddings: ", pos_embeddings.shape)

# - To create the input embeddings used in an LLM, we simply add the token and the positional embeddings:

input_embeddings = token_embeddings + pos_embeddings
print("Shape of the input embedding = token_embeddings + pos_embeddings.\n",input_embeddings.shape)

# - In the initial phase of the input processing workflow, the input text is segmented into separate tokens
# - Following this segmentation, these tokens are transformed into token IDs based on a predefined vocabulary:
