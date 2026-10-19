# Coding an LLM architecture (merge the previous implementation to create the base block for the LLM architecture, the trasformer one)

import tiktoken
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# function to read the example text
def read_text_file():
    with open("text_example.txt", "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data
  
# function o read and visualise information about the text
def read_show_info_text_file(tokenizer):
    text_data = read_text_file()  # read from text file

    # compute informations
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Read 'text_example.txt'.")
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    
    return text_data

# function to convert from text to token
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

# function to convert from token to text
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

# ---- input data load ----
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


# ---- muli head self attention ----
class MultiHeadAttention(nn.Module):
    # initalizing the matrices
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out                      # dim of the vector embedding
        self.num_heads = num_heads              # num attention heads
        self.head_dim = d_out // num_heads      # Reduce the projection dim to match desired output dim
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # query matrix
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)    # key matrix
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)    # value matrixù
        self.out_proj = nn.Linear(d_out, d_out)                 # Linear layer to combine head outputs (from more head to 1 embedding vector)
        self.dropout = nn.Dropout(dropout)                      # Drop-out layer against overfitting
        # add a register buffer, is not mandatory but useful for training
        self.register_buffer("mask",torch.triu(torch.ones(context_length, context_length),diagonal=1))

    # compute context vector
    def forward(self, x):
        b, num_tokens, d_in = x.shape   # get shape of batch, batch dimension is b
        keys = self.W_key(x)            # compute keys
        queries = self.W_query(x)       # compute queries
        values = self.W_value(x)        # compute values

        # implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)          # adding heads dimension
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)      # adding heads dimension
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)    # adding heads dimension

        # Transpose dim 1 and 2 -> (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute causal self-attention (queries and keys)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Use the mask to fill attention scores with -inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)     # normalization
        attn_weights = self.dropout(attn_weights)                                   # apply dropout

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # optional projection commonly used in many LLM
        context_vec = self.out_proj(context_vec)        # combine

        return context_vec

# standard layer normalization. Mean = 1 and varaince of 1
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

#NN for the transformer block (expande the embedding vector and then return to original size)
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# create transformer block (the unit block of the LLM). Combines:
# - causal multi-head attention
# - linear layers
# - feed forward neural network
# - dropout and shortcut connections
class TransformerBlock(nn.Module):
    # initialize by dictionary
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # - self-attention block
        shortcut = x                # shortcut connection for attention
        x = self.norm1(x)           # layer normalization
        x = self.att(x)             # shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)   # add dropout
        x = x + shortcut            # shortcut connection, add the original input to the output

        # - NN block
        shortcut = x                # shortcut connection for NN
        x = self.norm2(x)           # layer normalization
        x = self.ff(x)
        x = self.drop_shortcut(x)   # add dropout
        x = x + shortcut            # shortcut connection, add the original input to the output

        return x

# Coding the GPT model: transformer block is repeated multiple times.
class GPTModel(nn.Module):
    # initialize
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# fuction to print the number of parameters of the LLM architecture (gpt model)
def print_gpt_number_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    
    print("\nIn the original GPT-2 paper, the researchers reused the token embedding matrix as an output matrix (weight tying).")
    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    # compute the memory requirements of the model as follows
    total_size_bytes = total_params * 4                 # assuming float32
    total_size_mb = total_size_bytes / (1024 * 1024)    # into megabytes
    print(f"Total size of the model: {total_size_mb:.2f} MB")

# function implements greedy decoding, which is a simple and fast method to generate text in which at each step, the model chooses the word (or token) with the highest 
# probability as its next output. (the highest logit corresponds to the highest probability, so we technically wouldn't even have to compute the softmax function explicitly)
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop current context if it exceeds the supported context size and use only the last context token
        idx_cond = idx[:, -context_size:]
        # evaluate mode, don't update weight
        with torch.no_grad():
            logits = model(idx_cond)    # get the predictions

        # take only last predict token (last word). (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # now size of (batch, 1)

        # append the index to the running sequence. (batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=1)  # 

    return idx          #return the indexes, in the vocabulary, of the predicted words

"""
NOTE:
- OTHER GPT IMPLEMENTATIONS:
    GPT2-smallb(the 124M configuration we already implemented):
    - "emb_dim" = 768
    - "n_layers" = 12
    - "n_heads" = 12

    **GPT2-medium:**
    - "emb_dim" = 1024
    - "n_layers" = 24
    - "n_heads" = 16

    **GPT2-large:**
    - "emb_dim" = 1280
    - "n_layers" = 36
    - "n_heads" = 20

    **GPT2-XL:**
    - "emb_dim" = 1600
    - "n_layers" = 48
    - "n_heads" = 25
"""