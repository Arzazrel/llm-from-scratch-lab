import torch
import torch.nn as nn

# -------------------------------- OLD --------------------------------
""" 
Implementing a compact causal self-attention class.
- dropout: percentage of dropout [0,1]
"""
class CausalAttention(nn.Module):
    # initalizing the matrices
    def __init__(self, d_in, d_out, context_length,dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out                                      # dim of the vector embedding
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # query matrix
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)    # key matrix
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)    # value matrix
        self.dropout = nn.Dropout(dropout)                      # Drop-out layer against overfitting
        # add a register buffer, is not mandatory but useful for training
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) 

    # compute context vector
    def forward(self, x):
        b, num_tokens, d_in = x.shape   # get shape of batch, batch dimension is b
        keys = self.W_key(x)            # compute keys
        queries = self.W_query(x)       # compute queries
        values = self.W_value(x)        # compute values

        attn_scores = queries @ keys.transpose(1, 2)    # keep batch dimension at firs position (0) and transpose dim. 1 and 2
        # compute attention score: using bool mask to remove the values above the diagonal (contribution of future token)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        # instead of 0 put -inf in this way softmax put this position to 0 and normalize considering only the remaining values (faster, normalize in one step using only the remaning token)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # normalize with softmax
        attn_weights = self.dropout(attn_weights)                               # apply dropout (only during traning, not during inference)
        context_vec = attn_weights @ values                                     # compute context vector = weighted sum of values
        
        return context_vec

inputs = torch.tensor(
  [[0.13, 0.24, 0.75],      # Hello     (x^1)
   [0.14, 0.99, 0.66],      # my        (x^2)
   [0.72, 0.47, 0.67],      # name      (x^3)
   [0.37, 0.14, 0.84],      # is        (x^4)
   [0.10, 0.27, 0.18],      # John      (x^5)
   [0.03, 0.58, 0.67]]      # Who       (x^6)
)

d_in = 3        # token embedding dimension
d_out = 3       # final token embedding dimension
batch = torch.stack((inputs, inputs), dim=0)

# -------------------------------- NEW --------------------------------

# Extending single-head attention to multi-head attention

# V.0 -> Stacking multiple single-head attention layers. Easier but less efficient.
class MultiHeadAttentionWrapper(nn.Module):
    # initialise the matrices
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    # compute context vector
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


num_heads = 3
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads)
context_vecs = mha(batch)

print("\nBatch shape (2 inputs with 6 tokens each, and each token has embedding dimension 3): ",batch.shape)
print("Use non optimized MultiHeadAttention with ", num_heads, " heads.\nThe context vector are not aggregated, are in line in the result.")
print("Context vector values: ", context_vecs)
print("Context vector shape: ", context_vecs.shape)

# Implementing multi-head attention with weight splits
# we create single W_query, W_key, and W_value weight matrices and then split those into individual matrices for each attention head:
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

print("\nBatch shape (2 inputs with 6 tokens each, and each token has embedding dimension 3) [optimized method]: ",batch.shape)
print("Use optimized MultiHeadAttention with ", num_heads, " heads.\nThe context vector of each head are combined in one embedding vector.")
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)   # create the multi head attention object
context_vecs = mha(batch)       # compute context vector of the batch

print("Context vector values: ", context_vecs)
print("Context vector shape: ", context_vecs.shape)

"""
NOTE:
    # there is a compact and efficient implementation of the above -> [`torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) class in PyTorch
"""