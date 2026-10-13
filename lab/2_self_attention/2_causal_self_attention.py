import torch
import torch.nn as nn

# -------------------------------- OLD --------------------------------
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

# -------------------------------- NEW --------------------------------

# Hiding future words with causal attention
# - In causal attention, the attention weights above the diagonal are masked, ensuring that for any given input, the LLM is unable to utilize future tokens while calculating the context vectors with the attention weight
# - If we apply a dropout rate of 0.5 (50%), the non-dropped values will be scaled accordingly by a factor of 1/0.5 = 2.

print("Example of a casual self-attention mechanism with this text: 'Hello my name is John Who'")
print("Create a random embedding vector for this sentence. Embedding with 3 dimension. Using Drop-out to reduce overfitting.")
print(inputs)

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


# the data in the LLM are organised in batch, check if the mechanism support batch and not only one data
batch = torch.stack((inputs, inputs), dim=0)
print("\nBatch shape (2 inputs with 6 tokens each, and each token has embedding dimension 3): ",batch.shape)

torch.manual_seed(123)
context_length = batch.shape[1]                         # number of token in the sentence example
ca = CausalAttention(d_in, d_out, context_length, 0.0)  # create the casual attention object
context_vecs = ca(batch)                                # compute context vector of the batch

print("Context vector values: ", context_vecs)
print("Context vector shape: ", context_vecs.shape)