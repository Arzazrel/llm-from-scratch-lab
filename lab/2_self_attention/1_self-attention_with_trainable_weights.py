import torch
import torch.nn as nn

# Implementing self-attention with trainable weights. Using query, key, value.
print("Example of a simple self-attention mechanism without trainable weights with this text: 'Hello my name is John Who'")
print("\nCreate a random embedding vector for this sentence. Embedding with 3 dimension.")
inputs = torch.tensor(
  [[0.13, 0.24, 0.75],      # Hello     (x^1)
   [0.14, 0.99, 0.66],      # my        (x^2)
   [0.72, 0.47, 0.67],      # name      (x^3)
   [0.37, 0.14, 0.84],      # is        (x^4)
   [0.10, 0.27, 0.18],      # John      (x^5)
   [0.03, 0.58, 0.67]]      # Who       (x^6)
)
print(inputs)

d_in = 3        # token embedding dimension
d_out = 3       # final token embedding dimension

# Implementing a compact SelfAttention class using 'nn.Parameter' 
# - is a special type of tensor (torch.Tensor) designed specifically to represent the trainable parameters of a neural network.
class SelfAttention_v1(nn.Module):
    # initalizing the matrices
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))    # query matrix
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))    # key matrix
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))    # value matrix

    # compute context vector
    def forward(self, x):
        keys = x @ self.W_key           # compute keys
        queries = x @ self.W_query      # compute queries
        values = x @ self.W_value       # compute values
        
        attn_scores = queries @ keys.T      # compute attention scores (not normalize) by query and key
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # normalize with softmax
        context_vec = attn_weights @ values # compute context vector = weighted sum of values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("\nContext vector using nn.Parameter: ", sa_v1(inputs))



# - We can streamline the implementation above using PyTorch's Linear layers, which are equivalent to a matrix multiplication if we disable the bias units
class SelfAttention_v2(nn.Module):
    # initalizing the matrices
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # query matrix
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)    # key matrix
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)    # value matrix

    # compute context vector
    def forward(self, x):
        keys = self.W_key(x)        # compute keys
        queries = self.W_query(x)   # compute queries
        values = self.W_value(x)    # compute values
        
        attn_scores = queries @ keys.T          # compute attention scores (not normalize) by query and key
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # normalize with softmax
        context_vec = attn_weights @ values     # compute context vector = weighted sum of values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print("\nContext vector using nn.Linear: ", sa_v2(inputs))
print("\nNote that `SelfAttention_v1` and `SelfAttention_v2` give different outputs because they use different initial weights for the weight matrices.")
