import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Comparing Efficient Multi-Head Attention Implementations

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Running on {device}")

batch_size = 8
context_len = 1024
embed_dim = 768
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)
print("Use more reistic data input. batch_size: ", batch_size, " context_len: ", context_len, " embedded dimension: ", embed_dim)

# --------------------------- implemented previously ---------------------------
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
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)    # value matrix
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
# --------------------------- new ---------------------------

mha_wrapper = MultiHeadAttentionWrapper(
    d_in=embed_dim,
    d_out=embed_dim//12,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

out = mha_wrapper(embeddings)
print("Shape of the output by MultiHeadAttentionWrapper: ",out.shape)

mha = MultiHeadAttention(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

out = mha(embeddings)
print("Shape of the output by MultiHeadAttention: ",out.shape)

# Using PyTorch's torch.nn.MultiheadAttention with `scaled_dot_product_attention`
class MHAPyTorchClass(nn.Module):
    # initalizing the matrices
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, need_weights=True):
        super().__init__()

        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        self.need_weights = need_weights
        self.proj = nn.Linear(d_out, d_out)     # Linear layer to combine head outputs (from more head to 1 embedding vector)
        # add a register buffer, is not mandatory but useful for training
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    # compute context vector
    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask, need_weights=self.need_weights)
        # optional projection commonly used in many LLM
        output = self.proj(attn_output)         # combine heads

        return output


mha_pytorch_class_default = MHAPyTorchClass(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

out = mha_pytorch_class_default(embeddings)
print("Shape of the output by Pytorch's MultiHeadAttention: ",out.shape)

# - Set `need_weights` (default `True`) to need_weights=False so that MultiheadAttention uses `scaled_dot_product_attention`
#   need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
#                 Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention`` and achieve the best performance for MHA.
mha_pytorch_class_noweights = MHAPyTorchClass(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False,
    need_weights=False 
).to(device)

out = mha_pytorch_class_noweights(embeddings)
print("Shape of the output by Pytorch's MultiHeadAttention: ",out.shape)

# Speed comparison function
def time_pytorch_function(func, *input, num_repeats = 1_000):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(3):
        func(*input)
    torch.cuda.synchronize()

    start.record()
    for _ in range(num_repeats):
        func(*input)
        torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_repeats

# try the different version
embeddings_cuda = embeddings.to(torch.device("cuda"))
# define the function to test
functions = {
    "1) MHA wrapper class": mha_wrapper,
    "2) MHA optimized class": mha,
    "3) PyTorch MHA class defaults": mha_pytorch_class_default,
    "4) PyTorch MHA with need_weights=False": mha_pytorch_class_noweights
}
print("\nExecuting test for:\n1) MHA wrapper class\n2) MHA optimized class\n3) PyTorch MHA class defaults\n4) PyTorch MHA with need_weights=False")
execution_times = [time_pytorch_function(fn, embeddings_cuda) for name,fn in functions.items()]

# Plotting
# Customize further for dark mode aesthetics
plt.rcParams['figure.facecolor'] = '#121212'  # Dark figure background
plt.rcParams['axes.facecolor'] = '#121212'    # Dark axes background
plt.rcParams['axes.edgecolor'] = 'white'      # White axes border
plt.rcParams['axes.labelcolor'] = 'white'     # White labels
plt.rcParams['text.color'] = 'white'          # White text
plt.rcParams['xtick.color'] = 'white'         # White x ticks
plt.rcParams['ytick.color'] = 'white'         # White y ticks
plt.rcParams['grid.color'] = '#444444'        # Lighter grid lines for contrast
plt.rcParams['lines.linewidth'] = 2           # Thicker plot lines for visibility
plt.rcParams['lines.markersize'] = 8          # Larger markers for visibility

fig, ax = plt.subplots()
bars = plt.bar(functions.keys(), execution_times)
plt.ylabel('Execution time (ms)')
plt.xticks(rotation=45, ha="right")

# Calculate new ylim with a margin
max_execution_time = max(execution_times)
upper_ylim = max_execution_time + 0.2 * max_execution_time  # Adding a 20% margin
plt.ylim(0, upper_ylim)                                     # Setting new ylim

# Annotate bars with execution times
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * upper_ylim), round(yval, 2), ha='center', va='bottom')


plt.tight_layout()
plt.savefig("1.pdf")
plt.show()
