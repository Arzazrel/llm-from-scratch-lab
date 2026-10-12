import torch

# 3.3 Attending to different parts of the input with self-attention
# 3.3.1 A simple self-attention mechanism without trainable weights
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

# - Apply to all pairwise elements to compute the unnormalized attention score matrix:
print("\nCompute unnormalized attention score matrix (dot-product for each word in sentence):")

attn_scores = torch.empty(6, 6)         # initialize

# compute the dot product for each couple of the input
attn_scores = inputs @ inputs.T
"""
# simple implementation example of matrix dot product (less efficient multiplication)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
"""
print(attn_scores)

# - normalize, using softmax, each row so that the values in each row sum to 1
attn_weights = torch.softmax(attn_scores, dim=-1)
print("\nNormalizing to obtain attention weight matrix:\n", attn_weights)

print("Check -> all row sums:\n", attn_weights.sum(dim=-1))

# - compute all context vectors:
all_context_vecs = attn_weights @ inputs
print("\nAll context vector:", all_context_vecs)
