"""
Capitolo 2 - Working with text data
Sezione: 2.7 Creating token embeddings
"""

# --- Import necessari per eseguire questo file singolarmente ---
import torch
import torch.nn as nn

# 2.7 Creating token embeddings
# - The data is already almost ready for an LLM
# - But lastly let us embed the tokens in a continuous vector representation using an embedding layer
# - Usually, these embedding layers are part of the LLM itself and are updated (trained) during model training
# - Suppose we have the following four input examples with input ids 2, 3, 5, and 1 (after tokenization):

input_ids = torch.tensor([2, 3, 5, 1])

# - For the sake of simplicity, suppose we have a small vocabulary of only 6 words and we want to create embeddings of size 3:

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)    # create a layer vocab_size x output_emb

# - This would result in a 6x3 weight matrix:

print("Embedding matrix [ vocab_size, output_emb]: ", embedding_layer.weight)

# - For those who are familiar with one-hot encoding, the embedding layer approach above is essentially just a more efficient way of implementing one-hot encoding followed by matrix multiplication in a fully-connected layer, which is described in the supplementary code in [./embedding_vs_matmul](../03_bonus_embedding-vs-matmul)
# - Because the embedding layer is just a more efficient implementation that is equivalent to the one-hot encoding and matrix-multiplication approach it can be seen as a neural network layer that can be optimized via backpropagation
# - To convert a token with id 3 into a 3-dimensional vector, we do the following:
#print(embedding_layer(torch.tensor([3])))

print("Embedding for the token in input:\n", embedding_layer(input_ids))
