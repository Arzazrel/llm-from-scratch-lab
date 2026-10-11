"""
Capitolo 2 - BONUS: Embedding layer vs Linear layer nn.Embedding equivale a matmul con one-hot encoding.
"""

import torch
import torch.nn as nn

# Understanding the Difference Between Embedding Layers and Linear Layers
# - Embedding layers in PyTorch accomplish the same as linear layers that perform matrix multiplications; the reason we use embedding layers is computational efficiency
# - We will take a look at this relationship step by step using code examples in PyTorch
print("PyTorch version:", torch.__version__)

print("Understanding the Difference Between Embedding Layers and Linear Layers")

# Using nn.Embedding
print("\nUsing nn.Embedding")
# Suppose we have the following 3 training examples, which may represent token IDs in a LLM context
idx = torch.tensor([2, 3, 1])

# The number of rows in the embedding matrix can be determined by obtaining the largest token ID + 1.
# If the highest token ID is 3, then we want 4 rows, for the possible token IDs 0, 1, 2, 3
num_idx = max(idx)+1
out_dim = 5             # The desired embedding dimension is a hyperparameter

# - Let's implement a simple embedding layer:
# We use the random seed for reproducibility since weights in the embedding layer are initialized with small random values
torch.manual_seed(123)
embedding = torch.nn.Embedding(num_idx, out_dim)

print("Number of rows in the embedding matrix can be determined by obtaining the largest token ID + 1.")
print("Dimension ", num_idx, " , ", out_dim)
print("Embedding weight: ", embedding.weight)

# - We can then use the embedding layers to obtain the vector representation of a training example with ID 1:
print("Vector representation of a training example with ID 1", embedding(torch.tensor([1])))

# - Now, let's convert all the training examples we have defined previously:

idx = torch.tensor([2, 3, 1])
print("Vector representation of the tensor: ", embedding(idx))

# Using nn.Linear
print("\nUsing nn.Linear")
# - Now, we will demonstrate that the embedding layer above accomplishes exactly the same as `nn.Linear` layer on a one-hot encoded representation in PyTorch
# - First, let's convert the token IDs into a one-hot representation:

onehot = torch.nn.functional.one_hot(idx)
print("One hote encoding: ", onehot)

# - Next, we initialize a `Linear` layer, which caries out a matrix multiplication $X W^\top$:
torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)          # create layer without bias
print("Linear NN weight: ", linear.weight)

# - Note that the linear layer in PyTorch is also initialized with small random weights; to directly compare it to the `Embedding` layer above, we have to use the same small random weights, which is why we reassign them here:
linear.weight = torch.nn.Parameter(embedding.weight.T.detach())

# - Now we can use the linear layer on the one-hot encoded representation of the inputs:
print("Vector representation of the tensor (linear):\n", linear(onehot.float()))

# As we can see, this is exactly the same as what we got when we used the embedding layer:
print("Vector representation of the tensor (embedding):\n", embedding(idx))