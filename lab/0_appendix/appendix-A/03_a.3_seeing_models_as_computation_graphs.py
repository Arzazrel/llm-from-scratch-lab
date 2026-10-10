"""
Appendice A - Introduzione a PyTorch
Sezione: A.3 Seeing models as computation graphs
"""
import torch
import numpy as np
import torch.nn.functional as F

# A.3 Seeing models as computation graphs
# A simplified version of a model in which input data are multiplied by a weight to form an intermediate result; 
# this is then added to a bias to obtain the final result, which is fed into the activation function before the loss is calculated.

print("Simple neuron with 0 bias and sigmoid activation function (ex. of binary classification):")
print("true label: y = 1.0\ninput: x1 = 1.1\nweight: w1 = 2.2\nbias: b = 0.0")

y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2]) # weight parameter
b = torch.tensor([0.0])  # bias unit

z = x1 * w1 + b          # net input
print("\nz: ", z)

a = torch.sigmoid(z)     # activation & output
print("Activation function output: ", a)

loss = F.binary_cross_entropy(a, y)
print("Obtained loss with binary_cross_entropy: ", loss)
