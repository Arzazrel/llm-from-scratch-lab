"""
Appendice A - Introduzione a PyTorch
Sezione: A.5 Implementing multilayer neural networks
"""

import torch
import numpy as np

# class to implement a multilayer NN
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
                
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

# example using the class
model = NeuralNetwork(50, 3)            # create the model with the input and output number of neuron

print("Model architecture:\n",model)    # print the model architecture

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

print("Current weight metrix for the first layer\n",model.layers[0].weight)           # Print the current values ​​​​of the weight matrix associated with the first layer.

torch.manual_seed(123)              # fix seed for the number generator

print("Seed fixed, now rebuild the model:")
model = NeuralNetwork(50, 3)        
print("Current weight metrix for the first layer\n",model.layers[0].weight)
print("Shape:",model.layers[0].weight.shape)

torch.manual_seed(123)

# Create an input tensor X with shape (1, 50) filled with uniform random numbers between 0 and 1.
X = torch.rand((1, 50))
out = model(X)
print("Output logit: ",out)

print("\nNow torch.no_grad()")
"""
It performs the forward pass within a `torch.no_grad()` context block. This temporarily disables gradient tracking (autograd), drastically reducing memory consumption and
speeding up execution. This is the standard approach for validation, testing, or inference phases where updating weights is not required.
"""
with torch.no_grad():
    out = model(X)
print("Output logit: ",out)

"""
It applies the Softmax activation function along the class dimension (dim=1) to the raw output returned by the model. It converts raw scores (logits) into a 
normalized probability distribution (values ​​between 0 and 1 that sum to 1 for each sample), which is useful, for instance, in multi-class classification problems. 
This operation is performed without calculating gradients, thanks to torch.no_grad().
"""
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print("Output softmax: ",out)
