"""
Appendice A - Introduzione a PyTorch
Sezione: A.4 Automatic differentiation made easy
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import grad

# A.4 Automatic differentiation made easy
# Automatic differentiation: PyTorch builds the graph and propagates gradients backwards (backpropagation).


y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True) 

z = x1 * w1 + b 
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print("Activation function output: ", a)
print("Obtained loss with binary_cross_entropy: ", loss)
print("Loss gradient respect w1: ",grad_L_w1)
print("Loss gradient respect b: ",grad_L_b)

loss.backward()     # start the backpropagation

print("current values of the gradients used by the optimizer to update the model weights:")
print("w1: ",w1.grad)
print("b: ", b.grad)

"""
NOTE:
- autograd -> This function allows for the explicit calculation of the gradient of an output tensor (such as the loss) with respect to one or more input tensors (w1, b), 
              directly returning the gradient value without needing to invoke the .backward() method on the loss tensor.

- requires_grad=True -> instructs PyTorch to track all operations performed on this tensor to enable automatic gradient calculation (autograd), 
                        a fundamental step for the optimization and training of neural networks. 
                        Mantein the computational graph in memory after the calculation, preventing it from being destroyed and thus allowing it to be reused to calculate 
                        other gradients (such as the one with respect to b) within the same training cycle.

- .grad -> Current value of the gradient accumulated in the tensor w1. It corresponds to the partial derivative of the loss with respect to the weight w1 
           (calculated via .backward() or grad()) and is the value used by the optimizer to update the model weights.

"""
