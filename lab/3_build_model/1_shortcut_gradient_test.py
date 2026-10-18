# shortcut example to see the vanishing gradient problem

import torch
import torch.nn as nn

# ------------------------ OLD ------------------------
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    # use this implementation of GELU
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

# ------------------------ NEWs ------------------------

# example of Deep NN ith 5 layer
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output        # input plus output
            else:
                x = layer_output            # only output
        return x

# method to print the gradient for each layer of the NN
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])   # for semplicity target set to 0

    # Calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    loss.backward()                 # compute gradient

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")  # print the mean absolute gradient of the weights


# the example with print
layer_sizes = [3, 3, 3, 3, 3, 1]  
sample_input = torch.tensor([[1., 0., -1.]])

print("With deep NN use shortcut connection to prevent the vanishing gradient problem. The example net used has 5 layers with this size: ", layer_sizes)
print("Show  the mean gradient value (mean of scalar component) for each layers.")
print("\nGradient values WITHOUT shortcut connections:")

torch.manual_seed(123)      # fixed sed, deterministic behaviour
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)  # initialize model
print_gradients(model_without_shortcut, sample_input)                               # print gradient

print("\nGradient values WITH shortcut connections:")

torch.manual_seed(123)      # fixed sed, deterministic behaviour
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)      # initialize model
print_gradients(model_with_shortcut, sample_input)                                  # print gradient