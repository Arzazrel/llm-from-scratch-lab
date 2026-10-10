"""
Appendice A - Introduzione a PyTorch
Sezione: A.7 A typical training loop
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# A.7 A typical training loop

# ------------------------------------ OLD ------------------------------------
# define data input for training
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1]) # define label for the data input for training

# define data input for testing
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])           # define label for the data input for testing 

# class to maage dataset
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]        
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train) # create train dataset
test_ds = ToyDataset(X_test, y_test)    # create test dataset

# define the loader for the training set
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True          # remove the last batch if not complete
)

# define the loader for the testing set
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    drop_last=True          # remove the last batch if not complete
)

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
        
# ------------------------------------ NEWs ------------------------------------
        
# function to compute the accuracy
def compute_accuracy(model, dataloader):

    model = model.eval()
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(dataloader):
        
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
print("Training:")
# training loop
for epoch in range(num_epochs):
    
    model.train()               # train
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)                # output in logit
        loss = F.cross_entropy(logits, labels)  # Loss function
        
        optimizer.zero_grad()       # reset the last gradient, mandatory before the backpropagation
        loss.backward()             # start backpropagation
        optimizer.step()            # update weight and bias
    
        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()                # Optional model evaluation
 
print("\nTrained model:")
model.eval()
# Inference without gradient
with torch.no_grad():
    outputs = model(X_train)
print("Logit output: ",outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print("Softmax output: ",probas)

predictions = torch.argmax(probas, dim=1)
print("Prediction, compinent with highet value of softmax: ",predictions)

predictions = torch.argmax(outputs, dim=1)
print("Prediction, compinent with highet value of logit: ",predictions)

predictions == y_train
torch.sum(predictions == y_train)

# compute accuracy
compute_accuracy(model, train_loader)
compute_accuracy(model, test_loader)

# A.8 Saving and loading models

torch.save(model.state_dict(), "model.pth")     # save model
model = NeuralNetwork(2, 2)                     # needs to match the original model exactly
model.load_state_dict(torch.load("model.pth"))  # load model