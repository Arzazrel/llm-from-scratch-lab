"""
Appendice A - Introduzione a PyTorch
Sezione: A.6 Setting up efficient data loaders
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# A.6 Setting up efficient data loaders

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

print("\nData input for training:\n- data: ",X_train,"\n- label: ",y_train)
print("Data input for testing:\n- data: ",X_test,"\n- label: ",y_test)

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

print("\nLength of training set: ",len(train_ds))
print("Length of testing set: ",len(test_ds))

torch.manual_seed(123)

# define the loader for the training set
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0           # load with only the main thread
)

# define the loader for the testing set
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0           # load with only the main thread
)

print("\nSee the train_loader and test_loader")
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
for idx, (x, y) in enumerate(test_loader):
    print(f"Batch {idx+1}:", x, y)


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

print("\nRedo the set and now see the train_loader and test_loader")
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
for idx, (x, y) in enumerate(test_loader):
    print(f"Batch {idx+1}:", x, y)

