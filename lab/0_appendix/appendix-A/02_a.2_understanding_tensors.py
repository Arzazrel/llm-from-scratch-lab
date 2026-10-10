"""
Appendice A - Introduzione a PyTorch
Sezione: A.2 Understanding tensors
"""
# A.2 Understanding tensors

import torch
import numpy as np

print("Understanding tensors")
print("-"*50)

# ---------- A.2.1 Scalars, vectors, matrices, and tensors ----------
print("\nA.2.1 Scalars, vectors, matrices, and tensors")

tensor0d = torch.tensor(1)              # create a 0D tensor (scalar) from a Python integer
tensor1d = torch.tensor([1, 2, 3])      # create a 1D tensor (vector) from a Python list
# create a 2D tensor from a nested Python list (Matrix)
tensor2d = torch.tensor([[1, 2], 
                         [3, 4]])
# create a 3D tensor from a nested Python list (Cube)
tensor3d_1 = torch.tensor([[[1, 2], [3, 4]], 
                           [[5, 6], [7, 8]]])

# create a 3D tensor from NumPy array (Cube)
ary3d = np.array([[[1, 2], [3, 4]], 
                  [[5, 6], [7, 8]]])
print("Create a 3D tensor (NumPy array): ", ary3d)
                  
tensor3d_2 = torch.tensor(ary3d)        # Copies NumPy array
tensor3d_3 = torch.from_numpy(ary3d)    # Shares memory with NumPy array
print("Create tensor3d_2 with torch.tensor and tensor3d_3 with from_numpy.\nIndex [dim,row,column]")

# è un vettore a 3 dimensioni, vado a modificare la posizione con indici [0,0,0]
ary3d[0, 0, 0] = 0
ary3d[0,1,0] = 111
ary3d[0,0,1] = 222
ary3d[1,0,0] = 333
ary3d[1,1,1] = 999

print("Change ary3d.\n - 'torch.tensor' copies NumPy array\n - 'from_numpy' shares memory with NumPy array")
print("Print tensor3d_2: ",tensor3d_2)                       # remains unchanged
print("Print tensor3d_3: ",tensor3d_3)                       # changes because of memory sharing

# ---------- A.2.2 Tensor data types ----------
print("\nTensor data types")

tensor1d = torch.tensor([1, 2, 3])
print("tensor1d [1, 2, 3] - ", tensor1d.dtype)

floatvec = torch.tensor([1.0, 2.0, 3.0])
print("floatvec [1.0, 2.0, 3.0] - ", floatvec.dtype)

floatvec = tensor1d.to(torch.float32)
print("floatvec conversion from tensor1d to torch.float32 - ", floatvec.dtype)

# ---------- A.2.3 Common PyTorch tensor operations ----------
print("\nCommon PyTorch tensor operations")

tensor2d = torch.tensor([[1, 2, 3], 
                         [4, 5, 6]])
print("Create the 2D tensor [[1, 2, 3],[4, 5, 6]]")
print(tensor2d)

print("\nOperation:")
print("- shape -> Returns the dimensions of the tensor (its shape), indicating the number of elements for each dimension.\n", tensor2d.shape)
print("\n- reshape(3, 2)- 3 rows and 2 columns", tensor2d.reshape(3, 2))
print("\- view\n", tensor2d.view(3, 2))
print("\n-.T -> Calculate the transpose of the two-dimensional tensor by swapping the axes (rows become columns and vice versa).\n",tensor2d.T)
print("\nPerforms the explicit matrix product (matrix dot product) between the tensor and its transpose using the matmul() method.\n",tensor2d.matmul(tensor2d.T))
print("\nIt performs the exact same matrix product as the previous operation, utilizing the infix operator `@` introduced in Python and supported by PyTorch as a shortcut for `matmul()`.\n",tensor2d @ tensor2d.T)