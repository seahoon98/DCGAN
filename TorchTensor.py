import torch
import numpy as np

# 1 Dimensional tensor created using torch
torch_array = torch.tensor([0, 1, 2, 3])

# 1 Dimensional array created by numpy
np_array = np.array([0, 1, 2, 3])

# It is possible to transform numpy array into tensor
torch_from_array = torch.from_numpy(np_array)

# It is also possible to transform numpy array into tensor.
torch2numpy_array = torch_from_array.numpy()

# 2 Dimensional Tensor
np_array_2D = np.array([[0, 1, 2], [3, 4, 5]])
torch_array_2D = torch.from_numpy(np_array_2D)
# transform torch tensor to 2 dimensional numpy array 
torch2np_array = torch_array_2D.numpy()

def OneDim_tensor():
    print(torch_array)
    print(np_array)
    print(torch_from_array)
    print(torch2numpy_array)

def TwoDim_tensor():
    print(np_array_2D)
    print(torch_array_2D)
    print(torch2np_array)

#OneDim_tensor()
TwoDim_tensor()
