import torch
data = [[[1, 2], [2, 3]], [[1, 2], [2, 3]]]
x_data = torch.tensor(data)
print(x_data)
print(x_data.shape)
print(x_data.ndim)