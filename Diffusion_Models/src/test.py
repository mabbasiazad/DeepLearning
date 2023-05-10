import torch
import torch.nn as nn 
import numpy as np

timestep = torch.linspace(0, 999, 8)
print(timestep)
exit()

loss = nn.MSELoss()
input = torch.randn(1, 2, requires_grad=True)
target = torch.randn(1, 2)
print(input)
print(target)
my_output = torch.sqrt(torch.sum((input - target) **2))
print('my calculated output: ', my_output)
output = loss(input, target)
print('output: ', output)
output.backward()

