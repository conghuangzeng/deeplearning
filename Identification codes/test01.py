import numpy as np
# a = "12345"
import torch
# b = list(a)
# print(b)
# c = "".join(b)
# print(c)
# torch.repeat# print([c])
# a =torch.Tensor([1,2,3,4,5,6,7,8,9])
# print(a)
# print(a.shape)
# a = a.long()
# c = a.view(-1,1)
# print(c)
# # target = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1).to(device)
# b = torch.zeros(a.size()[0],10).scatter_(1,a.view(-1,1),1)
# print(b)
# a =np.array([1,2,3,4,5,6,7,8,9])
# print(a)
# print(a.shape)
# # a = a.long()
# c = a.reshape(-1,1)
# print(c)
# # target = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1).to(device)
# b = np.zeros(a.shape[0],10).scatter(1,c,1)
# print(b)
import torch
a= torch.arange(30).reshape(5,6)
print(a)
print('b:',a.repeat(2,2).shape)
print('c:',a.repeat(2,1,1).shape)
