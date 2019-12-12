import random
import numpy as np
import torch
# a = np.arange(24).reshape(6,4)
# print(a)
# b = np.arange(6).reshape(6,1)
# print(b)
# # indexs = torch.lt(b,5)
# # print(indexs)
# indexs = np.where(b>3)
# print(indexs)
# cond = b[indexs[0]][:,0]
# print(cond)
# x1 = a[indexs[0],0]
# print(x1)
# y1  =  a[indexs[0],1]
# print(y1)
# # box = np.array([x1,y1,])
# offset =np.arange(1,61).reshape(4,3,5)
# print(offset)
# cond = np.arange(1,16).reshape(3,5)
# print(cond)
#
# index = np.where(cond>4)
# _cond  = cond[index]
# print(_cond)

offset =torch.tensor([0.2,0.4,0.9,0.5,0.7,0.8,0.3,0.4,0.2,0.4,0.9,0.5,0.7,0.8,0.3,0.2,0.4,0.9,0.5,0.7,0.8,0.3,0.4,0.2,0.4,0.9,0.5,0.7,0.8,0.3,0.2,0.4,0.9,0.5,0.7,0.8,0.3,0.4,0.2,0.4,0.9,0.5,0.7,0.8,0.3,0.2,0.4,0.9,0.5,0.7,0.8,0.3,0.4,0.2,0.4,0.9,0.5,0.7,0.8,0.3],dtype=torch.float32).reshape(4,3,5)
# offset = torch.tensor(offset,dtype=)
print(offset)
cond = torch.tensor([0.2,0.4,0.9,0.5,0.7,0.8,0.3,0.4,0.2,0.4,0.9,0.5,0.7,0.8,0.3]).reshape(3,5)
# cond = torch.tensor(cond/4)
print(cond)

# index = np.where(cond>4)
# _cond  = cond[index]
# print(_cond)
index1 = torch.gt(cond,0.3)
index = torch.nonzero(index1)
print(index)
_cond =torch.tensor( cond[index[:,0],index[:,1]]).reshape(-1,1)
print(_cond)
print(_cond.size())
_offset =torch.tensor( offset[:,index[:,0],index[:,1]])
_offset = _offset.transpose(1,0)
print(_offset)
print(_offset.size())
a = torch.cat((_offset,_cond),1)
print(a.size())
print(a)
print(cond[0,1])
print(offset[:,0,1])