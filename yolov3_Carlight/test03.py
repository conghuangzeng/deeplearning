import torch

a = torch.randn(1,13,13,3,15)

index = a[...,0]>0.4
print(index.shape)

box = a[index]
print(box.shape)

indexes = torch.nonzero(index)
print(indexes.shape)

img = indexes[:,0]
print(img.shape)

img1 = indexes[:,1]
print(img1.shape)

img2 = indexes[:,2]
print(img2.shape)

img3 = indexes[:,3]
print(img3.shape)