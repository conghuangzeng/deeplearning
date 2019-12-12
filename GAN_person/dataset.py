import torch.utils.data as data
import torch
import  numpy as np
import os
import PIL.Image as pimg
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

class Datasets(data.Dataset):
    def __init__(self,path):
        self.transform = transform
        self.dataset = []
        self.path  = path

        self.dataset.extend(os.listdir(path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.dataset[index])
        img = pimg.open(img_path)
        img_data = self.transform(img)
        return img_data
if __name__ == '__main__':#这一步是验证
#
    my_data =Datasets(path=r"D:\shujuji\人脸图像生成GAN")
    x = my_data[50]#第一个[0]就是第一张图片，第二个[0]就是取的img_data，
    print(x.shape)