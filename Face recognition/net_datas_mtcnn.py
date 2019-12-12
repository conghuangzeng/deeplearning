import torch.utils.data as data
import torch
import  numpy as np
import os
import PIL.Image as pimg
from torchvision import transforms
transform = transforms.Compose([
    # transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
class Facedataset(data.Dataset):
    def __init__(self,path,transform):
        self.path=path
        self.transform=transform
        self.dataset=[]
        self.dataset.extend(open(os.path.join(path,"positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path,"negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path,"part.txt")).readlines())
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img_path=os.path.join(self.path,self.dataset[index]).strip().split()
        #将img_path理解成完成的图片路径信息，还有标签信息。
        img_data1=pimg.open(img_path[0])
        img_data=self.transform(img_data1)
        cond=torch.tensor(np.array(img_path[1],dtype=np.float32))
        offset=torch.tensor(np.array([img_path[2],img_path[3],img_path[4],img_path[5]],dtype=np.float32))

        return img_data,cond,offset
if __name__ == '__main__':#这一步是验证
#
    my_data =Facedataset(path=r"D:\celeba\48",transform=transform)
#     print(type(my_data))
#     #my_data返回三个东西img_data,cond,offset
#     # print(my_data.size())
    x = my_data[700056][1]#第一个[0]就是第一张图片，第二个[0]就是取的img_data，
    print(x)
    print(x.shape)
    x = my_data[700056][2]
    print(x)
    print(x.shape)

    # print(x.size())