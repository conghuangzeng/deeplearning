from  torchvision import transforms,datasets
import os
import torch.utils.data as data
import PIL.Image as pimg
import numpy as np
import torch
transform=transforms.Compose([

    transforms.Resize((52,144)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
)
label_path = r"label.txt"
class dataset(data.Dataset):

    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, label_path)).readlines())
        # self.to_img = transforms.ToPILImage()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.dataset[index]).strip().split()
        img_data1  = pimg.open(img_path[0])
        img_data = transform(img_data1)
        label = torch.tensor(np.array(img_path[1],dtype=np.int32))
        return img_data,label


if __name__ == '__main__':
    dataset = dataset(path=r"./dataset",transform=transform)
    img_data = dataset[0][0]
    print(img_data.shape)
    label = dataset[1][1]
    print(label)


