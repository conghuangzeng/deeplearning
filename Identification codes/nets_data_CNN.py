import os
import torch
import numpy as np
import PIL.Image as pimg
import torch.utils.data as data
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

class Sampling(data.Dataset):
    def __init__(self,path):
        self.transform = transform
        self.dataset = []
        self.path  = path

        self.dataset.extend(os.listdir(path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.dataset[index])
        # print(img_path)
        # print(self.dataset)
        img_data1 = pimg.open(img_path)
        img_data = self.transform(img_data1)
        labels = self.dataset[index].split(".")
        # print(labels)
        label = torch.tensor(np.array(labels[1:5], dtype=np.float32))
        # print(label)
        label1 = label.long()
        label1 = torch.zeros(label1.size()[0],10).scatter_(1,label1.reshape(-1,1),1)
        # label = self.one_hot(label)
        # print(label.shape)
        return img_data,label1

    def one_hot(self,x):
        zeros = np.zeros([4,10])
        for i in range(4):

           zeros[i,x[i]]=1
        return zeros


if __name__ == '__main__':

    my_data = Sampling(r"train_img")
    print(len(my_data[1]))
    print(my_data[1][1])
# train_data = data.DataLoader(my_data,batch_size=10,shuffle=True)
# for i,(img_data,label) in enumerate(train_data):
#     print(img_data.shape)
#     img1 = img_data[0]
#     tf= transforms.ToPILImage()
#     img  = tf(img1*0.5+0.5)
# #     # print(img.shape)
# #     # print(label.shape)
#     img.show()
#     print(label[0])


