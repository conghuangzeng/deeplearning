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
        img_data1 = pimg.open(img_path)
        img_data = self.transform(img_data1)
        labels = img_path.split(".")
        # labels = labels.split(".")
        # print(labels)
        label = torch.tensor(np.array(labels[1:5], dtype=np.float32))
        # print(label)
        label = self.one_hot(label)
        # print(label)
        return img_data,label

    # def one_hot(self,x):
    #     zeros = np.zeros([4,10])
    #     for i in range(4):
    #        index = x[i]
    #        # print(index)
    #        zeros[i][int(index)]=1
    #     return zeros


if __name__ == '__main__':

    my_data = Sampling(r"train_img")
    print(my_data[1])
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


