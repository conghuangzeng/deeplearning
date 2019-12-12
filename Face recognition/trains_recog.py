import torch.nn as nn
# import math
import torch
import torch.utils.data as data
from nets_recog import Resnet
from torchvision import datasets, transforms


transform = transforms.Compose([
    # transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
layers = [3, 4, 6, 3]
numclass = 10
feature_num = 512
net = Resnet(layers=layers,feature_num=feature_num,numclass=numclass)
net = torch.load(r"resnet_star_1115.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
net.train()
data_path= r"D:\shujuji\star face net\train_img"
# batch_size = 10
mydata =datasets.ImageFolder(data_path,transform=transform)
train_data  =  data.DataLoader(mydata,batch_size=50,shuffle=True)
loss_cross_fun = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(),lr = 0.001)
total = 0
correct = 0
for epoch in range(200):
    for i, (img, label) in enumerate(train_data):
        img, label = img.to(device), label.to(device)
        feature,output, cosa = net(img, label)
        # print(output.shape)
        index = torch.argmax(cosa, dim=1)
        # print(index)
        loss = loss_cross_fun(output, label)
        # print(label)

        # print(output)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total += img.size()[0]
        correct += (label.data == index.data).sum()
        acc = correct.float() / total
        # print("精确率{0}".format(acc))
        if i % 10 == 0:
            print("轮次：{0}，批次：{1},损失：{2}".format(epoch,i/10,loss.item()))
            print("精确率：{0}".format(acc))
    if epoch %5==0 and epoch>0:

        torch.save(net,"resnet_star_1115.pth")
        print("保存网络成功")










