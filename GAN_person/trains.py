import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from  nets import G_Net,D_Net
from dataset import Datasets

if not os.path.exists("./img"):
    os.makedirs("./img")
if not os.path.exists("./params"):
    os.makedirs("./params")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
batch_size = 200
num_epoch = 10
data = Datasets(r"D:\shujuji\人脸图像生成GAN")
data = DataLoader(data,batch_size,shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# d_net = D_Net().to(device)
# g_net = G_Net().to(device)
d_net =torch.load("dnet1107.pth").to(device)
g_net = torch.load("gnet1107.pth").to(device)
loss_fn = nn.BCELoss()

d_opt = torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5, 0.999))
g_opt = torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5, 0.999))
to_image = transforms.ToPILImage()
for epoch in range(500):
    for i,img in enumerate(data):
        real_img  = img.to(device)
        real_label = torch.ones(img.size(0),1).to(device)
        fake_label = torch.zeros(img.size(0),1).to(device)

        real_out = d_net(real_img).reshape(-1,1)#D输出的是一个概率值

        d_loss_real = loss_fn(real_out,real_label)

        z = torch.randn(img.size(0),256,1,1).to(device)
        fake_img = g_net(z)
        fake_out = d_net(fake_img).reshape(-1,1)
        # print(real_out,fake_out)
        d_loss_fake = loss_fn(fake_out,fake_label)

        d_loss = d_loss_real+d_loss_fake
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        '''训练生成器'''
        if i%5==0:
            z = torch.randn(img.size(0),256,1,1).to(device)
            #z的维度，不能太高，之前设的256，30轮左右就变黑了。128就差不多了。
            fake_img = g_net(z)
            g_fake_out = d_net(fake_img).reshape(-1,1)
            g_loss = loss_fn(g_fake_out,real_label)
            #g的损失根本就降不下去，理解。
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()
        real_img  = (real_img*0.5+0.5)
        fake_img = (fake_img * 0.5 + 0.5)
        # real_img.show()
        if i%10 == 0:
            print("批次:{}/{},d_损失:{:.3f},"
                  "g_损失:{:.3f},d_real_概率值:{:.3f},d_fake_概率值:{:.3f}"
                  .format(epoch,i/10,d_loss.item(),g_loss.item(),real_out.data.mean(),fake_out.data.mean()))
            # real_img = real_img.data.reshape(-1,1,28,28)
            # fake_img = fake_img.data.reshape(-1,1,28,28)
            # torch.save(d_net,"dnet1104.pth")
            # torch.save(g_net, "gnet1104.pth")
        if i % 50 == 0:

            save_image(real_img, "./img/{}-{}-real_img.jpg".format(epoch , i), nrow=10,normalize=True,scale_each=True)
            save_image(fake_img, "./img/{}-{}-fake_img.jpg".format(epoch , i), nrow=10,normalize=True,scale_each=True)
            print("保存图片成功")
    torch.save(d_net, "dnet1107.pth")
    torch.save(g_net, "gnet1107.pth")
    print("保存网络成功")

