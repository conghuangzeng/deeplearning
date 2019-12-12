import torch
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_net = torch.load("gnet1107.pth").to(device)
#注意，，我们最后用的就只是生成模型而已。。
for i in range(100):
    '''训练生成器'''
    z = torch.randn(100,256,1,1).to(device)
    test_img = g_net(z)
    test_img = (test_img * 0.5 + 0.5)
    # fake_img = (fake_img * 0.5 + 0.5) * 255
    if i%10 == 0:
        print("批次:{}".format(i))
        save_image(test_img,"./test_img/{}-test_img.jpg".format(i),nrow=10,normalize=True,scale_each=True)
        #normalize=True,scale_each=True这两个选项一定要。,不然图片显得很暗，我擦 。!