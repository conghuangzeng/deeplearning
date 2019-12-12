from torchvision.models import resnet50
import torch.nn as nn
import torch
net = resnet50()
net.load_state_dict(torch.load(r"resnet50-19c8e357.pth"))
# print(net)
class Main_net(nn.Module):
    def __init__(self):
        super(Main_net, self).__init__()
        self.feature = net
        self.line = nn.Linear(1000,128)

    def forward(self, x):
        y = self.feature(x)
        y = self.line(y)
        return y

if __name__ == '__main__':
    x = torch.randn(10,3,1,1)
    net = Main_net()
    y = net(x)
    print(y.shape)