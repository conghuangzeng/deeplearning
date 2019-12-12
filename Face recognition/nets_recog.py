import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import torch
import  numpy as np
class residual_block(nn.Module):#残差块
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, bias=False, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu(bn2)

        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        if self.downsample is not None:
            residual = self.downsample(x)
        bn3 += residual
        out = self.relu(bn3)

        return out


class Resnet(nn.Module):
    def __init__(self, layers,feature_num, numclass):
        self.inplanes = 64
        super(Resnet, self).__init__()  ## super函数是用于调用父类(超类)的一个方法
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)  ##inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(residual_block, 64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(residual_block, 128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(residual_block, 256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(residual_block, 512, blocks=layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.fc = nn.Linear(512 * residual_block.expansion, feature_num)
        self.arcloss = ArcLoss(feature_num,numclass)
        # self.arcloss1 = ArcMarginProduct(feature_num,numclass)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.expansion * planes:
            # print(planes, blocks)
            ## torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample))  ###该部分是将每个blocks的第一个residual结构保存在layers列表中,这个地方是用来进行下采样的
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  ##该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)

    def forward(self, x,label):
        x = self.conv1(x)#N,64,48*48
        bn1 = self.bn1(x)#BatchNorm2d
        relu = self.relu(bn1)
        maxpool = self.maxpool(relu)#nn.MaxPool2d(kernel_size=3, stride=2, padding=1),N,64,56*56
        layer1 = self.layer1(maxpool)##N,256,24*24
        layer2 = self.layer2(layer1)#N,512,12*12
        layer3 = self.layer3(layer2)#N,1024,6*6
        layer4 = self.layer4(layer3)#N,2048,3*3
        # print(layer1.shape)
        # print(layer2.shape)
        # print(layer3.shape)
        # print(layer4.shape)

        avgpool = self.avgpool(layer4)# nn.AvgPool2d(kernel_size=3, stride=1)
        x = avgpool.view(avgpool.size(0), -1)#变成nv结构
        x = self.fc(x)#nn.Linear(2048, feature_num)
        # print(x.shape)#(N,128)
        output, cosa = self.arcloss(x,label)
        return x,output,cosa

class ArcLoss(nn.Module):
    def __init__(self, in_feathers, out_feathers, s=64, m=0.50, easy_margin=True):
        super(ArcLoss, self).__init__()
        self.in_feathers = in_feathers
        self.out_feathers = out_feathers
        self.s = s
        self.m = m
        self.mm = math.sin(math.pi - self.m) * self.m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.randn(self.out_feathers, self.in_feathers))
        nn.init.xavier_uniform_(self.weight)  # 使用均匀分布来初始化weight
        # 差值的cos和sin
        self.th = math.cos(math.pi - self.m)
        # 阈值，避免theta + m >= pi

    def forward(self, input,label):
        X = F.normalize(input)  # x/|x|
        W = F.normalize(self.weight)  # w/|w|
        cosa = F.linear(X, W)/10  # cosa=(x*w)/(|x|*|w|)
        # cosa = F.linear(X, W).clamp(-1, 1)  # cosa=(x*w)/(|x|*|w|)
        # print(X.shape,W.shape,cosa.shape)
        # cosam = cosa - self.m
        a = torch.acos(cosa)
        cosam = torch.cos(a + self.m)  # cosam=cos(a+m)
        # sina = torch.sqrt(1.0 - torch.pow(cosa, 2) + 0.001)
        # cosam = cosa * math.cos(self.m) - sina * math.sin(self.m)  # cosam=cos(a+m)
        if self.easy_margin:
            cosam = torch.where(cosa > 0, cosam, cosa)
        # 如果使用easy_margin
        else:
            cosam = torch.where(cosa < self.th, cosam, cosa)
        one_hot = torch.zeros(cosa.size()).cuda().scatter_(1, label.view(-1, 1), 1)
        # 将样本的标签映射为one hot形式 例如N个标签，映射为（N，num_classes）
        output = self.s * (one_hot * cosam) + self.s * ((1.0 - one_hot) * cosa)
        # 对于正确类别（1*phi）即公式中的cos(theta + m)，对于错误的类别（1*cosine）即公式中的cos(theta）
        # 这样对于每一个样本，比如[0,0,0,1,0,0]属于第四类，则最终结果为[cosine, cosine, cosine, cosam, cosine, cosine]
        # 再乘以半径，经过交叉熵，正好是ArcFace的公式
        return output, cosa
class ArcMarginProduct(nn.Module):
    def __init__(self,in_features,out_features,s=30.0,m=0.50,easy_margin=False):
        super(ArcMarginProduct,self).__init__()
        self.in_features = in_features#前面4行需要先把init里面的变量全部定义了，不然把网络放到优化器里面，会有报警，说这个类的网络参数是一个空的list
        self.out_features = out_features
        self.s = s
        self.m = m
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)#给w赋值
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)#常数
        self.sin_m = math.sin(m)#常数
        self.th = math.cos(math.pi - m)#cos(math.pi - m)=-cos(m)，就是一个常数
        self.mm = math.sin(math.pi - m) * m#就是一个常数
#
#
    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        input  = input.to(self.device)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.001 - torch.pow(cosine, 2)).clamp(0, 1))#1.000001也是为了防止梯度爆炸和梯度消失，防止cos = ±1的情况
        phi = cosine * self.cos_m - sine * self.sin_m#phi=cos（a+m），cosine=cosa
        label = label.to(self.device)
        if self.easy_margin:#感觉phi的取值可以简化
            phi = torch.where(cosine > 0, phi, cosine)#这句有用，相当于是取a是0-90度，取cos（a+m）a是90-180度，取cosa，当phi取cosa的时候，output固定 = 30
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        label = label.cpu().long()
        one_hot = torch.zeros(cosine.size()).scatter_(1, label.view(-1, 1), 1).to(self.device)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4#w的模为1，x的模为1.相当于exp也不要了，只要cos（a+m）和cosa，原公式理解为cos（a+m）除以（cos（a+m）+cosa-onehot对应的cosa）
        output *= self.s#output太小了，乘以一个常数，30
        # print(output)

        return output,cosine
if __name__ == '__main__':
    x = torch.randn(20,3,96,96).cuda()
    layers = [3, 4, 6, 3]
    feature_num =512
    numclass = 20#zhege shi 每一张脸的特征维度
    label = torch.tensor(np.random.randint(0, 20, (1, 20))).long().cuda()
    net = Resnet(layers=layers,feature_num=feature_num,numclass=numclass).cuda()
    y = net(x,label)
    print(y[0].shape)
    print(y[1].shape)
    # arcloss = ArcLoss(128,20).cuda()
    #
    # y1 = arcloss(y,label)
    # print(y1[0].shape)

