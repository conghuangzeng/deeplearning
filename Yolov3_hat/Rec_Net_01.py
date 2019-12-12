import torch.nn as nn
import torch

#将图片尺寸固定为52*144
from torchvision.models import resnet50
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
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
class ResNet50(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.arcloss = ArcLoss(1024,4)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#这个方法不管是多大的特征图都会搞成1*1的,不好
        self.fc1 = nn.Linear(1024, 4)
        # 这个是自适应平均池化，感觉不受尺寸限制，不好，另一个方面来说，尺寸不收限制，网络也可以不用改，要固定尺寸就用平均池化

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x,label):
        x = self.conv1(x)  # 224*224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 72*26

        x = self.layer1(x)
        # print(x.shape)#256*36*13
        x = self.layer2(x)
        # print(x.shape)#512, 18*6
        x = self.layer3(x)
        # print(x.shape)#1024, 9*3
        # x = self.layer4(x)
        # print(x.shape)  # 2048, 7, 7
        # x = self.avgpool(x)  # 7*7
        # print(x.shape)#2048*1*1
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        x = avgpool(x)
        # print(x.shape)
        x  = x.reshape(x.size()[0],-1)
        # print(x.shape)#N,2048

        # x = self.fc1(x)
        output,cosa = self.arcloss(x,label)
        # x = self.fc(x)  # 最后网络输出的结果是一个N,1000的数据

        return output,cosa # N*4


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


if __name__ == '__main__':
    net = ResNet50(block=Bottleneck, layers=[1, 2, 3, 1]).cuda()
    #     # return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
    #     #                **kwargs)
    # net.load_state_dict(torch.load(r"resnet50-19c8e357.pth"))
    # my_resnet.load_state_dict(torch.load("my_resnet.pth"))
    x = torch.randn(1, 3, 144,52).cuda()
    label = torch.tensor([1.0]).cuda()
    label  = label.long()
    # x = torch.randn(1, 3, 150, 150)
    y,_ = net(x,label)
    print(y.shape)


