import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.cbam import *
import scipy.misc

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
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

import torch
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=5, zero_init_residual=False, norm_layer=None,
                 crossCBAM=False,origin=False,crosspatialCBAM= False,  test=False, simple=False, choice=""):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        self.num_classes = num_classes
        self.crossCBAM = crossCBAM
        self.crosspatialCBAM = crosspatialCBAM
        self.choice = choice
        self.test = test
        self.simple = simple
        self.origin = origin

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.simple:
            self.dropout = nn.Dropout(0.3)
            self.branch_bam = CBAM(512 * block.expansion)
            self.classifier_dep = nn.Linear(512 * block.expansion, 1024)
            self.classifier = nn.Linear(1024, self.num_classes)

        elif self.test:
            self.dropout = nn.Dropout(0.3)
            self.branch_bam1 = CBAM(512 * block.expansion)
            self.branch_bam2 = CBAM(512 * block.expansion)
            self.classifier_dep1 = nn.Linear(512 * block.expansion, 1024)
            self.classifier_dep2 = nn.Linear(512 * block.expansion, 1024)
            self.classifier_dep3 = nn.Linear(512 * block.expansion, 1024)
            #self.conv2 = nn.Conv2d(4096, 2048, kernel_size=3, stride=1, padding=1)
            self.classifier = nn.Linear(1024, self.num_classes)
            self.classifier_specific_1 = nn.Linear(1024, self.num_classes)
            self.classifier_specific_2 = nn.Linear(1024, self.num_classes)

        elif self.crossCBAM:
            self.dropout = nn.Dropout(0.3)
            self.branch_bam1 = CBAM(512 * block.expansion)
            self.branch_bam2 = CBAM(512 * block.expansion)
            self.classifier_dep1 = nn.Linear(512 * block.expansion, 1024)
            self.classifier_dep2 = nn.Linear(512 * block.expansion, 1024)
            self.branch_bam3 = CBAM(1024, no_spatial=True)
            self.branch_bam4 = CBAM(1024, no_spatial=True)

            self.classifier = nn.Linear(1024, self.num_classes)
            self.classifier_specific_1 = nn.Linear(1024, self.num_classes)
            self.classifier_specific_2 = nn.Linear(1024, self.num_classes)

        elif self.crosspatialCBAM:
            self.dropout = nn.Dropout(0.3)
            self.branch_bam1 = CBAM(512 * block.expansion)
            self.branch_bam2 = CBAM(512 * block.expansion)
            self.classifier_specific_1 = nn.Linear(512 * block.expansion, self.num_classes)
            self.classifier_specific_2 = nn.Linear(512 * block.expansion, self.num_classes)
            self.branch_bam3 = CBAM(2048)
            self.branch_bam4 = CBAM(2048)
            self.classifier = nn.Linear(2048, self.num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def calFeature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.origin:
            x = self.dropout(x)

        return x

    def forward(self, x1, x2=None):

        if(x2==None):
            x1 = self.calFeature(x1)
            if not self.origin:
                x1 = self.branch_bam(x1)
                out1 = self.avgpool(x1)
                out1 = out1.view(out1.size(0), -1)
                out1 = self.classifier_dep(out1)
                out1 = self.classifier(out1)
                return out1
            else:
                x1 = self.avgpool(x1)
                x1 = torch.flatten(x1, 1)
                out1 = self.fc(x1)
                return out1

        x1 = self.calFeature(x1)
        x2 = self.calFeature(x2)

        if self.origin:
            x1 = self.avgpool(x1)
            x1 = torch.flatten(x1, 1)
            x1 = self.fc(x1)

            x2 = self.avgpool(x2)
            x2 = torch.flatten(x2, 1)
            x2 = self.fc(x2)

            return x1,x2

        if self.simple:
            x1 = self.branch_bam(x1)
            x2 = self.branch_bam(x2)

            out1 = self.avgpool(x1)
            out2 = self.avgpool(x2)
            # print(out1.size())

            out1 = out1.view(out1.size(0), -1)
            out2 = out2.view(out2.size(0), -1)

            out1 = self.classifier_dep(out1)
            out2 = self.classifier_dep(out2)

            out1 = self.classifier(out1)
            out2 = self.classifier(out2)

            return out1, out2

        elif self.test:

            # #  task specific feature
            x1 = self.branch_bam1(x1)
            x2 = self.branch_bam2(x2)
            # print(x1.size())

            out1 = self.avgpool(x1)
            out2 = self.avgpool(x2)
            # print(out1.size())

            out1 = out1.view(out1.size(0), -1)
            out2 = out2.view(out2.size(0), -1)

            out1 = self.classifier_dep1(out1)
            out2 = self.classifier_dep2(out2)

            out1 = self.classifier_specific_1(out1)
            out2 = self.classifier_specific_2(out2)

            x = torch.stack((x1, x2), dim=0).sum(dim=0)
            # x = torch.cat((x1, x2), dim=1)

            # print(x.size())
            # x = self.conv2(x)
            x = self.avgpool(x)
            # print(x.size())

            x = x.view(x.size(0), -1)
            x = self.classifier_dep3(x)

            x = self.classifier(x)

            return out1, out2, x

        elif self.crossCBAM:
            x1 = self.avgpool(self.branch_bam1(x1))
            x2 = self.avgpool(self.branch_bam2(x2))

            # #  task specific feature
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            x1 = self.classifier_dep1(x1)
            x2 = self.classifier_dep2(x2)

            out1 = self.classifier_specific_1(x1)
            out2 = self.classifier_specific_2(x2)
            #
            # # learn task correlation
            x1_att = self.branch_bam3(x1.view(x1.size(0), -1, 1, 1))
            x2_att = self.branch_bam4(x2.view(x2.size(0), -1, 1, 1))

            x1_att = x1_att.view(x1_att.size(0), -1)
            x2_att = x2_att.view(x2_att.size(0), -1)

            x1 = torch.stack([x1, x2_att], dim=0).sum(dim=0)
            x2 = torch.stack([x2, x1_att], dim=0).sum(dim=0)

            # final classifier
            x1 = self.classifier1(x1)
            x2 = self.classifier2(x2)

            return x1, x2, out1, out2

        elif self.crosspatialCBAM:
            x1 = self.branch_bam1(x1)
            x2 = self.branch_bam2(x2)
            # print (x1.shape)
            # print (x2.shape)
            out1 = self.avgpool(x1)
            out2 = self.avgpool(x2)
            out1 = out1.view(out1.size(0), -1)
            out2 = out2.view(out2.size(0), -1)

            # 每张图单独的类别
            out1 = self.classifier_specific_1(out1)
            out2 = self.classifier_specific_2(out2)
            
            x1_att = self.branch_bam3(x1)
            # x2_att = self.branch_bam4(x1)

            # print (x1.shape, x2_att.shape)
            # element-wise sum
            # x1 = torch.stack([x1, x2_att], dim=0).sum(dim=0)
            x2 = torch.stack([x2, x1_att], dim=0).sum(dim=0)

            # x1 = self.avgpool(x1)
            x2 = self.avgpool(x2)

            x = x2
            x = x.view(x.size(0), -1)

            # x1 = x1.view(x1.size(0), -1)
            # x2 = x2.view(x2.size(0), -1)

            x = self.classifier(x)

            return out1, out2, x

        else:
            out1 = self.avgpool(x1)
            out1 = torch.flatten(out1, 1)
            out1 = self.fc(out1)

            out2 = self.avgpool(x1)
            out2 = torch.flatten(out2, 1)
            out2 = self.fc(out2)

            return out1, out2


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


