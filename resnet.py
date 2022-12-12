import torch 
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic block in Resnet with 2 convolutions"""

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=channels,
                               kernel_size=3,
                               padding=1,
                               stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=3,
                                padding=1,
                                stride=1,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = None
        if stride != 1 or in_channels != channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                            out_channels=channels,
                            kernel_size=1,
                            stride=2,
                            bias=False),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        """forward pass with residual connection"""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut != None:
            out += self.shortcut(x)
        else:
            out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet with only basic block"""

    def __init__(self, layers=[2,2,2,2], num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

        # init weights to xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)


    def _make_layer(self, channels, blocks, stride):
        layers = []

        for _ in range(0, blocks):
            layers.append(
                BasicBlock(in_channels=self.in_channels,
                            channels=channels,
                            stride=stride)
            )
            stride = 1
            self.in_channels = channels


        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
