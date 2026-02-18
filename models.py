import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainBlock(nn.Module):
    """
    A standard convolutional block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU.
    No skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out) 
        return out

class SkipBlock(nn.Module):
    """
    A residual block: (Conv -> BN -> ReLU -> Conv -> BN) + Identity -> ReLU.
    Contains a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkipBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BaseCNN(nn.Module):
    """
    Common architecture backbone for both Skip and Plain versions.
    """
    def __init__(self, block_type, num_blocks, num_classes=10, init_channels=64):
        super(BaseCNN, self).__init__()
        self.in_channels = init_channels
        self.conv1 = nn.Conv2d(3, init_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_channels)
        
        self.layers = nn.ModuleList()
        channels = [init_channels, init_channels*2, init_channels*4, init_channels*8]
        
        if init_channels == 16:
            channels = [16, 32, 64]
            
        for i, num_block in enumerate(num_blocks):
            stride = 1 if i == 0 else 2
            self.layers.append(self._make_layer(block_type, channels[i], num_block, stride=stride))
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[len(num_blocks)-1], num_classes)

    def _make_layer(self, block_type, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block_type(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def get_resnet18_plain(num_classes=10):
    return BaseCNN(PlainBlock, [2, 2, 2, 2], num_classes, init_channels=64)

def get_resnet18_skip(num_classes=10):
    return BaseCNN(SkipBlock, [2, 2, 2, 2], num_classes, init_channels=64)

def get_resnet56_plain(num_classes=10):
    return BaseCNN(PlainBlock, [9, 9, 9], num_classes, init_channels=16)

def get_resnet56_skip(num_classes=10):
    return BaseCNN(SkipBlock, [9, 9, 9], num_classes, init_channels=16)
