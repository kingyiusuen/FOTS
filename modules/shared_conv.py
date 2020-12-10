# References: https://github.com/Wovchena/text-detection-fots.pytorch/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class Decoder(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.squeeze = conv_bn_relu(in_channels, squeeze_channels)

    def forward(self, x, encoder_features):
        _, _, H, W = encoder_features.shape
        x = self.squeeze(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat([encoder_features, x], 1)
        return x

class SharedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.encoder1 = self.resnet.layer1 # 64
        self.encoder2 = self.resnet.layer2 # 128
        self.encoder3 = self.resnet.layer3 # 256
        self.encoder4 = self.resnet.layer4 # 512
        self.center = nn.Sequential(
            conv_bn_relu(512, 512, stride=2),
            conv_bn_relu(512, 1024)
        )
        self.decoder4 = Decoder(1024, 512)
        self.decoder3 = Decoder(1024, 256)
        self.decoder2 = Decoder(512, 128)
        self.decoder1 = Decoder(256, 64)
        self.remove_artifacts = conv_bn_relu(128, 64)
        
    def forward(self, x):
        """
        Args:
            img (Tensor): (batch_size, 3, 640, 640). Input image.

        Returns:
            final (Tensor): (batch_size, 64, 160, 160). Shared features.
        """
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        f = self.center(e4)
        d4 = self.decoder4(f, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        final = self.remove_artifacts(d1)
        return final
