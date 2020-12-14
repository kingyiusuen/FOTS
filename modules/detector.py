import torch
import torch.nn as nn
import numpy as np

class Detector(nn.Module):
    def __init__(self, crop_size=640):
        super().__init__()
        self.crop_size = crop_size
        self.score_map_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.geo_map_conv = nn.Conv2d(64, 4, kernel_size=1)
        self.angle_map_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, shared_features):
        """
        Args:
            shared_features (Tensor): (batch_size, 64, 160, 160). Output of
                shared convolutions. 
        
        Returns:
            score_maps (Tensor): (batch_size, 1, 160, 160).
            geo_maps (Tensor): (batch_size, 4, 160, 160).
            angle_maps (Tensor): (batch_size, 1, 160, 160).
        """
        # have to rescale geo_maps and angle_maps because
        # the range of the output of sigmoid is [0, 1] but
        # the ranges of distance and angle are not
        score_maps = self.score_map_conv(shared_features)
        score_maps = torch.sigmoid(score_maps)
        geo_maps = self.geo_map_conv(shared_features)
        geo_maps = torch.sigmoid(geo_maps) * self.crop_size
        angle_maps = self.angle_map_conv(shared_features)
        angle_maps = (torch.sigmoid(angle_maps) - 0.5) * np.pi / 2
        return score_maps, geo_maps, angle_maps