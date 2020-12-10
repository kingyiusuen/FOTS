import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss

class DetectionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, score_map_true, geo_map_true, angle_map_true, 
            score_map_pred, geo_map_pred, angle_map_pred, training_mask, lambda_reg=1):
        L_cls = self.cls_loss(score_map_true, score_map_pred, training_mask)
        L_reg = self.reg_loss(geo_map_true, geo_map_pred, angle_map_true, angle_map_pred)
        L_detect = L_cls + lambda_reg * L_reg
        return L_detect

    def cls_loss(self, score_map_true, score_map_pred, training_mask):
        """ Classification loss. """
        L_cls = F.binary_cross_entropy_with_logits(
            input=score_map_pred, 
            target=score_map_true, 
            reduction='none'
        )
        L_cls = torch.sum(L_cls * training_mask) / torch.sum(training_mask)
        return L_cls

    def reg_loss(self, geo_map_true, geo_map_pred, angle_map_true, angle_map_pred, lambda_angle=10):
        """ Regression loss. """
        L_iou = self.iou_loss(geo_map_true, geo_map_pred)
        L_angle = torch.cos(angle_map_pred - angle_map_true)
        L_reg = L_iou + lambda_angle * (1 - L_angle)
        return torch.mean(L_reg)

    def iou_loss(self, geo_map_true, geo_map_pred, smooth=1):
        """ IoU loss function. The algorithm is described in the paper
        Unitbox: An advanced object detection network.
        """
        t_true, b_true, l_true, r_true = torch.split(geo_map_true, 1, 1)
        t_pred, b_pred, l_pred, r_pred = torch.split(geo_map_pred, 1, 1)
        area_true = (t_true + b_true) * (l_true + r_true)
        area_pred = (t_pred + b_pred) * (l_pred + r_pred)
        h_intersect = torch.min(t_true, t_pred) + torch.min(b_true, b_pred)
        w_intersect = torch.min(l_true, l_pred) + torch.min(r_true, r_pred)
        area_intersect = h_intersect * w_intersect
        area_union = area_true + area_pred - area_intersect
        iou = (area_intersect + smooth) / (area_union + smooth)
        L_iou = -torch.log(iou)
        return L_iou

class RecognitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss = CTCLoss()

    def forward(self):
        pass

class FOTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.detection_loss = DetectionLoss()
        self.recognition_loss = RecognitionLoss()

    def forward(self):
        pass
