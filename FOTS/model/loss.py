import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss

class DetectionLoss(torch.nn.Module):
    def __init__(self, lambda_reg=1, lambda_angle=10):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_angle = lambda_angle

    def forward(self, score_map_pred, geo_map_pred, angle_map_pred, 
        score_map_true, geo_map_true, angle_map_true, training_mask):
        L_cls = self.cls_loss(score_map_pred, score_map_true, training_mask)
        L_reg = self.reg_loss(geo_map_true, geo_map_pred, angle_map_true, angle_map_pred)
        num_positive_samples = torch.sum(training_mask)
        L_cls = torch.sum(L_cls * training_mask) / num_positive_samples
        L_reg = torch.sum(L_reg * training_mask) / num_positive_samples
        L_detect = L_cls + self.lambda_reg * L_reg
        return {
            'cls_loss': L_cls, 
            'reg_loss': L_reg, 
            'detect_loss': L_detect
        }

    def cls_loss(self, score_map_pred, score_map_true, training_mask):
        """ Classification loss. """
        return F.binary_cross_entropy(
            input=score_map_pred, 
            target=score_map_true, 
            reduction='none'
        )

    def reg_loss(self, geo_map_pred, geo_map_true, angle_map_pred, angle_map_true):
        """ Regression loss. """
        L_iou = self.iou_loss(geo_map_pred, geo_map_true)
        L_angle = torch.cos(angle_map_pred - angle_map_true)
        return L_iou + self.lambda_angle * (1 - L_angle)

    def iou_loss(self, geo_map_pred, geo_map_true, smooth=1):
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
        return -torch.log(iou)

class RecognitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss = CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def forward(self, indexed_tokens_pred, seq_lens_pred, indexed_tokens_true, seq_lens_true):
        return self.ctc_loss(indexed_tokens_pred, indexed_tokens_true, seq_lens_pred, seq_lens_true)

class FOTSLoss(nn.Module):
    def __init__(self, lambda_recog=1):
        super().__init__()
        self.detection_loss = DetectionLoss()
        self.recognition_loss = RecognitionLoss()
        self.lambda_recog = lambda_recog

    def forward(self, score_map_pred, geo_map_pred, angle_map_pred,
            score_map_true, geo_map_true, angle_map_true, training_mask,
            indexed_tokens_pred, seq_lens_pred, indexed_tokens_true, seq_lens_true):
        L_detect_dict = self.detection_loss(
            score_map_pred, geo_map_pred, angle_map_pred,  
            score_map_true, geo_map_true, angle_map_true,
            training_mask
        )
        L_recog = self.recognition_loss(
            indexed_tokens_pred, seq_lens_pred, 
            indexed_tokens_true, seq_lens_true
        )
        L = L_detect_dict['detect_loss'] + self.lambda_recog * L_recog
        return {
            'cls_loss': L_detect_dict['cls_loss'], 
            'reg_loss': L_detect_dict['reg_loss'], 
            'detect_loss': L_detect_dict['detect_loss'], 
            'recog_loss': L_recog, 
            'fots_loss': L
        }