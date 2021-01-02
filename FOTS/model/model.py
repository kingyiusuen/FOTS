import torch
import torch.nn as nn

from FOTS.model import SharedConv, Detector
from FOTS.utils.bbox import restore_bbox
#import modules.alphabet

class FOTSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.SharedConv = SharedConv()
        self.Detector = Detector()
        #self.ROIRotate = ROIRotate()
        #self.Recognizer = Recognizer(num_of_classes=modules.alphabet.NUM_OF_CLASSES)

    def forward(self, imgs, bboxes=None, bbox_to_img_idx=None):
        """
        Args:
            imgs: Input images.
            bboxes: Coordinates of the ground-truth bounding boxes, ignored if 
                self.training is False.
            bbox_to_img_idx: Mapping between the bounding boxes and images, 
                ignored if self.training is False.
        """
        shared_features = self.SharedConv(imgs)
        score_maps, geo_maps, angle_maps = self.Detector(shared_features)
        # Quote from the FOTS paper:
        # "Different from object classification, text recognition is very
        # sensitive to detection noise. A small error in predicted text region
        # could cut off several characters, which is harmful to network
        # training, so we use ground truth text regions instead of predicted
        # text regions during training. When testing, thresholding and NMS are
        # applied to filter predicted text regions."
        #if not self.training:
        #   # get the predicted bounding boxes
        #    bboxes, bbox_to_img_idx = restore_bbox(score_maps, geo_maps, angle_maps)
        #rois, seq_lens = self.ROIRotate(shared_features, bboxes, bbox_to_img_idx)
        #log_probs = self.Recognizer(rois, seq_lens)
        return score_maps, geo_maps, angle_maps#, bboxes, bbox_to_img_idx, log_probs, seq_lens
