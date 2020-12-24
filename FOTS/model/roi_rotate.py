import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ROIRotate(nn.Module):
    def __init__(self, aabb_height=8):
        super().__init__()
        # the height of axis aligned feature maps is set to 8, as described in
        # the FOTS paper
        self.aabb_height = aabb_height
    
    def forward(self, shared_features, bboxes, bbox_to_img_idx):
        """ Apply transformation on oriented feature maps to obtain axis-aligned
        feature maps.

        Args:
            shared_features (Tensor): (batch_size, 64, 160, 160). Output of
                shared convolutions.
            bboxes (Tensor): (N, 4, 2) where N is the total number of (oriented)
                bounding boxes in the batch. X and y coordinates of the corner 
                points of the bounding boxes.
            bbox_to_img_idx (Tensor): (N). Mapping of which image each bounding
                box corresponds to.

        Returns:
            padded_aabb (Tensor): (N, 64, H, W) where H is self.aabb_height and 
                W is the maximum width of the axis-aligned feature maps. Padded 
                axis-aligned feature maps (regions of interest).
            aabb_widths (list): (N). The widths of the bounding 
                boxes before padding.
        """
        # obb is the oriented bounding box (shared feature maps before transformation) 
        # aabb is the axis aligned bounding box (shared feature after transformation)
        # downsample by a factor of 4 because the output of the shared convolutions
        # is 1/4 of the original image size 
        transform_mats, aabb_widths = self._get_transform_mats_and_aabb_widths(bboxes / 4)
        max_aabb_width = max(aabb_widths)
        obbs = torch.stack([shared_features[img_idx] for img_idx in bbox_to_img_idx])
        # generate a sampling grid and then use bilinear interpolation to find
        # the pixel values for the rotated feature maps in the sampling grid, 
        # read this blog post to learn more about this process: 
        # https://kevinzakka.github.io/2017/01/10/stn-part1/
        grid = F.affine_grid(transform_mats, obbs.shape, align_corners=True)
        aabbs = F.grid_sample(obbs, grid, align_corners=True)
        # as mentioned in the FOTS paper, the width of text proposals may vary
        # in practice, so we pad the feature maps to the longest width and
        # ignore the padding parts in recognition loss function
        num_of_aabbs, num_of_channels, _, _ = obbs.shape
        padded_aabbs = torch.zeros((num_of_aabbs, num_of_channels, self.aabb_height, max_aabb_width))
        for i in range(num_of_aabbs):
            padded_aabbs[i,:,:,:aabb_widths[i]] = aabbs[i,:,:self.aabb_height,:aabb_widths[i]]
        # preparation work for the Bidirectional LSTM in the recognizer
        #seq_lens = np.array(aabb_widths)
        #sorted_bbox_idx = np.argsort(-seq_lens) # sort in descending order, required by pack_padded_sequence()
        #seq_lens = seq_lens[sorted_bbox_idx]
        #padded_aabbs = padded_aabbs[sorted_bbox_idx]
        aabb_widths = torch.IntTensor(aabb_widths)
        return padded_aabbs, aabb_widths#, sorted_bbox_idx

    def _get_transform_mats_and_aabb_widths(self, bboxes):
        """ Returns data that will be used for feature map transformations.

        Args:
            bboxes (ndarray): (4, 2). X and y coordinates of the corners points 
                of the bounding boxes, sorted in clockwise order.

        Returns:
            transform_mats (Tensor): (N, 3, 3). A list of 3 x 3 affine 
                transformation matrices for transforming the input bounding 
                boxes to axis-aligned bounding boxes of a pre-specified height.
            aabb_widths (List[int]): Widths for the axis-aligned feature maps.

        References:
            https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/17
        """
        transform_mats = []
        aabb_widths = []
        for bbox in bboxes:
            # To find the transformation matrix, we need three points from input
            # image (source) and their corresponding locations in output image
            # (destination). Here we pick the top left, top right and bottom left.
            tl_src, tr_src, _, bl_src = bbox
            height_src = np.linalg.norm(tl_src - bl_src)
            width_src = np.linalg.norm(tl_src - tr_src)
            # try to maintain the aspect ratio and also make sure the width of the
            # axis-aligned bounding box does not exceed maximum possible feature map
            # width (the ones at the center of shared convolutions) 
            height_dst = self.aabb_height
            width_dst = min(round(height_dst * height_src / width_src), 160) 
            tl_dst = (0, 0)
            tr_dst = (width_dst, 0)
            bl_dst = (0, 8)
            src = np.float32([tl_src, tr_src, bl_src])
            dst = np.float32([tl_dst, tr_dst, bl_dst])
            # TODO: write my own function to avoid the conversion to numpy arrays?
            M = cv2.getAffineTransform(src, dst)
            M = np.vstack((M, [0, 0, 1]))
            # need to normalize the coordinates to [-1, 1], as required by F.affine_grid()
            N1 = self._get_normalize_mat(height_src, width_src)
            N2 = self._get_normalize_mat(height_dst, width_dst)
            # TODO: solve the inverse analytically?
            transform_mat = np.linalg.inv(N2) @ M @ N1
            transform_mat = transform_mat[:2,:]
            transform_mat = torch.Tensor(transform_mat)
            transform_mats.append(transform_mat)
            aabb_widths.append(width_dst)
        transform_mats = torch.stack(transform_mats)
        return transform_mats, aabb_widths

    def _get_normalize_mat(self, height, width):
        """ Find the transformation matrix that normalizes coordinates in range 
        [-1, 1]. 

        Args:
            height (int): height of feature map
            width (int): width of feature map

        Returns:
            N (ndarray): (3, 3). Transformation matrix.
        """
        N = np.zeros((3, 3))
        N[0, 0] = 2 / width
        N[0, 2] = -1
        N[1, 0] = 2 / height
        N[1, 1] = -1
        N[2, 2] = 1
        return N