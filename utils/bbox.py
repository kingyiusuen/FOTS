# Helper functions for handling bounding boxes during testing

import torch
import numpy as np
import cv2

from metrics import get_iou

def restore_bbox(score_maps, geo_maps, angle_maps, **kwargs):
    """ Restore bounding boxes in images given the confidence scores, distances 
    between each side and rotation angles.

    Args:
        score_maps (ndarray): (batch_size, 1, H, W).
        geo_maps (ndarray): (batch_size, 4, H, W).
        angle_maps (ndarray): (batch_size, 1, H, W).
    
    Returns:
        bboxes (ndarray).
        bbox_to_img_idx (ndarray).
    """
    # have to convert to numpy array otherwise some of the code will break
    score_maps = score_maps.detach().cpu().numpy()
    geo_maps = geo_maps.detach().cpu().numpy()
    geo_maps = geo_maps.transpose((0, 2, 3, 1)) # now geo_map is batch_size x H x W x 4
    angle_maps = angle_maps.detach().cpu().numpy()

    bboxes = []
    bbox_to_img_idx = []
    num_of_images = score_maps.shape[0]
    for i in range(num_of_images):
        # select the score map, geometry map and angle map for image i
        score_map = score_maps[i,0,:,:]
        geo_map = geo_maps[i,:,:,:]
        angle_map = angle_maps[i,0,:,:]
        # recover the bounding boxes for image i
        bboxes_i = restore_bbox_helper(score_map, geo_map, angle_map, **kwargs)
        if len(bboxes_i) == 0: # if no bounding box in image i
            continue
        bboxes.append(bboxes_i)
        bbox_to_img_idx.append([i] * len(bboxes_i))
    bboxes = np.concatenate(bboxes)
    bbox_to_img_idx = np.concatenate(bbox_to_img_idx)
    return bboxes, bbox_to_img_idx

def restore_bbox_helper(score_map, geo_map, angle_map, 
    score_map_threshold=0.5, nms_threshold=0.3, include_score=False):
    """ Restore the bounding boxes in a single image and then apply NMS.

    Args:
        score_map (ndarray): (H, W).
        geo_map (ndarray): (H, W, 4).
        angle_map (ndarray): (H, W).
        score_map_threshold (float): Threshold for initial filtering.
        nms_threshold (float): IoU threshold for surviving in NMS.
        include_score (bool): If True, bboxes will be (N, 9) where first 8
            are coordinates and last one is the confidence score. If False, 
            bboxes will be (N, 4, 2).

    Returns:
        bboxes (ndarray): (N, 4, 2) where N is the number of bounding
            boxes left after NMS.
    
    References:
        https://github.com/jiangxiluning/FOTS.PyTorch/blob/master/FOTS/utils/bbox.py#L166
    """
    # filter out areas that have low confidence
    origins = np.argwhere(score_map >= score_map_threshold)
    # sort along the y axis
    origins = origins[np.argsort(origins[:,0])]
    bboxes = rbox_to_bbox(
        origins[:,::-1] * 4, # have to reverse so that x comes first before y
        geo_map[origins[:,0], origins[:,1], :],
        angle_map[origins[:,0], origins[:,1]]
    ) # N x 4 x 2
    data = np.hstack([
        bboxes.reshape(-1, 8), # N x 8
        score_map[origins[:,0], origins[:,1]][:,np.newaxis], # N x 1
    ]) # N x 9
    bboxes = nms_locality(data, nms_threshold)
    if bboxes.shape[0] == 0:
        return np.array([])
    if include_score:
        return bboxes # N x 9
    bboxes = bboxes[:,:8] # drop the score map column -> N x 8
    return bboxes.reshape((-1, 4, 2)) # N x 4 x 2

def rbox_to_bbox(origins, geo_map, angle_map):
    """ Find the bounding boxes of the origins given the distances between 
    origin and the four sides of their bounding boxes, as well as the angles of 
    rotation.

    Args:
        origins (ndarray): (N, 2). 
        geo_map (ndarray): (N, 4).
        angle_map (ndarray): (N).

    Returns:
        bboxes (ndarray): (N, 4, 2). Bounding boxes of the origins.

    References:
        https://github.com/argman/EAST/blob/master/icdar.py#L387
    """
    num_of_points = len(origins)
    distances_to_top = geo_map[:,0]
    distances_to_right = geo_map[:,1]
    distances_to_bottom = geo_map[:,2]
    distances_to_left = geo_map[:,3]
    heights = distances_to_top + distances_to_bottom
    widths = distances_to_right + distances_to_left
    zeros = np.zeros(num_of_points)
    # For each origin, derive five points. The first four represent the
    # coordinates of the top-left, top-right, bottom-right and bottom-left of
    # the bounding boxes. The fifth one is where the origin would be if the 
    # bounding box is unrotated and starts at (0, 0). We need it for translating 
    # the bounding box to the desired location.
    points = np.array([
        [zeros.copy(), zeros.copy()],
        [widths, zeros.copy()],
        [widths, -heights],
        [zeros.copy(), -heights],
        [distances_to_left, -distances_to_top]
    ]).transpose((2, 1, 0)) # 4 x 2 x N -> N x 2 x 4
    # rotate those five points
    rotation_mat = np.array([
        np.cos(angle_map), np.sin(angle_map),
        np.sin(angle_map), -np.cos(angle_map)
    ]).transpose().reshape((-1, 2, 2)) # 4 x N -> N x 4 -> N x 2 x 2
    rotated_points = (rotation_mat @ points).transpose((0, 2, 1)) # N x 2 x 4 -> N x 4 x 2
    offset = (origins - rotated_points[:,4,:]).reshape((-1, 1, 2)) # N x 2 -> N x 1 x 2
    bboxes = rotated_points[:,:4,:] + offset # N x 4 x 2
    return bboxes

# Non-Maximum Suppression
# Reference: https://github.com/argman/EAST/blob/master/locality_aware_nms.py

def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g

def standard_nms(S, thres):
    # sort the confidence scores in descending order
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute the IoU of the current proposal with every other proposal
        ovr = np.array([get_iou(S[i], S[t]) for t in order[1:]])
        # keep the proposals if their IoU is smaller than the threshold
        inds = np.where(ovr <= thres)[0]
        order = order[inds+1] # add 1 because the first one is i, which has already been selected
    return S[keep]

def nms_locality(polys, thres=0.3):
    """ Locality-Aware NMS, as described in the EAST paper

    Args:
        polys (ndarray): (N, 9). First 8 are coordinates, last one is probability.
    
    Returns:
        (ndarray): (N, 9). Bounding boxes after NMS.
    """
    S = []
    p = None
    for g in polys:
        if p is not None and get_iou(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)
