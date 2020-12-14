# Helper functions for preparing the input data for the model

import cv2
import numpy as np
import torch

def rescale(img, bboxes, scale_x, scale_y):
    """ Rescale the input image and bounding boxes.

    Args:
        img (ndarray): (H, W, 3). Input image.
        bboxes (ndarray): (N, 4, 2) where N is the number of bounding boxes in 
            img. X and y coordinates of the corner points of the bounding 
            boxes.
        scale_x (float): Scale factor along the horizontal axis.
        scale_y (float): Scale factor along the vertical axis.
    
    Returns:
        img (ndarray): (H', W', 3) where H' and W' are the rescaled height and 
            width. Rescaled input image.
        bboxes (ndarray): (N, 4, 2). X and y coordinates of the corner 
            points of the rescaled bounding boxes.
    """
    img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
    bboxes[:,:,0] = bboxes[:,:,0] * scale_x # rescale width
    bboxes[:,:,1] = bboxes[:,:,1] * scale_y # rescale height
    return img, bboxes

def rotate(img, bboxes, angle):
    """ Rotate the input image and the bounding boxes without cropping.

    Args:
        img (ndarray): (H, W, 3). Input image.
        bboxes (ndarray): (N, 4, 2) where N is the number of bounding boxes in 
            img. X and y coordinates of the corner points of the bounding 
            boxes.
        angle (float): Angle of rotation. Rotate anti-clockwise if positive.
            Rotate clockwise if negative.
    
    Returns:
        img (ndarray): (H', W', 3) where H' and W' are the new height and 
            width after padding. Rotated input image.
        bboxes (ndarray): (N, 4, 2). X and y coordinates of the corner 
            points of the rotated bounding boxes.

    Reference: 
        https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c/37347070#37347070
    """
    # grab the dimensions of the image and then determine the center
    height, width = img.shape[0], img.shape[1]
    center_x, center_y = width / 2, height / 2
    # grab the rotation matrix
    rotation_mat = cv2.getRotationMatrix2D((center_x, center_y), angle=angle, scale=1.0)
    abs_cos = np.abs(rotation_mat[0, 0])
    abs_sin = np.abs(rotation_mat[0, 1])
    # compute the new bounding dimensions of the image
    new_width = int((height * abs_sin) + (width * abs_cos))
    new_height = int((height * abs_cos) + (width * abs_sin))
    # adjust the rotation matrix to take translation into account
    rotation_mat[0, 2] += (new_width / 2) - center_x
    rotation_mat[1, 2] += (new_height / 2) - center_y
    # perform the rotation
    img = cv2.warpAffine(img, rotation_mat, (new_width, new_height))
    bboxes = cv2.transform(bboxes, rotation_mat)
    return img, bboxes

def random_crop(img, bboxes, texts, crop_size=640, max_attempts=100):
    """ Make a random crop from the input image. Will return the original inputs
    if can't a find a way to crop.

    TODO: Pad the image if the image size is smaller than the crop area?

    Args:
        img (ndarray): (H, W, 3). Input image.
        bboxes (ndarray): (N, 4, 2) where N is the number of bounding boxes in 
            img. X and y coordinates of the corner points of the bounding 
            boxes.
        texts (ndarray): (N). Text labels.
        crop_size (int): Size of the cropped image.
        max_attempts (int): Number of cropping attempts before returning the
            original inputs.

    Returns:
        img (ndarray): (crop_size, crop_size, 3). Cropped image.
        bboxes (ndarray): (N', 4, 2) where N is the number of bounding boxes 
            remaining after cropping. X and y coordinates of the corner points of 
            those bounding boxes.
        texts (ndarray): (N'). Text labels remaining after cropping.

    Reference: 
        https://github.com/Pay20Y/FOTS_TF/blob/dev/data_provider/data_utils.py#L183 
    """
    height, width = img.shape[0], img.shape[1]
    if height < crop_size or width < crop_size:
        raise RuntimeError("Fail to crop")
    # for checking which x and y coordinates have already been filled with 
    # a bounding box
    filled_xs = np.zeros(width, dtype=np.int32)
    filled_ys = np.zeros(height, dtype=np.int32)
    # make sure we won't crop out of bound
    filled_xs[(width-crop_size):width] = 1   
    filled_ys[(height-crop_size):height] = 1
    for bbox in bboxes:
        bbox_min_x = np.min(bbox[:,0])
        bbox_max_x = np.max(bbox[:,0])
        bbox_min_y = np.min(bbox[:,1])
        bbox_max_y = np.max(bbox[:,1])
        filled_xs[bbox_min_x:bbox_max_x] = 1
        filled_ys[bbox_min_y:bbox_max_y] = 1

    # store which x and y coordinates have no text
    empty_xs = np.squeeze(np.where(filled_xs == 0))
    empty_ys = np.squeeze(np.where(filled_ys == 0))

    fail = True
    for _ in range(max_attempts):
        crop_area_min_x = np.random.choice(empty_xs)
        crop_area_max_x = crop_area_min_x + crop_size
        crop_area_min_y = np.random.choice(empty_ys)
        crop_area_max_y = crop_area_min_y + crop_size
        # lambda functions to check whether a point is inside/outside the crop area
        is_point_inside = lambda point: crop_area_min_x <= point[0] <= crop_area_max_x and \
                                        crop_area_min_y <= point[1] <= crop_area_max_y
        is_point_outside = lambda point: not (crop_area_min_x <= point[0] <= crop_area_max_x) or \
                                         not (crop_area_min_y <= point[1] <= crop_area_max_y)
        # lambda functions to check whether a bounding box is completely inside/outside the crop area
        is_bbox_inside = lambda bbox: all(is_point_inside(point) for point in bbox)
        is_bbox_outside = lambda bbox: all(is_point_outside(point) for point in bbox)
        # indexs of bounding boxes that are completely inside/outside the crop area
        bboxes_inside_crop_area = [i for i, bbox in enumerate(bboxes) if is_bbox_inside(bbox)]
        bboxes_outside_crop_area = [i for i, bbox in enumerate(bboxes) if is_bbox_outside(bbox)]
        # try again if some bounding box is partially clipped
        if len(bboxes_inside_crop_area) + len(bboxes_outside_crop_area) != len(bboxes):
            continue
        # try again if there is no text in the cropped area
        if len(bboxes_inside_crop_area) == 0:
            continue
        if len(text for text in texts[bboxes_inside_crop_area] if text != "#") == 0:
            continue
        # crop
        img = img[crop_area_min_y:crop_area_max_y, crop_area_min_x:crop_area_max_x, :]
        # select the bounding boxes and texts that are inside the cropped area
        bboxes = bboxes[bboxes_inside_crop_area]
        texts = texts[bboxes_inside_crop_area]
        # adjust the coordinates of the bounding boxes after cropping
        bboxes[:,:,0] -= crop_area_min_x
        bboxes[:,:,1] -= crop_area_min_y
        return img, bboxes, texts
    raise RuntimeError("Fail to crop")

def aug(img, bboxes, texts, crop_size=640, max_attempts=100):
    """ Perform image augmentation as described in the FOTS paper.

    Args:
        img (ndarray): (H, W, 3). Input image.
        bboxes (ndarray): (N, 4, 2) where N is the number of bounding boxes in 
            img. X and y coordinates of the corner points of the bounding 
            boxes.
        texts (ndarray): (N). Text labels.
        crop_size (int): Size of the cropped image.
        max_attempts (int): Number of cropping attempts.

    Returns:
        img (ndarray): (H', W', 3) where H' and W' are the new height and width
            after augmentation. Augmented input image.
        bboxes (ndarray): (N', 4, 2) where N' is the number of remaining 
            bounding box after cropping. X and y coordinates of the corner 
            points of the remaining bounding boxes.
        texts (ndarray): (N'). Text labels remaining after cropping. 

    """
    # resize the longer side of the image to 2560 pixels while maintaining the aspect ratio
    scale = 2560 / np.maximum(img.shape[0], img.shape[1])
    img, bboxes = rescale(img, bboxes, scale_x=scale, scale_y=scale)

    # rotate [-10, 10] degree
    angle = np.random.uniform(-10, 10)
    img, bboxes = rotate(img, bboxes, angle)

    # rescale the height of the image with ratio from 0.8 to 1.2 while keeping width unchanged
    scale = np.random.uniform(0.8, 1.2)
    img, bboxes = rescale(img, bboxes, scale_x=1, scale_y=scale)

    # random cropping
    img, bboxes, texts = random_crop(img, bboxes, texts, crop_size=crop_size)
    return img, bboxes, texts

def shrink_polygon(polygon, R=0.3):
    """ Shrink a polygon, as described in the EAST paper.
    
    Args:
        polygon (ndarray): (4, 2). X and y coordinates of the corner points of 
            a quadrangle, sorted in clockwise order.
        R (int): shrink ratio

    Returns:
        polygon (ndarray): (4, 2). X and y coordinates of the corner points of 
            a shrunk quadrangle, sorted in clockwise order.

    Reference: 
        https://github.com/jiangxiluning/FOTS.PyTorch/blob/master/FOTS/data_loader/datautils.py#L161
    """
    r = [None, None, None, None]
    for i in range(4):
        r[i] = min(np.linalg.norm(polygon[i] - polygon[(i + 1) % 4]),
                   np.linalg.norm(polygon[i] - polygon[(i - 1) % 4]))
    # first shrink the two longer edges of a quadrangle, and then the two shorter ones
    # find the longer pair
    if np.linalg.norm(polygon[0] - polygon[1]) + np.linalg.norm(polygon[2] - polygon[3]) > \
            np.linalg.norm(polygon[0] - polygon[3]) + np.linalg.norm(polygon[1] - polygon[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((polygon[1][1] - polygon[0][1]), (polygon[1][0] - polygon[0][0]))
        polygon[0][0] += R * r[0] * np.cos(theta)
        polygon[0][1] += R * r[0] * np.sin(theta)
        polygon[1][0] -= R * r[1] * np.cos(theta)
        polygon[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((polygon[2][1] - polygon[3][1]), (polygon[2][0] - polygon[3][0]))
        polygon[3][0] += R * r[3] * np.cos(theta)
        polygon[3][1] += R * r[3] * np.sin(theta)
        polygon[2][0] -= R * r[2] * np.cos(theta)
        polygon[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((polygon[3][0] - polygon[0][0]), (polygon[3][1] - polygon[0][1]))
        polygon[0][0] += R * r[0] * np.sin(theta)
        polygon[0][1] += R * r[0] * np.cos(theta)
        polygon[3][0] -= R * r[3] * np.sin(theta)
        polygon[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((polygon[2][0] - polygon[1][0]), (polygon[2][1] - polygon[1][1]))
        polygon[1][0] += R * r[1] * np.sin(theta)
        polygon[1][1] += R * r[1] * np.cos(theta)
        polygon[2][0] -= R * r[2] * np.sin(theta)
        polygon[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        theta = np.arctan2((polygon[3][0] - polygon[0][0]), (polygon[3][1] - polygon[0][1]))
        polygon[0][0] += R * r[0] * np.sin(theta)
        polygon[0][1] += R * r[0] * np.cos(theta)
        polygon[3][0] -= R * r[3] * np.sin(theta)
        polygon[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((polygon[2][0] - polygon[1][0]), (polygon[2][1] - polygon[1][1]))
        polygon[1][0] += R * r[1] * np.sin(theta)
        polygon[1][1] += R * r[1] * np.cos(theta)
        polygon[2][0] -= R * r[2] * np.sin(theta)
        polygon[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((polygon[1][1] - polygon[0][1]), (polygon[1][0] - polygon[0][0]))
        polygon[0][0] += R * r[0] * np.cos(theta)
        polygon[0][1] += R * r[0] * np.sin(theta)
        polygon[1][0] -= R * r[1] * np.cos(theta)
        polygon[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((polygon[2][1] - polygon[3][1]), (polygon[2][0] - polygon[3][0]))
        polygon[3][0] += R * r[3] * np.cos(theta)
        polygon[3][1] += R * r[3] * np.sin(theta)
        polygon[2][0] -= R * r[2] * np.cos(theta)
        polygon[2][1] -= R * r[2] * np.sin(theta)
    return polygon

def sort_points_clockwise(bbox):
    """ Sort the corner points of a bounding box clockwise (in top-left,
    top-right, bottom-right and bottom-left order).

    Args:
        bbox (ndarray): (4, 2). X and y coordinates of the corner points of
            a bounding box.

    Returns:
        bbox (ndarray): (4, 2). X and y coordinates of the corner points of
            the input bounding box in clockwise order.

    Examples:                                                                              
                                          tl            
                    tr                 ---\             
                ----\           bl  -/    --\          
    tl     -----/     |            -\         -\        
        ---/           \              --\        --\     
        \            ---|                -\        -  tr
        \     -----/    br                --\    -/     
        ----/                                --/       
        bl                                     br       
    
    References: 
        https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    """
	# sort the points based on their x-coordinates
    x_sorted = bbox[np.argsort(bbox[:,0]),:]
	# grab the left-most two points and right-most two points
    left_most = x_sorted[:2,:]
    right_most = x_sorted[2:,:]
	# sort the left-most points according to the y-coordinates 
    # so we can grab the top-left and bottom-left points respectively
    left_most = left_most[np.argsort(left_most[:,1]),:]
    tl, bl = left_most
	# do the same for the right-most two points
    right_most = right_most[np.argsort(right_most[:,1]),:]
    tr, br = right_most
    return np.array([tl, tr, br, bl], dtype=np.int32)

def find_rotation_angle(bbox):
    """ Find the angle of rotation (in radian) of a bounding box. 
        
    Args:
        bbox (ndarray): (4, 2). X and y coordinates of the corners points 
            of the bounding boxes, sorted in clockwise order.

    Returns:
        angle (float): Angle of rotation. Positive if we have to rotate an
            axis-aligned bounding box anti-clockwise to obtain the input 
            bounding box. Negative otherwise.
    """
    _, _, br, bl = bbox 
    # If y-coord of top-left is the same as y-coord of top-right,
    # then the rectangle is not rotated
    if bl[1] == br[1]: 
        return 0
    # find the angle between the bottom side and the x-axis using the formula
    # cos(angle) = adjacent / hypotenuse
    cos_angle = (br[0] - bl[0]) / np.linalg.norm(bl - br)
    angle = np.arccos(cos_angle)
    if bl[1] < br[1]: # bottom right is higher than bottom left
        return angle
    return -angle

def distance_from_point_to_line(point, line_p1, line_p2):
    """ Compute the distance between a point and a line. 

    Args:
        point (ndarray): (2). X and y coordinates of the point.
        line_p1 (ndarray): (2). X and y coordinates of the starting point
            of a line.
        line_p2 (ndarray): (2). X and y coordinates of the ending point of
            a line.

    Args:
        distance (int): The distance between the point and the line.

    References:
        https://github.com/jiangxiluning/FOTS.PyTorch/blob/master/FOTS/data_loader/datautils.py#L228
    """
    norm_of_cross_product = np.linalg.norm(np.cross(line_p2 - line_p1, line_p2 - point))
    norm_of_line = np.linalg.norm(line_p2 - line_p1)
    distance = norm_of_cross_product / norm_of_line
    return distance

def generate_rbox(img, bboxes, texts):
    """ Returns data that will used in the detector.

    Args:
        img (ndarray): (H, W, C). Input image.
        bboxes (ndarray): (N, 4, 2) where N is the number of bounding boxes in 
            img. X and y coordinates of the corner points of the bounding 
            boxes.
        texts (ndarray): (N).

    Returns:
        score_map (ndarray): (1, H, W). Probabilities of each pixel being a 
            positive sample.
        geo_map (ndarray): (4, H, W). Distances between each pixel location and 
            the top, right, bottom and left sides of a bounding box respectively.
        angle_map (ndarray): (1, H, W). Angles of rotation. Positive values 
            represent an anti-clockwise rotation. Negative values represent a 
            clockwise rotation.
        training_mask (ndarray): (1, H, W). A binary matrix that indicate 
            whether each pixel should contribute to the classification loss.
    """

    height, width = img.shape[0], img.shape[1]
    score_map = np.zeros((1, height, width))
    geo_map = np.zeros((4, height, width),)
    angle_map = np.zeros((1, height, width))
    training_mask = np.ones((1, height, width))
    for bbox, text in zip(bboxes, texts):
        tl, tr, br, bl = bbox

        # shrink the bounding box by a factor of 0.3
        shrunk_bbox = shrink_polygon(bbox.astype(np.float32)).astype(int)
        shrunk_bbox_mask = np.zeros((height, width))
        cv2.fillPoly(shrunk_bbox_mask, [shrunk_bbox], 1)

        # for each point inside the shrunk bounding box, fill the score map with 1
        cv2.fillPoly(score_map, [shrunk_bbox], 1)

        # ignore the area between a bounding box and its shrunk version 
        cv2.fillPoly(training_mask, [bbox], 0)
        cv2.fillPoly(training_mask, [shrunk_bbox], 1)

        # ignore if the bounding box is too small
        len1 = np.linalg.norm(tl - tr)
        len2 = np.linalg.norm(tr - br)
        if len1 < 10 or len2 < 10:
            cv2.fillPoly(training_mask, [bbox], 0)

        # ignore if the text is 'Don't care'
        if text == "#":
            cv2.fillPoly(training_mask, [bbox], 0)

        # find the rotation angle
        angle = find_rotation_angle(bbox)
        
        # for each point inside the shrunk bounding box, compute its distance to
        for y, x in np.argwhere(shrunk_bbox_mask == 1):
            point = np.array([x, y])
            geo_map[0, y, x] = distance_from_point_to_line(point, tl, tr) # top
            geo_map[1, y, x] = distance_from_point_to_line(point, tr, br) # right
            geo_map[2, y, x] = distance_from_point_to_line(point, br, bl) # bottom
            geo_map[3, y, x] = distance_from_point_to_line(point, bl, tl) # left
            angle_map[0, y, x] = angle

    # downsample by a factor of 4 because the output of the shared convolutions
    # is 1/4 of the original image size 
    # also expand the dimension of some outputs so that all outputs have the 
    # same number of channel
    score_map = score_map[:,::4,::4]
    geo_map = geo_map[:,::4,::4]
    angle_map = angle_map[:,::4,::4]
    training_mask = training_mask[:,::4,::4]
    return score_map, geo_map, angle_map, training_mask

def tuple_of_numpy_arrays_to_tensor(xs):
    return torch.stack([torch.from_numpy(x) for x in xs])

def flatten_tuple_of_numpy_arrays(xs):
    return np.concatenate([x.flatten() for x in xs])

def collate_fn(batch):
    """ Process a list of samples to form a batch. Use in dataloader. """
    img_filenames, imgs, bboxes, texts, score_maps, geo_maps, angle_maps, training_masks = zip(*batch)
    # the img_idx here is the index in the batch, different from the idx in datasets.py
    bbox_to_img_idx = [i for i, bboxes_i in enumerate(bboxes) for _ in bboxes_i]
    imgs = torch.stack(imgs)
    score_maps = tuple_of_numpy_arrays_to_tensor(score_maps)
    geo_maps = tuple_of_numpy_arrays_to_tensor(geo_maps)
    angle_maps = tuple_of_numpy_arrays_to_tensor(angle_maps)
    training_masks = tuple_of_numpy_arrays_to_tensor(training_masks)
    bboxes = np.concatenate(bboxes)
    texts = flatten_tuple_of_numpy_arrays(texts)
    return img_filenames, imgs, bboxes, texts, score_maps, geo_maps, angle_maps, training_masks, bbox_to_img_idx