from shapely.geometry import Polygon
import numpy as np

def get_iou(bbox1, bbox2):
    """ Find the intersection over union of two bounding boxes. """
    bbox1 = Polygon(bbox1[:8].reshape((4, 2)))
    bbox2 = Polygon(bbox2[:8].reshape((4, 2)))
    if not bbox1.is_valid or not bbox2.is_valid:
        return 0
    inter = bbox1.intersection(bbox2).area
    union = bbox1.area + bbox2.area - inter
    if union == 0:
        return 0
    else:
        return inter / union

# if can't import shapely, replace get_iou with the following:
#
#def rectangle_area(rect):
#    tl, tr, br, _ = rect
#    return np.linalg.norm(tl - tr) * np.linalg.norm(tr - br)
#
#def get_iou(g, p):
#    """ Find the intersection over union of two polygons. """
#    max_x = int(max(np.max(g[0,:8]), np.max(p[0,:8])))
#    max_y = int(max(np.max(g[1,:8]), np.max(p[1,:8])))
#    g = g[:8].reshape((4, 2))
#    p = p[:8].reshape((4, 2))
#    g_mask = np.zeros((max_x, max_y), dtype=np.uint8)
#    p_mask = np.zeros((max_x, max_y), dtype=np.uint8)
#    # the area of intersection is only an approximation 
#    # because we are converting float to int
#    cv2.fillPoly(g_mask, [np.int0(g)], 1)
#    cv2.fillPoly(p_mask, [np.int0(p)], 1)
#    inter = cv2.countNonZero(np.bitwise_and(g_mask, p_mask))
#    union = rectangle_area(g) + rectangle_area(p) - inter
#    if union == 0:
#        return 0
#    return inter / union

def precision_recall_f1(bboxes_pred, bboxes_true, iou_threshold=0.5):
    """
    Args:
        bboxes_pred (list): A list of M elements where M is the number of 
            images. Each element is a numpy array of shape (N_i, 9) where N_i 
            is the number of predicted bounding boxes in image i and the nine 
            elements are the predicted coordinates and confidence score of the 
            bounding boxes in that image. 
        bboxes_true (list): A list of M elements where M is the number of 
            images. Each element is an array of shape (N_i', 8) where N_i' is 
            the number of ground truth bounding boxes in image i and the eight 
            elements are the coordinates of the ground truth bounding boxes in 
            that image. (N_i and N_i' may be different.)
        iou_threshold (float): IoU threshold indicating whether the prediction
            will be considered true positive or false positive.
    """
    true_positives = 0
    false_positives = 0
    num_of_images = len(bboxes_pred)
    for i in range(num_of_images):
        # sort predicted bounding boxes according to the confidence score in 
        # descending order
        bboxes_pred[i] = sorted(bboxes_pred[i], key=lambda x: x[8], reverse=True)
        # for each predicted bounding box, loop over every ground truth bounding
        # box, find the ground truth bounding box that results in the maximum
        # IoU with the current predicted bounding box; if the maximum IoU is
        # higher or equal to than the threshold and that ground truth bounding
        # box has not been paired up with another predicted bounding box, label
        # as true positive; otherwise, false positive
        used = [False] * len(bboxes_true[i])
        for j in range(len(bboxes_pred[i])):
            max_iou = float("-inf")
            for k in range(len(bboxes_true[i])):
                iou = get_iou(bboxes_pred[i][j], bboxes_true[i][k])
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = k
            if max_iou >= iou_threshold and not used[max_iou_idx]:
                true_positives += 1
                used[max_iou_idx] = True
            else:
                false_positives += 1
    num_of_gt_bboxes = sum(len(bboxes_true[i]) for i in range(num_of_images))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / num_of_gt_bboxes
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1