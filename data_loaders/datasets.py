import os
import re

import cv2
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms

from utils.preprocessing import aug, sort_points_clockwise, generate_rbox

class TotalText(Dataset):
    def __init__(self, data_dir='./', train=True):
        self.data_dir = data_dir
        self.train = train
        dir_name = "Train" if self.train else "Test"
        self.img_dir = os.path.join(self.data_dir, f'datasets/totaltext/{dir_name}')
        self.gt_dir = os.path.join(self.data_dir, f'datasets/totaltext/GT_{dir_name}')
        self.img_filenames = [file.name.split('.')[0] for file in os.scandir(self.img_dir)] # keep the prefix, drop the file extension
        self.transform = transforms.Compose([
            transforms.ToTensor(), # this modifies the shape of img from H x W x C to C x H x W
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ) # from ImageNet
        ])

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, f'{img_filename}.jpg')
        img = cv2.imread(img_path)
        bboxes, texts = self._load_annotations(img_filename)
        img, bboxes, texts = aug(img, bboxes, texts)
        score_map, geo_map, angle_map, training_mask = generate_rbox(img, bboxes, texts)
        img = self.transform(img)
        return img_filename, img, bboxes, texts, score_map, geo_map, angle_map, training_mask
    
    def _load_annotations(self, img_filename):
        gt_filename = f'gt_{img_filename}.mat' if self.train else f'poly_gt_{img_filename}.mat'
        gt_path = os.path.join(self.gt_dir, gt_filename)
        gt = loadmat(gt_path)['gt']
        bboxes = []
        texts = []
        # ground truth format is ['x:', x-coords, 'y:', y-coords, text, orientation]
        for _, x, _, y, text, orientation in gt:
            # remove unnecessary dimensions
            x, y = np.squeeze(x), np.squeeze(y)
            # zip the x and y arrays to get a list of points
            # data type must be np.int32; otherwise, cv2.minAreaRect will raise an error
            original_bbox = np.array(list(zip(x, y)), dtype=np.int32)
            # get the minimum bounding rectangle for the bbox (may be rotated)
            # to make sure that the bounding box is a rectangle, instead of a 
            # parallelogram or any other shape
            min_bounding_rect = cv2.minAreaRect(original_bbox) 
            # get the four corner points of the rectangle
            new_bbox = cv2.boxPoints(min_bounding_rect)
            # sort the points in top left, top right, bottom right, bottom left order
            new_bbox = sort_points_clockwise(new_bbox)
            bboxes.append(new_bbox)
            texts.append(text[0]) # text is an array with one element
        # make sure the number of bboxes is the same as the number of texts
        assert len(bboxes) == len(texts)
        # convert to numpy arrays
        bboxes = np.int0(bboxes)
        texts = np.array(texts)
        return bboxes, texts

class SynthText(Dataset):
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'datasets/SynthText')
        gt_path = os.path.join(self.data_dir, 'datasets/SynthText/gt.mat')
        gt = loadmat(gt_path, squeeze_me=True, variable_names=['imnames', 'wordBB', 'txt'])
        self.img_filenames = gt['imnames']
        self.all_bboxes = gt['wordBB']
        self.all_texts = gt['txt']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ) # from ImageNet
        ])

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        img = cv2.imread(img_path)
        bboxes, texts = self._load_annotations(idx)
        img, bboxes, texts = aug(img, bboxes, texts)
        score_map, geo_map, angle_map, training_mask = generate_rbox(img, bboxes, texts)
        img = self.transform(img)
        return img_filename, img, bboxes, texts, score_map, geo_map, angle_map, training_mask

    def _load_annotations(self, idx):
        # the format of the ground truth annotation file can be found in
        # https://www.robots.ox.ac.uk/~vgg/data/scenetext/readme.txt
        texts = [text for texts in self.all_texts[idx] for text in re.split('\n| ', texts.strip()) if text]
        bboxes = self.all_bboxes[idx]
        # zip x and y to get to a list of points
        bboxes = [list(zip(bboxes[0,:,i], bboxes[1,:,i])) for i in range(bboxes.shape[2])]
        # make sure the number of bboxes is the same as the number of texts
        assert len(bboxes) == len(texts)
        # convert to numpy arrays
        bboxes = np.int0(bboxes)
        texts = np.array(texts)
        return bboxes, texts
