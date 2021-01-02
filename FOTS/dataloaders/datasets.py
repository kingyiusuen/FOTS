import os
import re
import yaml
import cv2
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from shapely.geometry import Polygon

from ..utils.preprocessing import rescale_with_padding, img_aug, sort_points_clockwise, generate_rbox

with open("config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

class ICDAR(Dataset):
    def __init__(self, year, train=True):
        self.train = train
        self.year = year
        if self.year == 2013:
            if self.train:
                self.img_dir = config["dataset"]["ICDAR2013"]["train_img"]
                self.gt_dir = config["dataset"]["ICDAR2013"]["train_gt"]
            else:
                self.img_dir = config["dataset"]["ICDAR2013"]["test_img"]
                self.gt_dir = config["dataset"]["ICDAR2013"]["test_gt"]
        elif self.year == 2015:
            if self.train:
                self.img_dir = config["dataset"]["ICDAR2015"]["train_img"]
                self.gt_dir = config["dataset"]["ICDAR2015"]["train_gt"]
            else:
                self.img_dir = config["dataset"]["ICDAR2015"]["test_img"]
                self.gt_dir = config["dataset"]["ICDAR2015"]["test_img"]
        else:
            raise ValueError('Year must be either 2013 or 2015.')
        self.img_filenames = [file.name for file in os.scandir(self.img_dir)]
        self.transform = transforms.Compose([
            # this modifies the shape of img from H x W x C to C x H x W and
            # the range of values into [0.0, 1.0]
            transforms.ToTensor(), 
            # use the same normalization as ResNet because the backbone
            # network is ResNet
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, f'{img_filename}')
        img = cv2.imread(img_path)
        img_filename_prefix = img_filename.split('.')[0] # drop the file extension part
        bboxes, texts = self.load_annotations(img_filename_prefix)
        if not self.train:
            img = self.transform(img)
            return img, bboxes, texts
        img, bboxes, texts = img_aug(img, bboxes, texts)
        score_map, geo_map, angle_map, training_mask = generate_rbox(img, bboxes, texts)
        img = self.transform(img)
        return img_filename, img, bboxes, texts, score_map, geo_map, angle_map, training_mask

    def load_annotations(self, img_filename_prefix):
        gt_filename = f'gt_{img_filename_prefix}.txt'
        gt_path = os.path.join(self.gt_dir, gt_filename)
        bboxes = []
        texts = []
        with open(gt_path, mode='r') as f:
            for line in f:
                line = line.strip('\ufeff\xef\xbb\xbf').strip().split(',') # remove escape sequences
                if self.year == 2013 and not self.train:
                    x, y, w, h = line[:4]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    x1, y1 = x, y
                    x2, y2 = x + w, y
                    x3, y3 = x + w, y + h
                    x4, y4 = x, y + h
                    text = ','.join(line[4:])
                    text = text.strip('"')
                else:
                    x1, y1, x2, y2, x3, y3, x4, y4 = line[:8]
                    text = ','.join(line[8:]) # prevent cases like "$5,000"
                bbox = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
                if not Polygon(bbox).is_valid:
                    continue
                bboxes.append(bbox)
                texts.append(text)
        assert len(bboxes) == len(texts)
        bboxes = np.int0(bboxes)
        texts = np.array(texts)
        return bboxes, texts

class SynthText(Dataset):
    def __init__(self):
        self.img_dir = config["dataset"]["SynthText"]
        gt_path = os.path.join(self.img_dir, "gt.mat")
        gt = loadmat(gt_path, squeeze_me=True, variable_names=['imnames', 'wordBB', 'txt'])
        self.img_filenames = gt['imnames']
        self.all_bboxes = gt['wordBB']
        self.all_texts = gt['txt']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        img = cv2.imread(img_path)
        # for some images the number of bounding boxes does not match the
        # number of words and there is really no way to fix it so just return 
        # a random image
        try:
            bboxes, texts = self.load_annotations(idx)
            height, width, _ = img.shape
            if height < 640 or width < 640:
                img, bboxes, texts = rescale_with_padding(img, bboxes, texts, size=640)
            score_map, geo_map, angle_map, training_mask = generate_rbox(img, bboxes, texts)
            img = self.transform(img)
            return img_filename, img, bboxes, texts, score_map, geo_map, angle_map, training_mask
        except:
            return self.__getitem__(np.random.randint(0, len(self)))

    def load_annotations(self, idx):
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
