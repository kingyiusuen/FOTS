import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from FOTS.model import SharedConv, Detector
from FOTS.data_loaders import ICDAR2013DataLoader, ICDAR2015DataLoader, SynthTextDataLoader
from FOTS.model.loss import DetectionLoss
from FOTS.utils.bbox import restore_bbox
from FOTS.utils.metrics import true_false_positives
from FOTS.utils.preprocessing import rescale, pad_to_fixed_size, collate_fn
from trainer import Trainer

class FOTSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv = SharedConv()
        self.detector = Detector()
        self.detection_loss = DetectionLoss()

    def forward(self, imgs):
        pass

    def training_step(self, batch):
        _, imgs, _, _, _, *detect_true, training_masks = batch
        detect_pred = self.shared_step(imgs)
        cls_loss, reg_loss, detect_loss = self.detection_loss(*detect_pred, *detect_true, training_masks)
        return {'loss': detect_loss, 
                'log': {'train_cls_loss': cls_loss.item(),
                        'train_reg_loss': reg_loss.item(),
                        'train_detect_loss': detect_loss.item()}}

    def validation_step(self, batch):
        _, imgs, _, _, _, *detect_true, training_masks = batch
        detect_pred = self.shared_step(imgs)
        cls_loss, reg_loss, detect_loss = self.detection_loss(*detect_pred, *detect_true, training_masks)
        return {'log': {'val_cls_loss': cls_loss.item(),
                        'val_reg_loss': reg_loss.item(),
                        'val_detect_loss': detect_loss.item()}}

    def test_step(self, batch):
        img, bboxes_true, _ = batch # not really a batch because we're working with one image at a time
        # if image size is too large, CPU/GPU may run out of memory, so resize it
        _, height, width = img.shape
        if height > 2400 or width > 2400:
            scale = 2400 / max(height, width)
            img, bboxes_true = rescale(img, bboxes_true, scale, scale)
        # due to the network architecture, the height and width of the input
        # image must be a multiple of 32, so pad it
        new_height = height + (32 - height % 32) % 32
        new_width = width + (32 - width % 32) % 32
        img = img.permute(1, 2, 0)
        img = pad_to_fixed_size(img, new_height, new_width)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0) # get an extra dimension which is supposed to represent the batch
        detect_pred = self.shared_step(img)
        bboxes_pred, _ = restore_bbox(*detect_pred)
        return true_false_positives([bboxes_pred], [bboxes_true])

    def shared_step(self, imgs):
        shared_features = self.shared_conv(imgs)
        detect_pred = self.detector(shared_features)
        return detect_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="dataset name for training the model", choices=["ICDAR2013", "ICDAR2015", "SynthText"])
    parser.add_argument("--epoch", type=int, help="number of training epochs", default=10)
    parser.add_argument("--test", help="dataset name for testing the model", choices=["ICDAR2013", "ICDAR2015"])
    parser.add_argument("--predict", help="path to an image or a folder of images on which you want to perform text recognition")
    parser.add_argument("--ckpt", help="path to a checkpoint")
    args = parser.parse_args()

    # create a model instance and load a checkpoint if provided
    model = FOTSModel()
    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    # perform text detection and recognition on images
    if args.predict:
        if not args.ckpt:
            raise RuntimeError("Please specify the path to a checkpoint with --ckpt.")
        model(args.imgs)
    # resume training or re-train from scratch
    elif args.train:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, threshold=0.05, verbose=True)
        if args.ckpt:
            print(f"Resume training using checkpoint {args.ckpt}.")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            trainer = Trainer(model, optimizer, lr_scheduler, num_epochs=args.epoch, **checkpoint['trainer_config'])
        else:
            print("Re-training the model from scratch because no checkpoint is specified.")
            trainer = Trainer(model, optimizer, lr_scheduler, num_epochs=args.epoch)
        if args.train == "ICDAR2013":
            icdar2013 = ICDAR2013DataLoader(batch_size=4, num_workers=1, collate_fn=collate_fn)
            trainer.fit(icdar2013.train_dataloader(), icdar2013.val_dataloader())
        elif args.train == "ICDAR2015":
            icdar2015 = ICDAR2015DataLoader(batch_size=4, num_workers=1, collate_fn=collate_fn)
            trainer.fit(icdar2015.train_dataloader(), icdar2015.val_dataloader())
        elif args.train == "SynthText":
            synthtext = SynthTextDataLoader(batch_size=32, num_workers=1, collate_fn=collate_fn)
            trainer.fit(synthtext.train_dataloader())
    # get evaluation metrics (precision, recall, f1 score) on a dataset
    elif args.test:
        if not args.ckpt:
            raise RuntimeError("Please specify the path to a checkpoint with --ckpt.")
        trainer = Trainer(model)#, **checkpoint['trainer_config'])
        # SynthText cannot be tested because it has no test set
        if args.test == "ICDAR2013":
            icdar2013 = ICDAR2013DataLoader(batch_size=None, num_workers=1)
            trainer.test(icdar2013.test_dataloader())
        elif args.test == "ICDAR2015":
            icdar2015 = ICDAR2015DataLoader(batch_size=None, num_workers=1)
            trainer.test(icdar2015.test_dataloader())
            
    
    

