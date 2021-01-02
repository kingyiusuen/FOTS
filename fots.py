import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from FOTS.model import FOTSModel
from FOTS.dataloaders import ICDAR, ICDAR2013DataLoaderFactory, ICDAR2015DataLoaderFactory, SynthTextDataLoaderFactory
from FOTS.utils.preprocessing import fit_img_to_net_arch
from FOTS.utils.bbox import restore_bbox
from FOTS.utils.metrics import true_false_positives, precision_recall_f1
from trainer import Trainer

def test(model, test_dataset):
    true_positives = 0
    false_positives = 0
    num_of_gt_bboxes = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, bboxes_true, _ in tqdm(test_dataset, desc="Testing"):
            img, bboxes_true = fit_img_to_net_arch(img, bboxes_true)
            img = torch.as_tensor(img, dtype=torch.float)
            img.to(device)
            detect_pred = model(img)
            bboxes_pred, _ = restore_bbox(*detect_pred)
            test_outputs = true_false_positives([bboxes_pred], [bboxes_true])
            true_positives += test_outputs['true_positives']
            false_positives += test_outputs['false_positives']
            num_of_gt_bboxes += test_outputs['num_of_gt_bboxes']
    return precision_recall_f1(true_positives, false_positives, num_of_gt_bboxes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="dataset name for training the model", choices=["ICDAR2013", "ICDAR2015", "SynthText"])
    parser.add_argument("--epoch", type=int, help="number of training epochs", default=10)
    parser.add_argument("--test", help="dataset name for testing the model", choices=["ICDAR2013", "ICDAR2015"])
    parser.add_argument("--predict", help="path to an image or a folder of images on which you want to perform text recognition")
    parser.add_argument("--ckpt", help="path to a checkpoint")
    args = parser.parse_args()

    if args.train + args.test + args.predict != 1:
        raise RuntimeError("Use one and only one of --train, --test or --predict.")

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
        raise NotImplementedError
    # resume training or train from scratch
    elif args.train:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        if args.ckpt:
            print(f"Resume training using checkpoint {args.ckpt}.")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            trainer = Trainer(model, optimizer, lr_scheduler, num_epochs=args.epoch, **checkpoint['trainer_config'])
        else:
            print("Training the model from scratch because no checkpoint is provided.")
            trainer = Trainer(model, optimizer, lr_scheduler, num_epochs=args.epoch)
        if args.train == "ICDAR2013":
            dataloader_factory = ICDAR2013DataLoaderFactory(val_ratio=0.2)
        elif args.train == "ICDAR2015":
            dataloader_factory = ICDAR2015DataLoaderFactory(val_ratio=0.2)
        elif args.train == "SynthText":
            dataloader_factory = SynthTextDataLoaderFactory(val_ratio=0.1)
        train_dataloader, val_dataloader = dataloader_factory.get_dataloaders(batch_size=4, num_workers=1)
        trainer.train(train_dataloader, val_dataloader)
    # get evaluation metrics (precision, recall, f1 score) on a test set
    # SynthText cannot be tested because it has no test set
    elif args.test:
        if not args.ckpt:
            raise RuntimeError("Please specify the path to a checkpoint with --ckpt.")
        if args.test == "ICDAR2013":
            test_dataset = ICDAR(year=2013, train=False)
        elif args.test == "ICDAR2015":
            test_dataset = ICDAR(year=2015, train=False)
        test(model, test_dataset)