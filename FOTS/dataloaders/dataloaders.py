import torch
from torch.utils.data import random_split, DataLoader

from .datasets import ICDAR, SynthText
from ..utils.preprocessing import collate_fn

class BaseDataLoaderFactory:
    def __init__(self, train_dataset, val_ratio=0.2, seed=0):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_ratio = val_ratio
        self.seed = seed

    def get_dataloaders(self, **kwargs):
        torch.manual_seed(self.seed) # ensure getting the same train/val split next time we resume training
        n = len(self.train_dataset)
        lengths = [round(n * (1 - self.val_ratio)), round(n * self.val_ratio)]
        train_dataset, val_dataset = random_split(self.train_dataset, lengths)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, **kwargs)
        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, **kwargs)
        return train_dataloader, val_dataloader

class ICDAR2013DataLoaderFactory(BaseDataLoaderFactory):
    def __init__(self, *arg, **kwargs):
        train_dataset = ICDAR(year=2013, train=True) # 229 images
        BaseDataLoaderFactory.__init__(self, train_dataset, **kwargs)

class ICDAR2015DataLoaderFactory(BaseDataLoaderFactory):
    def __init__(self, *arg, **kwargs):
        train_dataset = ICDAR(year=2015, train=True) # 1,000 images
        BaseDataLoaderFactory.__init__(self, train_dataset, **kwargs)

class SynthTextDataLoaderFactory(BaseDataLoaderFactory):
    def __init__(self, *arg, **kwargs):
        train_dataset = SynthText() # 858,750 images
        BaseDataLoaderFactory.__init__(self, train_dataset, **kwargs)
