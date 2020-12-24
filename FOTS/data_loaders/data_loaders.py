import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from .datasets import ICDAR, SynthText

class ICDAR2013DataLoader:
    # train: 183 images
    # val: 46 images
    # test: 223 images
    def __init__(self, **kwargs):
        super().__init__()
        torch.manual_seed(0) # ensure getting the same train/val split next time we resume training
        self.kwargs = kwargs
        icdar2013_full = ICDAR(year=2013, train=True)
        n = len(icdar2013_full)
        lengths = [round(n * 0.8), round(n * 0.2)]
        self.icdar2013_train, self.icdar2013_val = random_split(icdar2013_full, lengths)
        self.icdar2013_test = ICDAR(year=2013, train=False)

    def train_dataloader(self):
        return DataLoader(self.icdar2013_train, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.icdar2013_val, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.icdar2013_test, **self.kwargs)

class ICDAR2015DataLoader:
    # train: 800 images
    # val: 200 images
    # test: 500 images
    def __init__(self, **kwargs):
        super().__init__()
        torch.manual_seed(0)
        self.kwargs = kwargs
        icdar2015_full = ICDAR(year=2015, train=True)
        n = len(icdar2015_full)
        lengths = [round(n * 0.8), round(n * 0.2)]
        self.icdar2015_train, self.icdar2015_val = random_split(icdar2015_full, lengths)
        self.icdar2015_test = ICDAR(year=2015, train=False)

    def train_dataloader(self):
        return DataLoader(self.icdar2015_train, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.icdar2015_val, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.icdar2015_test, **self.kwargs)

class SynthTextDataLoader:
    # 858,750 images
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def train_dataloader(self):
        synth_text = SynthText()
        return DataLoader(synth_text, **self.kwargs)
