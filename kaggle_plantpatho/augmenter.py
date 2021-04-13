"""Module to perform efficient preprocess and data augmentation."""

import torch
import torch.nn as nn

import kornia as K
import numpy as np


# Define the augmentations pipeline

DATA_MEAN = [0.485, 0.456, 0.406]
DATA_STD = [0.229, 0.224, 0.225]


class Resize(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
    
    def forward(self, x):
        return K.geometry.resize(x[None], self.size)[0]


class LitPreprocess(nn.Module):
    """Applies the processing to the image in the worker before collate."""
    def __init__(self, size):
        super().__init__()
        self.preprocess = nn.Sequential(
            #K.augmentation.RandomResizedCrop((224, 224)),
            Resize((size, size)),  # use this better to see whole image
            K.augmentation.Normalize(
                torch.tensor(DATA_MEAN),
                torch.tensor(DATA_STD),
            ),
        )
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = K.utils.image_to_tensor(np.array(x)).float()/255.
        assert len(x.shape) == 3, x.shape
        out = self.preprocess(x)
        return out[0]


class LitAugmenter(nn.Module):
    """Applies random augmentation to a batch of images."""
    def __init__(self, viz: bool = False):
        super().__init__()
        self.viz = viz
        '''self.geometric = [
            K.augmentation.RandomAffine(60., p=0.75),
        ]'''
        self.augmentations = nn.Sequential(
            K.augmentation.RandomRotation(degrees=30.),
            K.augmentation.RandomPerspective(distortion_scale=0.4),
            K.augmentation.RandomResizedCrop((224, 224)),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            #K.augmentation.GaussianBlur((3, 3), (0.1, 2.0), p=1.0),
            #K.augmentation.ColorJitter(0.01, 0.01, 0.01, 0.01, p=0.25),
        )
        self.denorm = K.augmentation.Denormalize(
            torch.tensor(DATA_MEAN),
            torch.tensor(DATA_STD),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4, x.shape
        out = x
        #idx = torch.randperm(len(self.geometric))[0]  # OneOf
        #out = self.geometric[idx](x)
        out = self.augmentations(out)
        if self.viz:
            out = self.denorm(out)
        return out
