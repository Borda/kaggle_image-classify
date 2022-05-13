"""Module to perform efficient preprocess and data augmentation."""

import numpy as np
import torch
import torch.nn as nn
from kornia import augmentation, geometry, image_to_tensor

# Define the augmentations pipeline
from PIL import Image
from torch import Tensor
from torchvision import transforms as T

from kaggle_imgclassif import DATASET_IMAGE_MEAN, DATASET_IMAGE_STD

#: default training augmentation
TORCHVISION_TRAIN_TRANSFORM = T.Compose(
    [
        T.Resize(size=512, interpolation=Image.BILINEAR),
        T.RandomRotation(degrees=30),
        T.RandomPerspective(distortion_scale=0.4),
        T.RandomResizedCrop(size=224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD),  # custom
    ]
)
#: default validation augmentation
TORCHVISION_VALID_TRANSFORM = T.Compose(
    [
        T.Resize(size=256, interpolation=Image.BILINEAR),
        T.CenterCrop(size=224),
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD),  # custom
    ]
)


class Resize(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        return geometry.resize(x[None], self.size)[0]


class LitPreprocess(nn.Module):
    """Applies the processing to the image in the worker before collate."""

    def __init__(self, img_size: int):
        super().__init__()
        self.preprocess = nn.Sequential(
            # K.augmentation.RandomResizedCrop((224, 224)),
            Resize((img_size, img_size)),  # use this better to see whole image
            augmentation.Normalize(Tensor(DATASET_IMAGE_MEAN), Tensor(DATASET_IMAGE_STD)),
        )

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = image_to_tensor(np.array(x)).float() / 255.0
        assert len(x.shape) == 3, x.shape
        out = self.preprocess(x)
        return out[0]


class LitAugmenter(nn.Module):
    """Applies random augmentation to a batch of images."""

    def __init__(self, viz: bool = False):
        super().__init__()
        self.viz = viz
        self.augmentations = nn.Sequential(
            augmentation.RandomRotation(degrees=30.0),
            augmentation.RandomPerspective(distortion_scale=0.4),
            augmentation.RandomResizedCrop((224, 224)),
            augmentation.RandomHorizontalFlip(p=0.5),
            augmentation.RandomVerticalFlip(p=0.5),
            # K.augmentation.GaussianBlur((3, 3), (0.1, 2.0), p=1.0),
            # K.augmentation.ColorJitter(0.01, 0.01, 0.01, 0.01, p=0.25),
        )
        self.denorm = augmentation.Denormalize(Tensor(DATASET_IMAGE_MEAN), Tensor(DATASET_IMAGE_STD))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, x.shape
        out = x
        # idx = torch.randperm(len(self.geometric))[0]  # OneOf
        # out = self.geometric[idx](x)
        out = self.augmentations(out)
        if self.viz:
            out = self.denorm(out)
        return out


#: Kornia default augmentations
KORNIA_TRAIN_TRANSFORM = LitPreprocess(512)
KORNIA_VALID_TRANSFORM = LitPreprocess(224)
