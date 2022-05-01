import pytest
import torch

from kaggle_plantpathology.augment import LitAugmenter


@pytest.mark.parametrize("img_shape", [(1, 3, 192, 192), (2, 3, 224, 224)])
def test_augmenter(img_shape):
    B, C, H, W = img_shape
    img = torch.rand(img_shape)
    aug = LitAugmenter()

    assert aug(img).shape == (B, C, 224, 224)
