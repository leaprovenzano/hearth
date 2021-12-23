import random
import os
import pytest
import uuid
import torch

from torchvision.utils import save_image
import torchvision.transforms as T  # noqa: N812

from matplotlib.image import AxesImage


from hearth.vision.datasets import RGBImageDataset
from hearth.vision.transforms import RandomFlipOrient


_ids = [str(uuid.uuid4()) for i in range(100)]

_augmentations = T.RandomApply(
    [
        T.RandomHorizontalFlip(p=0.5),
        RandomFlipOrient(0.5),
        T.ColorJitter(brightness=0.1, hue=0.1),
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.RandomAutocontrast(),
        T.RandomEqualize(),
    ],
    p=0.8,
)


@pytest.fixture(scope='module')
def image_dir(tmpdir_factory):
    d = tmpdir_factory.mktemp('images')
    dirname = str(d)

    # compute and save 100 random images of random sizes
    for _id in _ids:
        w = random.randint(32, 512)
        h = random.randint(32, 512)
        img = torch.rand(3, w, h)
        path = os.path.join(dirname, f'{_id}.jpg')
        save_image(img, path)

    return dirname


@pytest.mark.parametrize(
    'filenames, expected', [(list(map(lambda x: f'{x}.jpg', _ids[:10])), 10), (None, 100)]
)
def test_len(filenames, expected, image_dir):
    ds = RGBImageDataset(image_dir, files=filenames)
    assert len(ds) == expected


def test_ids(image_dir):
    ds = RGBImageDataset(image_dir)
    ids = list(ds.ids())
    assert len(ids) == len(_ids)
    assert set(ids) == set(_ids)


@pytest.mark.parametrize('augment,', [None, _augmentations])
def test_item_indexing(augment, image_dir):
    ds = RGBImageDataset(image_dir, size=32, augment=augment)
    img = ds[5]
    assert img.shape == (3, 32, 32)
    assert img.max().item() <= 1.0
    assert img.amin().item() >= 0.0


@pytest.mark.parametrize('idx,', [slice(0, 5), [9, 0, 11, 66, 3]])
@pytest.mark.parametrize('augment,', [None, _augmentations])
def test_batch_indexing(idx, augment, image_dir):
    ds = RGBImageDataset(image_dir, size=32, augment=augment)
    img = ds[idx]

    assert img.shape == (5, 3, 32, 32)
    assert img.max().item() <= 1.0
    assert img.amin().item() >= 0.0


def test_show_img(image_dir):
    ds = RGBImageDataset(image_dir, size=32)
    plt = ds.show_img(99)
    assert isinstance(plt, AxesImage)
