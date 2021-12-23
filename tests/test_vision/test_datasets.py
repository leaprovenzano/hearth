import random
import os
import pytest
import uuid
import torch
from torchvision.utils import save_image
from hearth.vision.datasets import RGBImageDataset

_ids = [str(uuid.uuid4()) for i in range(100)]


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


def test_len(image_dir):
    ds = RGBImageDataset(image_dir)
    assert len(ds) == 100
