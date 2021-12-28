import os
from dataclasses import dataclass
from typing import Union, Callable, Optional, Sequence, Tuple, List

import torch
from torch import Tensor
from torch.utils.data import Dataset

import torchvision.transforms as T  # noqa: N812

from hearth.data.datasets import BatchesMixin

from hearth.vision.utils import is_valid_filename, _image_file_end_expr, show_img, read_image
from hearth.vision.transforms import ResizeCrop


@dataclass
class RGBImageDataset(Dataset, BatchesMixin):
    """simple RGB image dataset that loads images from a directory. supports augmentations and \
    various utility methods as well as batch indexing (from list or slice).

    Images accessed with __getitem__ are read, resized, optionally augmented and finally transformed
    into float tensors in range ``[0,1]``.

    Args:
        directory: path to the directory to read images from. Homedir style paths are supported.
        size: integer or tuple size for image. If a ::attr::`size` is a single integer or
            a tuple of the same two values we will use
            :class:`hearth.vision.transforms.ResizeCrop` for resizing and
            :attr:`crop_noise` will be passed. If a tuple of different sizes are passed for
            non-square inputs :attr:`crop_noise` will be ignored and we will simply resize to the
            requested size. Defaults to 224 (commonly used in imagenet etc).
        crop_noise: noise: for shifting the crop window along the longest dimension.
            Defaults to 0.0 (no noise).
        augment: optional augmentation function which accepts a single image tensor of shape
            ``(C, W, H)`` of integers in range ``[0,255]`` returns augmented version of that image.
            Defaults to None (no augmentation).
        files: If a sequence of filenames is provided here we will use them instead of scanning the
            directory. The filenames are assumed to exist in the directory. This can be useful when
            breaking up a train and validation set which share the same directory or when working
            with a subset of the data. Defaults to None (directoy will be scanned.)

    """

    directory: str
    size: Union[int, Tuple[int, int]] = 224
    crop_noise: float = 0.0
    augment: Optional[Callable[[Tensor], Tensor]] = None
    files: Optional[Sequence[str]] = None

    def __post_init__(self):
        if self.files is None:
            self._files = list(
                filter(is_valid_filename, os.listdir(os.path.expanduser(self.directory)))
            )
        else:
            self._files = list(self.files)

        if isinstance(self.size, int):
            resize = ResizeCrop(self.size, self.crop_noise)
        elif self.size[0] == self.size[1]:
            resize = ResizeCrop(self.size[0], self.crop_noise)
        else:
            resize = T.Resize(self.size)
        self._resize = resize

    def _augment(self, img: Tensor) -> Tensor:
        if self.augment is None:
            return img
        return self.augment(img)

    def _build_path(self, filename: str) -> str:
        return os.path.join(self.directory, filename)

    def _get_filename(self, i: int) -> str:
        return self._files[i]

    def ids(self):
        """yields a list of ids (filenames stripped of endings)"""
        for f in self._files:
            yield _image_file_end_expr.sub('', f)

    def _load_img(self, filename, augment: bool = True):
        img = self._resize(read_image(self._build_path(filename)))
        if augment:
            img = self._augment(img)
        return img / 255

    def show_img(self, i: int, augment: bool = True):
        """display the image at index i with optional augmentation applied"""
        return show_img(self._load_img(self._files[i], augment=augment))

    def _get_batch(self, i: Union[slice, List[int]]):
        if isinstance(i, slice):
            filenames = self._files[i]
        else:
            filenames = [self._get_filename(ix) for ix in i]
        return torch.stack([self._load_img(f) for f in filenames])

    def _get_image(self, i: int):
        return self._load_img(self._files[i])

    def __getitem__(self, i: Union[int, slice, List[int]]):
        if isinstance(i, int):
            return self._get_image(i)
        return self._get_batch(i)

    def __len__(self) -> int:
        return len(self._files)
