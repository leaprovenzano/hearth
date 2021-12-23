from typing import Sequence
import random

import torch
from torch import Tensor
import torchvision.transforms as T  # noqa: N812

from hearth.modules import BaseModule
from hearth.vision.ops import square_center_crop


class NormalizeImg(BaseModule):
    """normalize a 3 channel image or a batch of 3 channel images.

    Inputs are expected to be floats in range 0.0:1.0 of shape (B?, C, W, H) where B? is an
    optional batch dimension.

    Args:
        mean: mean for each channel, defaults to imagenet defaults as outlined in torchvision.
        std: std for each channel, defaults to imagenet defaults as outlined in torchvision.

    Example:
        >>> from hearth.vision.transforms import NormalizeImg
        >>> _ = torch.manual_seed(0)
        >>>
        >>> # use torchvision imagenet defaults...
        >>> normalize = NormalizeImg()
        >>> single_image = torch.rand(3, 64, 64)
        >>> normalized_single_image = normalize(single_image)
        >>> normalized_single_image.mean((1, 2))
        tensor([0.0448, 0.2087, 0.4471])

        >>> normalized_single_image.std((1, 2))
        tensor([1.2547, 1.2854, 1.2874])

        >>> batch_images = torch.rand(5, 3, 64, 64)
        >>> normalized_batch_images = normalize(batch_images)
        >>> normalized_batch_images.mean((0, 2, 3))
        tensor([0.0778, 0.1930, 0.4113])

        >>> normalized_batch_images.std((0, 2, 3))
        tensor([1.2605, 1.2804, 1.2793])
    """

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).reshape(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).reshape(-1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class ResizeCrop(BaseModule):
    """resize a single image by scaling smallest dimension and center cropping with optional shift.

    Args:
        size: new size for square image.
        noise: for shifting the crop window along the longest dimension Defaults to 0.0 (no noise).

    Example:
        >>> import torch
        >>> from hearth.vision.transforms import ResizeCrop
        >>>
        >>> resize = ResizeCrop(224, crop_noise=0.5)
        >>> img = torch.randint(0, 255, size=(3, 1024, 986))
        >>> resize(img).shape
        torch.Size([3, 224, 224])
    """

    def __init__(self, size: int, crop_noise: float = 0.0):
        super().__init__()
        self.size = size
        self.crop_noise = crop_noise
        self.resize = T.Resize(self.size)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return square_center_crop(self.resize(img), noise=self.crop_noise)


class RandomFlipOrient(BaseModule):
    """randomly flip orientation of a single image

    Args:
        p: probability of flipping orientaion. Defaults to 0.5.

    Example:
        >>> from hearth.vision.transforms import RandomFlipOrient
        >>>
        >>> # set p=1.0 to ensure stuff happens
        >>> transform = RandomFlipOrient(p=1.0)
        >>> img = torch.randint(0, 255, size=(3, 128, 256))
        >>> transform(img).shape
        torch.Size([3, 256, 128])
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() <= self.p:
            return img.swapaxes(-2, -1).contiguous()
        return img
