from typing import Sequence
import torch
from torch import Tensor
from hearth.modules import BaseModule


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
