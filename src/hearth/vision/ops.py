import random
from torch import Tensor


def square_center_crop(img: Tensor, noise: float = 0.0) -> Tensor:
    """crop a single image of shape (channels, width, height) around the center with optional\
     random shift along the longest dimension.

    Args:
        img: single image tensor of shape (channels, width, height).
        noise: for shifting the crop window along the longest dimension Defaults to 0.0 (no noise).

    Example:
        >>> import torch
        >>> from hearth.vision.ops import square_center_crop
        >>>
        >>> img = torch.randint(0, 255, size=(3, 345, 256))
        >>> square_center_crop(img).shape
        torch.Size([3, 256, 256])

        >>> square_center_crop(img, noise=0.5).shape
        torch.Size([3, 256, 256])

        >>> square_center_crop(img.transpose(1, 2), noise=0.5).shape
        torch.Size([3, 256, 256])
    """
    _, w, h = img.shape
    # if height and width are same there's nothing to do...
    if w == h:
        return img
    out_size = min(w, h)
    cw = w - out_size
    ch = h - out_size
    noise = random.uniform(-noise, noise)
    # if theres's width to crop
    if cw:
        crop_left = int(round(cw / (2 + noise)))
        return img[:, crop_left : out_size + crop_left]
    # otherwise its longest on height...  crop that way...
    crop_top = int(round(ch / (2 + noise)))
    return img[:, :, crop_top : out_size + crop_top]
