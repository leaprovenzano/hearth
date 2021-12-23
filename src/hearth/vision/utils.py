import os
import re
import sys

import matplotlib.pyplot as plt


from torchvision.io import read_image as _read_img, ImageReadMode

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


ReadModeT = Literal['UNCHANGED', 'GRAY', 'GRAY_ALPHA', 'RGB', 'RGB_ALPHA']


_image_file_end_expr = re.compile(r'\.(png|jpe?g)$')


def is_valid_filename(path: str) -> bool:
    """check if a path to an image file is a valid png or jpg by checking filename.

    Args:
        path: path to check.

    Example:
        >>> from hearth.vision.utils import is_valid_filename
        >>>
        >>> is_valid_filename('blah.jpg')
        True

        >>> is_valid_filename('blah.png')
        True

        >>> is_valid_filename('blah.jpeg')
        True

        >>> is_valid_filename('nested/dir/blah.jpg')
        True

        >>> is_valid_filename('blah.txt')
        False
    """
    return _image_file_end_expr.search(path) is not None


def read_image(path: str, mode: ReadModeT = 'RGB'):
    """read an image (with proper handling for user dir) in given mode.

    Args:
        path: path to image file
        mode: string name for read mode, lowercased values will be translated to upercase. Defaults
            to 'RGB'.
    """
    return _read_img(os.path.expanduser(path), getattr(ImageReadMode, mode.upper()))


def show_img(img):
    """plot a single image."""
    return plt.imshow(img.transpose(0, -1))
