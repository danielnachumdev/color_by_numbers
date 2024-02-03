import numpy as np
import cv2

from .helpers import *


def transform_image(in_path: str, out_path: str, n_colors: int, q_function, *, verbose: bool = False) -> None:
    img = read_rgb(in_path)
    l, a, b = lab_to_channels(rgb_to_lab(img))
    # for layer in [l,a,b]
    save_gray("./gray.png", l)
    gray = l
    kernel = gaussian_kernel(31, 5)
    blurred = apply_kernel(gray, kernel)
    quantized, colors = q_function(blurred, n_colors)
    for c in colors:
        # mark all of the pixels which equal a value, convert to {0,1}.
        # if we want to save the mask and see we should multiply by 255 so we can see the difference
        mask = np.array(quantized == c)  # .astype(np.uint8)
        new_color = calculate_average_color(img, mask)
        img[mask] = new_color

    save_rgb(out_path, img)


__all__ = [
    'transform_image'
]
