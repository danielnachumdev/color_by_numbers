import numpy as np

from .helpers import *


def transform_image(in_path: str, out_path: str, n_colors: int, q_function, *, verbose: bool = False) -> None:
    if verbose:
        print("Rrading image")
    img = read_rgb(in_path)
    if verbose:
        print("Converting to grayscale")
    gray = rgb_to_grayscale(img)
    if verbose:
        print("Applying blur kernel")
    kernel = gaussian_kernel(31, 5)
    blurred = apply_kernel(gray, kernel)
    if verbose:
        print("Quantizing the colors")
    quantized, colors = q_function(blurred, n_colors)
    if verbose:
        print("Applying quantized colors to the image")
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
