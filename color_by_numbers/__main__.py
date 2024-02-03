from .helpers import *


def transform_image(in_path: str, out_path: str, n_colors: int = 2) -> None:
    img = read_rgb(in_path)
    gray = rgb_to_grayscale(img)
    kernel = gaussian_kernel(21, 5)
    blurred = apply_kernel(gray, kernel)
    # save_gray("./output/img1_blurred_gray.png", blurred)
    quantized = quantize_image(blurred, n_colors)
    res = grayscale_to_rgb(quantized)
    save_rgb(out_path, res)


__all__ = [
    'transform_image'
]
