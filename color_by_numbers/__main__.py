import numpy as np

from .helpers import read_rgb, save_rgb, combine_channels, separate_channels


def transform_image_rgb_individual_quantization(in_path: str, out_path: str, channel_mapping_function) -> None:
    img = read_rgb(in_path)
    res = combine_channels(
        list(map(
            channel_mapping_function,
            separate_channels(img)
        )))
    save_rgb(out_path, res)


def transform_image(in_path: str, out_path: str, n_colors: int) -> None:
    from .helpers import rgb_to_lab, lab_to_rgb, gaussian_kernel, apply_kernel, quantize_image1, save_gray
    img = read_rgb(in_path)
    l, a, b = separate_channels(rgb_to_lab(img))
    k = gaussian_kernel(31, 5)
    l = apply_kernel(l, k).astype(np.uint8)
    l, _ = quantize_image1(l, n_colors)
    img = lab_to_rgb(combine_channels([l, a, b]))
    save_rgb(out_path, img)


__all__ = [
    'transform_image_rgb_individual_quantization',
    'transform_image'
]
