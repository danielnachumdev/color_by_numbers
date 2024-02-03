from color_by_numbers import transform_image_rgb_individual_quantization, transform_image
from color_by_numbers.helpers import *
from tqdm import tqdm
from danielutils import measure, file_exists
import numpy as np


def test_best_func():
    func = measure(transform_image_rgb_individual_quantization)
    quantization_functions = [quantize_image1, quantize_image2, quantize_image3]
    dct = {quantize_image1: 0, quantize_image2: 0, quantize_image3: 0}
    tries = 10
    n_colors_range = range(5, 55, 5)
    t = tqdm(total=tries * len(quantization_functions) * len(n_colors_range))
    for i in (range(tries)):
        for quantization_function in quantization_functions:
            total = 0
            for n_colors in n_colors_range:
                duration = func("./input/img1.jpg", "./output/img1.jpg", n_colors, quantization_function)
                total += duration
                t.update()
            dct[quantization_function] += total
        print(dct)


@run_if(not file_exists("./output/v1.jpg"))
@announce
def v1():
    n_colors = 5
    kernel = gaussian_kernel(31, 5)
    transform_image_rgb_individual_quantization(
        "./input/img1.jpg",
        "./output/v1.jpg",
        lambda channel: quantize_dumb(apply_kernel(channel, kernel).astype(np.uint8), n_colors),
    )


@run_if(not file_exists("./output/v2.jpg"))
@announce
def v2():
    n_colors = 5
    kernel = gaussian_kernel(31, 5)
    transform_image_rgb_individual_quantization(
        "./input/img1.jpg",
        "./output/v2.jpg",
        lambda channel: quantize_image1(apply_kernel(channel, kernel).astype(np.uint8), n_colors)[0],
    )


@run_if(not file_exists("./output/v3.jpg"))
@announce
def v3():
    transform_image(
        "./input/img1.jpg",
        "./output/v3.jpg",
        5
    )


def main():
    v1()
    v2()
    v3()


if __name__ == "__main__":
    main()
