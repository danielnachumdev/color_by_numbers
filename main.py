from color_by_numbers import transform_image
from color_by_numbers.helpers import quantize_image1, quantize_image2, quantize_image3
from tqdm import tqdm
from danielutils import measure


def test_best_func():
    func = measure(transform_image)
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


def main():
    transform_image("./input/img1.jpg", "./output/img1.jpg", 4, quantize_image1, verbose=True)


if __name__ == "__main__":
    main()
