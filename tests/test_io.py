import unittest
import numpy as np

from color_by_numbers.helpers import read_rgb, save_rgb, save_gray, read_gray, rgb_to_grayscale


class TestIO(unittest.TestCase):
    def test_rgb_with_gray(self):
        rgb = read_rgb('./input/black_line.png')
        save_rgb("./output/black_line.png", rgb)
        res = read_rgb("./output/black_line.png")
        self.assertTrue(np.array_equal(rgb, res))

    def test_rgb_with_rgb(self):
        rgb = read_rgb('./input/colored_line.png')
        save_rgb("./output/colored_line.png", rgb)
        res = read_rgb("./output/colored_line.png")
        self.assertTrue(np.array_equal(rgb, res))

    def test_gray_with_gray(self):
        gray = read_gray('./input/black_line.png')
        save_gray("./output/black_line.png", gray)
        res = read_gray("./output/black_line.png")
        self.assertTrue(np.array_equal(gray, res))

    def test_gray_with_rgb(self):
        gray = read_gray('./input/colored_line.png')
        save_gray("./output/colored_line.png", gray)
        res = read_gray("./output/colored_line.png")
        self.assertTrue(np.array_equal(gray, res))
