import unittest

import numpy as np

from color_by_numbers.helpers import rgb_to_grayscale, grayscale_to_rgb, read_rgb, read_gray


class TestConversions(unittest.TestCase):
    pass
    # def test_rgb_to_grayscale(self):
    #     img = read_rgb("./input/colored_line.png")
    #     converted_gray = rgb_to_grayscale(img)
    #     gray = read_gray("./input/colored_line.png")
    #     self.assertTrue(np.array_equal(converted_gray, gray))
