import numpy as np
import unittest

from color_by_numbers.helpers import apply_kernel


class TestApplyKernelFunction(unittest.TestCase):

    def test_apply_kernel_square_identity(self):
        # Test applying a square identity kernel (no change should occur)
        image = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

        identity_kernel = np.array([[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]])

        result = apply_kernel(image, identity_kernel)
        np.testing.assert_array_equal(result, image)

    def test_apply_kernel_ones(self):
        # Test applying a kernel of all ones (blurring effect)
        image = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

        ones_kernel = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

        result = apply_kernel(image, ones_kernel)
        expected_result = np.array([[12, 21, 18],
                                    [27, 45, 33],
                                    [24, 39, 27]])
        np.testing.assert_array_equal(result, expected_result)

    def test_apply_kernel_edge_detection(self):
        # Test applying an edge detection kernel
        image = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

        edge_detection_kernel = np.array([[-1, -1, -1],
                                          [-1, 8, -1],
                                          [-1, -1, -1]])

        result = apply_kernel(image, edge_detection_kernel)
        expected_result = np.array([[5, 8, 3],
                                    [-8, 0, -6],
                                    [13, 8, 15]])
        np.testing.assert_array_equal(result, expected_result)
