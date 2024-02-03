import numpy as np
import cv2
from scipy.signal import convolve2d
from sklearn.cluster import KMeans


def read_rgb(path: str) -> np.ndarray:
    # Read an image
    image = cv2.imread(path)  # Replace 'your_image.jpg' with the path to your image file

    # OpenCV reads images in BGR format by default
    # If you want to convert it to RGB, you can do the following:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def read_gray(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_rgb(path: str, rgb: np.ndarray) -> None:
    rgb_8bit = cv2.convertScaleAbs(rgb)
    cv2.imwrite(path, cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR))


def save_gray(path: str, gray: np.ndarray) -> None:
    cv2.imwrite(path, gray)


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    # Convert RGB to Grayscale
    gray_array = np.dot(rgb[..., :3], [0.2989, 0.587, 0.114])

    # Make sure the resulting array is of dtype uint8
    gray_array = gray_array.astype(np.uint8)

    return gray_array


def grayscale_to_rgb(gray_array: np.ndarray) -> np.ndarray:
    # Create a three-channel array from the grayscale image
    rgb_array = np.stack((gray_array,) * 3, axis=-1)

    return rgb_array


def gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.

    Parameters:
    - size: Size of the kernel (odd number).
    - sigma: Standard deviation of the Gaussian distribution.

    Returns:
    - Gaussian kernel as a NumPy array.
    """

    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )

    return kernel / np.sum(kernel)


def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a given kernel to a 2D image.

    Parameters:
    - image: 2D NumPy array representing the image.
    - kernel: 2D NumPy array representing the kernel.

    Returns:
    - Resulting image after applying the kernel.
    """
    return convolve2d(image, kernel, mode='same', boundary='symm')


def quantize_image1(image, num_shades, random_seed=42):
    # Reshape the 2D array into a 1D array for k-means clustering
    pixels = image.reshape((-1, 1))

    # Convert to float32 for k-means
    pixels = np.float32(pixels)

    # Set a seed for reproducibility
    np.random.seed(random_seed)

    # Initialize cluster centers deterministically
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_shades, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)

    # Flatten the labels array and convert it back to the original image shape
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    return segmented_image.astype(np.uint8), centers.squeeze().astype(np.uint8)


def quantize_image2(image, num_shades, random_seed=42):
    # Reshape the 2D array into a 1D array for k-means clustering
    pixels = image.reshape((-1, 1))

    # Convert to float32 for k-means
    pixels = np.float32(pixels)

    # Set a seed for reproducibility
    np.random.seed(random_seed)

    # Initialize cluster centers deterministically
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_shades, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Flatten the labels array and convert it back to the original image shape
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    # Post-processing step to reduce unique colors to the specified num_shades
    unique_colors = np.unique(segmented_image)
    if len(unique_colors) > num_shades:
        thresholds = np.linspace(0, 255, num_shades + 1)[1:-1]
        for i in range(num_shades):
            low, high = thresholds[i], thresholds[i + 1]
            mask = np.logical_and(segmented_image >= low, segmented_image <= high)
            segmented_image[mask] = np.mean([low, high])

    return segmented_image.astype(np.uint8)


def quantize_image3(image, num_shades, random_seed=42):
    # Reshape the 2D array into a 1D array for k-means clustering
    pixels = image.reshape((-1, 1))

    # Convert to float32 for k-means
    pixels = np.float32(pixels)

    # Set a seed for reproducibility
    np.random.seed(random_seed)

    # Use k-means from sklearn with deterministic initialization
    kmeans = KMeans(n_clusters=num_shades, random_state=random_seed)
    kmeans.fit(pixels)

    # Get the cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Flatten the labels array and convert it back to the original image shape
    segmented_image = centers[labels].reshape(image.shape)

    return segmented_image.astype(np.uint8)


def calculate_average_color(img, mask) -> np.ndarray:
    """
    Calculate the average color of pixels where the mask is True in the RGB image.

    Parameters:
        - img: RGB image (NumPy array).
        - mask: Boolean mask (NumPy array) where True corresponds to relevant pixels.

    Returns:
        - average_color: NumPy array representing the average color (B, G, R).
    """
    # Use the mask to filter out relevant pixels in the image
    relevant_pixels = img[mask]

    # Calculate the average color across each channel
    average_color = np.mean(relevant_pixels, axis=0)

    return average_color.astype(np.uint8)


__all__ = [
    'read_gray',
    "read_rgb",
    "save_gray",
    "save_rgb",
    "apply_kernel",
    "gaussian_kernel",
    "rgb_to_grayscale",
    "grayscale_to_rgb",
    "quantize_image1",
    "quantize_image2",
    "quantize_image3",
    "calculate_average_color"
]
