#!/usr/bin/env python3
"""
This is a library of functions for performing color-based image segmentation
of an image and finding its centroid.
"""
import cv2
import numpy as np

def image_segment(img, threshold_low, threshold_high):
    """
    Perform color-based segmentation on an image.
    Args:
        img (np.array): A color image (3 channels: BGR).
        threshold_low (tuple): Lower bounds for each channel (B, G, R).
        threshold_high (tuple): Upper bounds for each channel (B, G, R).
    Returns:
        img_segmented (np.array): A grayscale image where white pixels (255)
        represent pixels within the bounds, and black
        pixels (0) represent pixels outside the bounds.
    """
    # Convert thresholds to numpy arrays
    threshold_low = np.array(threshold_low, dtype=np.uint8)
    threshold_high = np.array(threshold_high, dtype=np.uint8)# cv2.inRange for binary mask
    mask = cv2.inRange(img, threshold_low, threshold_high)# Convert to grayscale image
    img_segmented = mask
    return img_segmented

def image_line_vertical(img, x_line):
    """
    Adds a green 3px vertical line to the image.

    Args:
        img (np.array): A color image (3 channels: BGR).
        x_line (int): The x-coordinate of the vertical line.

    Returns:
        img_line (np.array): A copy of the input image with the vertical line drawn.
    """
    # Create a copy of the image to avoid modifying the original
    img_line = img.copy()
    # Draw a green vertical line
    cv2.line(img_line, (x_line, 0), (x_line, img.shape[0]), (0, 255, 0), 3)
    return img_line

def image_one_to_three_channels(img):
    """
    Transforms a grayscale (1-channel) image into a color (3-channel) image.
    Args:
        img (np.array): A grayscale image (1 channel).
    Returns:
        img_three (np.array): A color image (3 channels).
    """
    img_three = np.tile(img.reshape(img.shape[0], img.shape[1], 1), (1, 1, 3))    # Reshape the image to 3 dimensions
    return img_three

def image_centroid_horizontal(img):
    """
    Compute the median of the x-coordinates of all white pixels in a binary image.
    Args:
        img (np.array): A binary image (1 channel) where pixels are either white (255) or black (0).
    Returns:
        x_centroid (int): The median of the x-coordinates of white pixels. Returns 0 if no white pixels are found.
    """
    # Ensure the image is grayscale
    if len(img.shape) != 2:
        raise ValueError("Input image must be grayscale (1 channel).")
    x_coords = np.where(img == 255)[1]# Get the x-coordinates of all white pixels
    # Compute the median of x
    if len(x_coords) > 0:
        x_centroid = int(np.median(x_coords))  # median to int
    else:
        x_centroid = 0  # Default value if no white pixels
    return x_centroid
def image_centroid_test():
    """
    Perform a sequence of test operations on the test image.
    """
    # Load the test image
    img = cv2.imread('line-test.png')  # file path
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return
    # Define thresholds for segmentation (BGR format)
    t_low = (0, 0, 0)  # Lower bounds for B, G, R channels
    t_high = (255, 255, 255)  # Upper bounds for B, G, R channels
    # Perform segmentation
    img_seg = image_segment(img, t_low, t_high)
    # Compute the horizontal centroid of white pixels
    x_centroid = image_centroid_horizontal(img_seg)
    # Convert the segmented image to 3 channels for visualization
    color = image_one_to_three_channels(img_seg)
    # Add a vertical line at the centroid
    line = image_line_vertical(color, x_centroid)
    # Display the original and segmented images
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.imshow('Segmented Image with Centroid Line', line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    image_centroid_test()
def image_mix(img_object, img_background, threshold_low, threshold_hi gh):
    """
    Combine an object image with a new background by replacing the solid-color
    background in the object image with the new background.
    Args:
        img_object (np.array): A color image with an object on a solid-color background.
        img_background (np.array): A color image of an arbitrary background.
        threshold_low (tuple): Lower bounds for each channel (B, G, R).
        threshold_high (tuple): Upper bounds for each channel (B, G, R).
    Returns:
        img_mix (np.array): An image where the solid-color background is replaced
                            with the new background.
    """
    # segmentation, identify background in image
    mask = image_segment(img_object, threshold_low, threshold_high)
    # Create a copy of the object
    img_mix = img_object.copy()
    # Replace the background pixels in the object image with the corresponding pixels from the new background image
    img_mix[mask == 255] = img_background[mask == 255]
    return img_mix