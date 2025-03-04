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
    threshold_high = np.array(threshold_high, dtype=np.uint8)

    # Use cv2.inRange to create a binary mask
    mask = cv2.inRange(img, threshold_low, threshold_high)

    # The mask is already a grayscale image where white pixels (255) represent
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
    # Reshape the image to 3 dimensions
    img_three = np.tile(img.reshape(img.shape[0], img.shape[1], 1), (1, 1, 3))
    return img_three


def image_centroid_horizontal(img):
    """
    Compute the median of the x-coordinates of all white pixels in a binary image.
    Args:
        img (np.array): A binary image (1 channel) where pixels are either white (255) or black (0).
    Returns:
        x_centroid (int): The median of the x-coordinates of white pixels.
        Returns 0 if no white pixels are found.
    """
    # Ensure the image is grayscale
    if len(img.shape) != 2:
        raise ValueError("Input image must be grayscale (1 channel).")
    x_coords = np.where(img == 255)[1]  # Get the x-coordinates of all white pixels
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
        print("Error: Could not load image.")
        return

    # Define color thresholds
    thresh_low = (0, 20, 0)
    thresh_high = (150, 255, 150)

    # Segment the image
    img_seg = image_segment(img, thresh_low, thresh_high)

    # Compute the centroid
    x_centroid = image_centroid_horizontal(img_seg)

    # Convert to 3-channel image
    img_seg_color = image_one_to_three_channels(img_seg)

    # Draw vertical line at centroid
    img_result = image_line_vertical(img_seg_color, x_centroid)

    # Save the images instead of displaying them
    cv2.imwrite('segmented_image0.png', img_seg)
    cv2.imwrite('segmented_image_with_centroid_line0.png', img_result)

    print(f"Centroid X-Coordinate: {x_centroid}")
    print("Images saved as 'segmented_image0.png' and 'segmented_image_with_centroid_line0.png'.")


def image_mix(img_object, img_background,threshold_low, threshold_high):
    """
    Combine an object image with a new background by replacing the solid-color
    background in the object image with the new background.
    Args:
        img_obj (np.array): A color image with an object on a solid-color background.
        img_bg (np.array): A color image of an arbitrary background.
        thresh_low (tuple): Lower bounds for each channel (B, G, R).
        thresh_high (tuple): Upper bounds for each channel (B, G, R).
    Returns:
        img_mix (np.array): An image where the solid-color background is replaced
                            with the new background.
    """
    # Segmentation, identify background in image
    mask = image_segment(img_object, threshold_low, threshold_high)
    img_mix = img_object.copy()  # Create a copy of the object
    # Resize the background image to match the object image dimensions
    img_bg_resized = cv2.resize(img_background, (img_object.shape[1],img_object.shape[0]))
    # Replace the background pixels in the object image with
    # the corresponding pixels from the new background image
    img_mix[mask == 255] = img_bg_resized[mask == 255]
    return img_mix


if __name__ == '__main__':
    image_centroid_test()

    # Chroma key example
    img_obj = cv2.imread('object_on_green_screen.png')  # Replace with image
    img_bg = cv2.imread('new_background.png')  # Replace with background

    if img_obj is None or img_bg is None:
        print("Error: Could not load object or background image.")
    else:
        # Define thresholds for the background color
        thresh_low = (0, 50, 0)  # Lower bounds for B, G, R channels
        thresh_high = (100, 255, 100)  # Upper bounds for B, G, R channels

        # Chroma key compositing
        img_mix_result = image_mix(img_obj, img_bg, thresh_low, thresh_high)

        # Save or display the result
        cv2.imwrite('output_image.png', img_mix_result)
        print("Output image saved as 'output_image.png'.")
