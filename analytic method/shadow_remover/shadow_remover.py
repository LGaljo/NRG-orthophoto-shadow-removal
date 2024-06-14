from typing import Tuple, List

import cv2 as cv
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from skimage.filters import threshold_multiotsu
import concurrent.futures

from tqdm import tqdm

'''
Shadow removal code by Yalım Doğan
https://github.com/YalimD/image_shadow_remover

This software is implemented according to the methods presented in:

- Murali, Saritha, and V. K. Govindan. 
"Removal of shadows from a single image." 
the Proceedings of First International Conference on Futuristic Trends 
in Computer Science and Engineering. Vol. 4.

- Murali, Saritha, and V. K. Govindan. 
"Shadow detection and removal from a single image using LAB color space." 
Cybernetics and information technologies 13.1 (2013): 95-103.

'''


# Applies median filtering over given point
def median_filter(img: np.ndarray,
                  point: np.ndarray,
                  filter_size: int) -> List:
    indices = [[x, y]
               for x in range(point[1] - filter_size // 2, point[1] + filter_size // 2 + 1)
               for y in range(point[0] - filter_size // 2, point[0] + filter_size // 2 + 1)]

    indices = list(filter(lambda x: not (x[0] < 0 or x[1] < 0 or
                                         x[0] >= img.shape[0] or
                                         x[1] >= img.shape[1]), indices))

    pixel_values = [0, 0, 0]

    # Find the median of pixel values
    for channel in range(3):
        pixel_values[channel] = list(img[index[0], index[1], channel] for index in indices)
    pixel_values = list(np.median(pixel_values, axis=1))

    return pixel_values


# Applies median filtering on given contour pixels, the filter size is adjustable
def edge_median_filter(img: np.ndarray,
                       contours_list: tuple,
                       filter_size: int = 7) -> np.ndarray:
    temp_img = np.copy(img)

    for partition in contours_list:
        for point in partition:
            temp_img[point[0][1]][point[0][0]] = median_filter(img,
                                                               point[0],
                                                               filter_size)

    return cv.cvtColor(temp_img, cv.COLOR_HSV2BGR)


def display_region(org_image: np.ndarray,
                   shadow_clear_image: np.ndarray,
                   label: int,
                   label_region: np.ndarray,
                   contours: tuple) -> None:
    # For debugging, cut the current shadow region from the image
    reverse_mask = cv.cvtColor(cv.bitwise_not(label_region), cv.COLOR_GRAY2BGR)
    img_w_hole = org_image & reverse_mask

    temp_filter = cv.cvtColor(label_region, cv.COLOR_GRAY2BGR)
    cv.drawContours(temp_filter, contours, -1, (255, 0, 0), 3)

    fig, axes = plt.subplots(2, 2)

    ax = axes.ravel()

    plt.title(f"Shadow Region {label}")

    ax[0].imshow(cv.cvtColor(org_image, cv.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")

    ax[1].imshow(cv.cvtColor(temp_filter, cv.COLOR_BGR2RGB))
    ax[1].set_title("Shadow Region")

    ax[2].imshow(cv.cvtColor(img_w_hole, cv.COLOR_BGR2RGB))
    ax[2].set_title("Shadow Region Cut")

    ax[3].imshow(cv.cvtColor(shadow_clear_image, cv.COLOR_BGR2RGB))
    ax[3].set_title("Corrected Image")

    plt.tight_layout()
    plt.show()


def correct_region_lab(org_img: np.ndarray,
                       shadow_clear_img: np.ndarray,
                       shadow_indices: np.ndarray,
                       non_shadow_indices: np.ndarray) -> np.ndarray:
    # Q: Rather than asking for RGB constants individually, why not adjust L only?
    # A: L component isn't enough to REVIVE the colors that were under the shadow.

    # Calculate average LAB values in current shadow region and non-shadow areas
    shadow_average_lab = np.mean(org_img[shadow_indices[0], shadow_indices[1], :], axis=0)

    # Get the average LAB from border areas
    border_average_lab = np.mean(org_img[non_shadow_indices[0], non_shadow_indices[1], :],
                                 axis=0)

    # Calculate ratios that are going to be used on clearing the current shadow region
    # This is different for each region, therefore calculated each time
    lab_ratio = border_average_lab / shadow_average_lab

    shadow_clear_img = cv.cvtColor(shadow_clear_img, cv.COLOR_BGR2LAB)
    shadow_clear_img[shadow_indices[0], shadow_indices[1]] = (
        np.uint8(shadow_clear_img[shadow_indices[0], shadow_indices[1]] * lab_ratio)
    )
    shadow_clear_img = cv.cvtColor(shadow_clear_img, cv.COLOR_LAB2BGR)

    return shadow_clear_img


def correct_region_bgr(org_img: np.ndarray,
                       shadow_clear_img: np.ndarray,
                       shadow_indices: np.ndarray,
                       non_shadow_indices: np.ndarray) -> np.ndarray:
    # Calculate average BGR values in current shadow region and non-shadow areas
    shadow_average_bgr = np.mean(org_img[shadow_indices[0], shadow_indices[1], :], axis=0)

    # Get the average BGR from border areas
    border_average_bgr = np.mean(org_img[non_shadow_indices[0], non_shadow_indices[1], :], axis=0)
    bgr_ratio = border_average_bgr / shadow_average_bgr

    # Adjust BGR
    shadow_clear_img[shadow_indices[0], shadow_indices[1]] = np.uint8(
        shadow_clear_img[shadow_indices[0],
        shadow_indices[1]] * bgr_ratio)

    return shadow_clear_img


def process_regions(org_image: np.ndarray,
                    mask: np.ndarray,
                    lab_adjustment: bool,
                    shadow_dilation_kernel_size: int,
                    shadow_dilation_iteration: int,
                    shadow_size_threshold: int,
                    verbose: bool) -> np.ndarray:
    lab_img = cv.cvtColor(org_image, cv.COLOR_BGR2LAB)
    shadow_clear_img = np.copy(org_image)  # Used for constructing corrected image

    # We need connected components
    # Initialize the labels of the blobs in our binary image
    labels = measure.label(mask)

    non_shadow_kernel_size = (shadow_dilation_kernel_size, shadow_dilation_kernel_size)
    non_shadow_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, non_shadow_kernel_size)

    CHANNEL_MAX = 255

    # Now, we will iterate over each label's pixels
    for label in np.unique(labels):
        if not label == 0:
            temp_filter = np.zeros(mask.shape, dtype="uint8")
            temp_filter[labels == label] = CHANNEL_MAX

            # Only consider blobs with size above threshold
            if cv.countNonZero(temp_filter) >= shadow_size_threshold:
                shadow_indices = np.where(temp_filter == CHANNEL_MAX)

                non_shadow_temp_filter = cv.dilate(temp_filter, non_shadow_kernel,
                                                   iterations=shadow_dilation_iteration)

                # Get the new set of indices and remove shadow indices from them
                non_shadow_temp_filter = cv.bitwise_xor(non_shadow_temp_filter, temp_filter)
                non_shadow_indices = np.where(non_shadow_temp_filter == CHANNEL_MAX)

                # Contours are used for extracting the edges of the current shadow region
                contours, hierarchy = cv.findContours(temp_filter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                if lab_adjustment:
                    shadow_clear_img = correct_region_lab(lab_img, shadow_clear_img,
                                                          shadow_indices, non_shadow_indices)
                else:
                    shadow_clear_img = correct_region_bgr(org_image, shadow_clear_img,
                                                          shadow_indices, non_shadow_indices)

                # Then apply median filtering over edges to smooth them
                # At least on the images I tried, this doesn't work as intended.
                # It is possible that this is the result of using a high frequency image only

                # Image is converted to HSV before filtering, as BGR components of the image
                # is more interconnected, therefore filtering each channel independently wouldn't be correct
                shadow_clear_img = edge_median_filter(cv.cvtColor(shadow_clear_img, cv.COLOR_BGR2HSV), contours)
                if verbose:
                    display_region(org_image, shadow_clear_img, label, temp_filter, contours)

    return shadow_clear_img


def calculate_mask(org_image: np.ndarray) -> np.ndarray:
    gray_image = cv.cvtColor(org_image, cv.COLOR_BGR2GRAY)

    # Applying multi-Otsu threshold for the default value, generating three classes.
    thresholds = threshold_multiotsu(gray_image)

    foreground = np.digitize(gray_image, bins=[thresholds[0]])

    # Convert mask to boolean
    suspected_shadows = foreground.astype(bool)

    # Create an empty mask for true shadows
    true_shadows = np.zeros_like(gray_image)

    # Split the image into B, G, R channels
    # R, G, B = org_image.split()
    R = org_image[:, :, 1]
    G = org_image[:, :, 1]
    B = org_image[:, :, 2]

    Ga = 12

    # Iterate over each suspected shadow to determine if it's a true shadow or false
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if not suspected_shadows[i, j]:
                if B[i, j] + Ga < G[i, j]:
                    # It's vegetation, rule out as shadow
                    continue
                else:
                    # It's a true shadow
                    true_shadows[i, j] = 255

    return true_shadows


def remove_shadows(org_image: np.ndarray,
                   lab_adjustment: bool,
                   shadow_dilation_iteration: int,
                   shadow_dilation_kernel_size: int,
                   shadow_size_threshold: int,
                   verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    mask = calculate_mask(org_image)

    shadow_clear_img = process_regions(org_image,
                                          mask,
                                          lab_adjustment,
                                          shadow_dilation_kernel_size,
                                          shadow_dilation_iteration,
                                          shadow_size_threshold,
                                          verbose)

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

    return shadow_clear_img, mask


def split_image_into_tiles(image, tile_size):
    tiles = []
    h, w, _ = image.shape
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append((tile, x, y))
    return tiles


def merge_tiles(tiles, image_shape, tile_size):
    h, w, _ = image_shape
    merged_image = np.zeros(image_shape, dtype=np.uint8)
    for tile, x, y in tiles:
        merged_image[y:y+tile_size, x:x+tile_size] = tile
    return merged_image


def process_image_file(img_name: str,
                       save: bool = False,
                       lab_adjustment: bool = False,
                       shadow_dilation_kernel_size: int = 5,
                       shadow_dilation_iteration: int = 3,
                       shadow_size_threshold: int = 2500,
                       verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    org_image = cv.imread(img_name)
    print("Read the image {}".format(img_name))
    tile_size = 128

    # shadow_clear, mask = remove_shadows(org_image,
    #                                     lab_adjustment,
    #                                     shadow_dilation_iteration,
    #                                     shadow_dilation_kernel_size,
    #                                     shadow_size_threshold,
    #                                     verbose=verbose)
    #
    # Split the image into 256x256 tiles
    tiles = split_image_into_tiles(org_image, tile_size)

    # Process each tile to remove shadows
    processed_tiles = []
    processed_mask_tiles = []
    for tile, x, y in tiles:
        shadow_free_tile, mask_tile = remove_shadows(tile,
                                                     lab_adjustment,
                                                     shadow_dilation_iteration,
                                                     shadow_dilation_kernel_size,
                                                     shadow_size_threshold,
                                                     verbose)
        processed_tiles.append((shadow_free_tile, x, y))
        processed_mask_tiles.append((mask_tile, x, y))

    # Merge the processed tiles back into a single image
    shadow_clear = merge_tiles(processed_tiles, org_image.shape, tile_size)
    mask = merge_tiles(processed_mask_tiles, org_image.shape, tile_size)

    _, axes = plt.subplots(1, 3)
    ax = axes.ravel()

    plt.title("Final Results")

    ax[0].imshow(cv.cvtColor(org_image, cv.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")

    ax[1].imshow(cv.cvtColor(mask, cv.COLOR_BGR2RGB))
    ax[1].set_title("Shadow Regions")

    ax[2].imshow(cv.cvtColor(shadow_clear, cv.COLOR_BGR2RGB))
    ax[2].set_title("Corrected Image")

    plt.tight_layout()
    plt.show()

    if save:
        path = img_name.split("/")
        f_name = '/'.join(path[:-1]) + '/' + path[-1].split('.')[0] + "_shadowClear." + path[-1].split('.')[1]
        cv.imwrite(f_name, shadow_clear)
        print("Saved result as " + f_name)

    return org_image, mask, shadow_clear


if __name__ == "__main__":
    for img in [
        "DOF5-20240602-D0738 1.tif",
        "DOF5-20240602-E0514 1.tif",
        "DOF5-20240602-E0637 1.tif",
        "DOF5-20240602-E0746 1.tif",
        "DOF5-20240602-G0715 1.tif",
        "DOF5-20240602-K1013 1.tif",
    ]:
        process_image_file(
            # '../../DOF_D96TM_2018_2021_83809_71597_16_2024-01-22_175803.jpg',
            f'../../img/merged/{img}',
            True,
            True,
            3,
            1,
            16,
            False)
