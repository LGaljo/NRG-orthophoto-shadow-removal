import numpy as np
import cv2


def otsu_thresholding(image):
    """
    Compute Otsu's threshold for an image.
    """
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(histogram)
    weight2 = np.cumsum(histogram[::-1])[::-1]

    mean1 = np.cumsum(histogram * bin_mids) / (weight1 + 1e-12)
    mean2 = (np.cumsum((histogram * bin_mids)[::-1]) / (weight2[::-1] + 1e-12))[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    return threshold


def tri_class_thresholding(image, preset_threshold=0.01):
    """
    Perform tri-class thresholding on the input image.
    """
    image = image.astype(np.float32) / 255.0  # Normalize the image to [0, 1]
    t = otsu_thresholding(image)
    t_prev = 0
    while abs(t - t_prev) >= preset_threshold:
        background = image[image <= t]
        foreground = image[image > t]

        if len(background) == 0 or len(foreground) == 0:
            break

        mean_background = np.mean(background)
        mean_foreground = np.mean(foreground)

        TBD = (image > mean_background) & (image <= mean_foreground)

        t_prev = t
        t = otsu_thresholding(image[TBD])

    # Final segmentation based on the last threshold
    segmented_image = np.zeros_like(image)
    segmented_image[image <= t_prev] = 0
    segmented_image[image > t] = 255
    segmented_image[(image > t_prev) & (image <= t)] = 128

    return (segmented_image * 255).astype(np.uint8)


# Example usage:
if __name__ == "__main__":
    # image = cv2.imread('../13-2.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('../img/merged/tiffs/DOF5-20240624-K11041280x2816.tif', cv2.IMREAD_GRAYSCALE)
    segmented_image = tri_class_thresholding(image)

    cv2.imshow('Original Image', image)
    cv2.imshow('Tri-Class Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
