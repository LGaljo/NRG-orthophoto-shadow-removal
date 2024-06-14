import cv2
import numpy as np


def log_chromaticity(image):
    # Convert to float to avoid overflow issues
    image = image.astype(np.float32) + 1.0
    log_image = np.log(image)

    # Subtract the log intensity (mean of R, G, B logs)
    log_intensity = np.mean(log_image, axis=2, keepdims=True)
    log_chrom = log_image - log_intensity
    return log_chrom


def detect_shadow_edges(log_chrom):
    # Detect edges in the log-chromaticity image
    edges = cv2.Canny((log_chrom * 255).astype(np.uint8), 50, 150)
    return edges


def classify_edges(edges):
    # This is a placeholder for edge classification
    # For simplicity, consider all edges as shadow edges
    shadow_edges = edges
    return shadow_edges


def smooth_illumination(shadow_edges, image):
    # Create a mask of shadow edges
    shadow_mask = shadow_edges > 0

    # Create an illumination image by blurring the original image
    illumination_image = cv2.GaussianBlur(image, (21, 21), 0)

    # Combine original image with the smoothed illumination
    combined_image = np.where(shadow_mask[..., None], illumination_image, image)
    return combined_image


def remove_shadows(image):
    log_chrom = log_chromaticity(image)
    edges = detect_shadow_edges(log_chrom)
    shadow_edges = classify_edges(edges)
    result = smooth_illumination(shadow_edges, image)
    return result


if __name__ == "__main__":
    image_path = "../13-2.png"  # Replace with your image path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not open or find the image: {image_path}")
        exit()

    shadow_free_image = remove_shadows(image)
    cv2.imwrite("../shadow_free_image.jpg", shadow_free_image)
    cv2.imshow("Shadow Free Image", shadow_free_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
