#!/usr/bin/env python
# normalize_image.py
# This script loads a specific image from the dataset and normalizes it

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms.v2 import functional as F, Compose, ToImage, ToDtype

# Define the image path
IMAGE_PATH = "dataset/unity_dataset/mixed_visibility_dataset_320/train/train_A/DOF5-20240602-D0717_shadowClear_12e28f16b79320f8-x255-z255-Hard-39.png"

# Define image dimensions (from config.py)
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256

def normalize_image(image_path, save_output=True, display_output=True):
    """
    Load an image, normalize it, and optionally save and display the result
    
    Args:
        image_path (str): Path to the image file
        save_output (bool): Whether to save the normalized image
        display_output (bool): Whether to display the normalized image
    
    Returns:
        tuple: Original image array, normalized image array
    """
    print(f"Loading image from: {image_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Resize the image to the dimensions used in the model
    image = image.resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Store original image for comparison
    original_image = image_array.copy()
    
    # Normalize the image by dividing by 255 (as done in predict.py)
    normalized_image = image_array / 255.0
    normalized_image = normalized_image.astype(np.float32)
    
    # Alternative normalization method (commented out in dataset.py)
    # Convert to tensor for alternative normalization
    to_tensor = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=False)
    ])
    
    to_scaled_tensor = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True)
    ])

    image_tensor = to_tensor(image)
    image_scaled_tensor = to_scaled_tensor(image)
    
    # Find the maximum value
    max_value = torch.max(image_tensor)
    
    # Normalize by the maximum value
    alt_normalized_tensor = image_tensor / max_value if max_value > 0 else image_tensor
    
    # Convert back to numpy for display
    alt_normalized_image = alt_normalized_tensor.numpy()

    alt_normalized_image_2 = image_scaled_tensor.numpy()
    
    # Save the normalized images if requested
    if save_output:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the normalized image (0-1 range)
        plt.imsave(os.path.join(output_dir, "normalized_image_div255.png"), normalized_image)
        
        # Save the alternatively normalized image (0-1 range)
        plt.imsave(os.path.join(output_dir, "normalized_image_maxvalue.png"), np.transpose(alt_normalized_image, (1, 2, 0)))
        
        # Save the alternatively normalized image (0-1 range)
        plt.imsave(os.path.join(output_dir, "normalized_image_scaled_tensor.png"), np.transpose(alt_normalized_image_2, (1, 2, 0)))

        print(f"Saved normalized images to {output_dir}")
    
    # Display the images if requested
    if display_output:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(normalized_image)
        plt.title("Normalized Image (div by 255)")
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(np.transpose(alt_normalized_image, (1, 2, 0)))
        plt.title("Normalized Image (div by max value)")
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(np.transpose(alt_normalized_image_2, (1, 2, 0)))
        plt.title("Normalized Image (scaled=True tensor)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    
    return original_image, normalized_image, alt_normalized_image, alt_normalized_image_2

if __name__ == "__main__":
    # Normalize the image
    original, normalized, alt_normalized, alt_normalized_2 = normalize_image(IMAGE_PATH)
    
    # Print some statistics about the images
    print(f"Original image shape: {original.shape}")
    print(f"Original image min: {original.min()}, max: {original.max()}")
    print(f"Normalized image min: {normalized.min()}, max: {normalized.max()}")
    print(f"Alt normalized image min: {np.min(alt_normalized)}, max: {np.max(alt_normalized)}")
    print(f"Alt normalized 2 image min: {np.min(alt_normalized_2)}, max: {np.max(alt_normalized_2)}")