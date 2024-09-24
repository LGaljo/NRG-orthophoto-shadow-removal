import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=5):
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, image.shape)

    # Add the noise to the image
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return noisy_image


# Load image
image_path = '../13-2.png'
image = Image.open(image_path)  # Read image in grayscale
image = np.array(image)

# Add Gaussian noise
noisy_image = add_gaussian_noise(image)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Image with Gaussian Noise")
plt.imshow(noisy_image, cmap='gray')

plt.show()
