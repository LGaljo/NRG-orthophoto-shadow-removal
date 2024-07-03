import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from PIL import Image, ImageCms

from skimage.filters import threshold_multiotsu

# Setting the font size for all plots.
matplotlib.rcParams['font.size'] = 9

# The input image.
# rgb_image = Image.open("../13-2.png")
# rgb_image = Image.open('../img/merged/DOF_D96TM_2018_2021_83809_71597_16_2024-01-22_175803.jpg')
# rgb_image = Image.open('../test_images/shadowless-4-2.jpg').convert('L')
# rgb_image = Image.open('../13-2.png').convert('L')
rgb_image = Image.open('../img/merged/tiffs/DOF5-20240624-K11041280x2816.tif')
image = rgb_image.convert('L')
rgb_image = rgb_image.convert('RGB')
image = np.array(image)

# Applying multi-Otsu threshold for the default value, generating
# three classes.
thresholds = threshold_multiotsu(image)

# Using the threshold values, we generate the three regions.
regions = np.digitize(image, bins=thresholds)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# Plotting the original image.
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(image.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
    ax[1].axvline(thresh, color='r')

# Plotting the Multi Otsu result.
ax[2].imshow(regions, cmap='jet')
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

plt.subplots_adjust()

plt.savefig('../multi-otsu.png')
plt.show()

foreground = np.digitize(image, bins=[thresholds[0]])
# im = Image.fromarray(np.uint8(foreground * 255))
# im.show()

# Convert mask to boolean
suspected_shadows = foreground.astype(bool)

# Create an empty mask for true shadows
true_shadows = np.zeros_like(image, dtype=bool)

# Split the image into B, G, R channels
R, G, B = rgb_image.split()
R = np.array(R)
G = np.array(G)
B = np.array(B)

# Calculate the grayscale averages for B, G, and R wavebands
Gb = np.mean(np.array(B))
Gg = np.mean(np.array(G))
Gr = np.mean(np.array(R))

Ga = 6

# Iterate over each suspected shadow to determine if it's a true shadow or false
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if not suspected_shadows[i, j]:
            if B[i, j] + Ga < G[i, j]:
                # It's vegetation, rule out as shadow
                continue
            else:
                # It's a true shadow
                true_shadows[i, j] = True

# Convert the true shadow mask to an image
im = Image.fromarray(np.uint8(true_shadows * 255))
im.show()
im.save('shadow mask.jpg')
