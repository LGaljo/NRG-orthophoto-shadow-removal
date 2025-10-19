from datetime import datetime

import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np


if __name__ == "__main__":
    # image_path = 'unity_dataset/train_A/x250_z500_04b1cbda305c0646-35.png'
    # image_path = './img/DOF_D96TM_2018_2021_81527_73163_16.jpg'
    # image_path = 'C:\\Users\\lukag\\AppData\\LocalLow\\Magistrska naloge - Luka Galjot\\Mag Generate Shadows\\DOF_D96TM_2018_2021_83655_72571_16_2024-01-22_174815_x125-z105-Hard-e94d70fc758f8f2d-3.png'
    # image_path = 'C:\\Users\\lukag\\AppData\\LocalLow\\Magistrska naloge - Luka Galjot\\Mag Generate Shadows\\DOF_D96TM_2018_2021_82327_72099_16_2024-01-22_183524_x125-z105-Soft-1fe0bb42293cc62e-2.png'
    # image_path = 'C:\\Users\\lukag\\AppData\\LocalLow\\Magistrska naloge - Luka Galjot\\Mag Generate Shadows\\DOF_D96TM_2018_2021_82327_72099_16_2024-01-22_183524_x125-z105-Soft-1fe0bb42293cc62e-3.png'
    image_path = 'DOF_D96TM_2018_2021_83868_71675_16_2024-01-22_175110_x125-z125-Soft-643fd9ed7dae560a.png'
    # Load the image using skimage
    image = io.imread(image_path)
    image_grayscale = color.rgb2gray(image) * 256

    # Split the image into RGB channels
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Calculate the histogram for each channel
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    gray_hist, gray_bins = np.histogram(image_grayscale, bins=255, range=(0, 255))
    r_hist, r_bins = np.histogram(r, bins=255, range=(0, 255))
    g_hist, g_bins = np.histogram(g, bins=255, range=(0, 255))
    b_hist, b_bins = np.histogram(b, bins=255, range=(0, 255))

    # Plot the histograms
    plt.figure(figsize=(8, 6))
    plt.plot(gray_bins[:-1], gray_hist, color='gray', alpha=0.8, label='Gray')
    plt.plot(r_bins[:-1], r_hist, color='red', alpha=0.6, label='Red')
    plt.plot(g_bins[:-1], g_hist, color='green', alpha=0.6, label='Green')
    plt.plot(b_bins[:-1], b_hist, color='blue', alpha=0.6, label='Blue')
    # plt.figure(figsize=(12, 12))
    # plt.hist(r, bins=256, density=False, color='red', alpha=0.5)
    # plt.hist(g, bins=256, density=False, color='green', alpha=0.4)
    # plt.hist(b, bins=256, density=False, color='blue', alpha=0.3)

    plt.title(f'RGB Histogram {image_path}')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.savefig(f'histogram {datetime.now().strftime("%Y%m%d%H%M%S")}.png')
    plt.show()
    plt.pause(2)
    plt.close()
