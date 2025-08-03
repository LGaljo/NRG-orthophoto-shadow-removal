# USAGE
# python compare_unet_checkpoints_histogram.py
# import the necessary packages
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from model_unet import UNet

# Define model checkpoints to compare
MODEL_CHECKPOINTS = [
    {
        "name": "USOS",
        "model_path": "output/unet_shadow_20250703063322_usos_e240.pth",
    },
    {
        "name": "SRD",
        "model_path": "output/unet_shadow_20250629075948_srd_e500.pth",
    },
    {
        "name": "ISTD",
        "model_path": "output/unet_shadow_20250626223044_istd_e500.pth",
    },
    # Add more checkpoints as needed
]

# Define the dataset path
# DATASET_PATH = "../dataset/test_data"
DATASET_PATH = "../dataset/ortophoto_pretraining/test"

def make_prediction(model, image_path):
    """
    Make a prediction using the model and return the original image and prediction
    """
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, convert to RGB
        image = Image.open(image_path).convert('RGB')

        # resize the image
        image = image.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        image = np.array(image) / 255
        image = image.astype(np.float32)
        orig_image = image.copy()

        # prepare the image for the model
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image_tensor = torch.from_numpy(image).to(config.DEVICE)

        # make the prediction and convert to numpy array
        pred_image = model(image_tensor).squeeze()
        pred_image = pred_image.cpu().numpy()
        pred_image = np.transpose(pred_image, (1, 2, 0))

        return orig_image, pred_image

def compute_histogram(image, bins=256):
    """
    Compute histogram for each channel of the image
    """
    # Ensure image is in range [0, 1]
    if np.max(image) > 1.0:
        image = image / 255.0

    # Initialize histograms for each channel
    hist_r = np.histogram(image[:, :, 0], bins=bins, range=(0, 1))[0]
    hist_g = np.histogram(image[:, :, 1], bins=bins, range=(0, 1))[0]
    hist_b = np.histogram(image[:, :, 2], bins=bins, range=(0, 1))[0]

    # Normalize histograms
    hist_r = hist_r / np.sum(hist_r)
    hist_g = hist_g / np.sum(hist_g)
    hist_b = hist_b / np.sum(hist_b)

    return hist_r, hist_g, hist_b

def compute_combined_histogram(image, bins=256):
    """
    Compute a combined histogram of all RGB channels
    """
    # Ensure image is in range [0, 1]
    if np.max(image) > 1.0:
        image = image / 255.0

    # Flatten all channels into a single array
    flattened = image.reshape(-1)

    # Compute histogram
    hist_combined = np.histogram(flattened, bins=bins, range=(0, 1))[0]

    # Normalize histogram
    hist_combined = hist_combined / np.sum(hist_combined)

    return hist_combined

def compare_checkpoints():
    """
    Compare different model checkpoints by computing histograms of their predictions
    and also compute histograms of the pre-processed input images for comparison.

    This function:
    1. Loads different model checkpoints
    2. Processes test images through each model
    3. Computes histograms for both input images and model predictions
    4. Computes combined RGB histograms (all channels merged into one)
    5. Calculates average histograms for each model and for input images
    6. Plots and saves the histograms for comparison
    """
    # Create a directory to save results
    os.makedirs("histogram_comparison", exist_ok=True)

    # Get all shadow images
    shadow_images = glob.glob(os.path.join(DATASET_PATH, "*.jpg")) + \
                    glob.glob(os.path.join(DATASET_PATH, "*.png"))

    # Sort images
    shadow_images.sort()

    # Check if we have images
    num_images = len(shadow_images)
    if num_images == 0:
        print(f"No test images found in {DATASET_PATH}")
        return

    print(f"Found {num_images} test images")

    # Dictionary to store average histograms for each model
    avg_histograms = {}

    # Dictionary to store average histograms for input images
    input_avg_histograms = {
        'r': np.zeros(256),
        'g': np.zeros(256),
        'b': np.zeros(256),
        'combined': np.zeros(256)
    }

    # Process each model checkpoint
    for checkpoint in MODEL_CHECKPOINTS:
        print(f"Processing checkpoint: {checkpoint['name']}")

        # Load the model
        model_path = checkpoint["model_path"]
        print(f"Loading model from: {model_path}")
        model = torch.load(model_path, map_location=config.DEVICE).to(config.DEVICE)

        # Initialize histograms for this model
        model_hist_r = np.zeros(256)
        model_hist_g = np.zeros(256)
        model_hist_b = np.zeros(256)
        model_hist_combined = np.zeros(256)

        # Process each image
        for i, shadow_path in enumerate(shadow_images):
            # Print progress
            if i % 10 == 0:
                print(f"Processing image {i+1}/{num_images}")

            # Make prediction
            orig_image, pred_image = make_prediction(model, shadow_path)

            # Compute histogram for prediction
            hist_r, hist_g, hist_b = compute_histogram(pred_image)
            hist_combined = compute_combined_histogram(pred_image)

            # Compute histogram for input image (only for the first checkpoint to avoid redundancy)
            if checkpoint['name'] == MODEL_CHECKPOINTS[0]['name']:
                input_hist_r, input_hist_g, input_hist_b = compute_histogram(orig_image)
                input_hist_combined = compute_combined_histogram(orig_image)
                # Accumulate input histograms
                input_avg_histograms['r'] += input_hist_r
                input_avg_histograms['g'] += input_hist_g
                input_avg_histograms['b'] += input_hist_b
                input_avg_histograms['combined'] += input_hist_combined

            # Accumulate histograms
            model_hist_r += hist_r
            model_hist_g += hist_g
            model_hist_b += hist_b
            model_hist_combined += hist_combined

            # Save a sample of predictions and their histograms (first 5 images)
            if i < 10:
                # Create figure with four subplots (2x2 grid)
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

                # Plot the original image
                ax1.imshow(orig_image)
                ax1.set_title("Input Image")
                ax1.axis('off')

                # Plot the prediction
                ax2.imshow(pred_image)
                ax2.set_title(f"Prediction - {checkpoint['name']}")
                ax2.axis('off')

                # Plot the per-channel histograms
                bins = np.linspace(0, 1, 256)

                # If we're processing the first checkpoint, also plot input histograms
                # if checkpoint['name'] == MODEL_CHECKPOINTS[0]['name']:
                ax3.plot(bins, input_hist_r, color='r', alpha=0.4, label='Input Red', linestyle='--')
                ax3.plot(bins, input_hist_g, color='g', alpha=0.4, label='Input Green', linestyle='--')
                ax3.plot(bins, input_hist_b, color='b', alpha=0.4, label='Input Blue', linestyle='--')

                # Plot prediction histograms
                ax3.plot(bins, hist_r, color='r', alpha=0.7, label='Pred Red')
                ax3.plot(bins, hist_g, color='g', alpha=0.7, label='Pred Green')
                ax3.plot(bins, hist_b, color='b', alpha=0.7, label='Pred Blue')

                ax3.set_title(f"Per-Channel Histograms - {checkpoint['name']}")
                ax3.set_xlabel("Pixel Intensity")
                ax3.set_ylabel("Frequency")
                ax3.legend()

                # Plot the combined RGB histograms
                # if checkpoint['name'] == MODEL_CHECKPOINTS[0]['name']:
                ax4.plot(bins, input_hist_combined, color='black', alpha=0.7, label='Input Combined', linestyle='--')
                ax4.plot(bins, hist_combined, color='purple', alpha=0.7, label='Pred Combined')

                ax4.set_title(f"Combined RGB Histogram - {checkpoint['name']}")
                ax4.set_xlabel("Pixel Intensity")
                ax4.set_ylabel("Frequency")
                ax4.legend()

                # Save the figure
                plt.tight_layout()
                plt.savefig(f"histogram_comparison/{checkpoint['name']}_sample_{i+1}.png")
                plt.close()

        # Compute average histograms
        model_hist_r /= num_images
        model_hist_g /= num_images
        model_hist_b /= num_images
        model_hist_combined /= num_images

        # Store average histograms
        avg_histograms[checkpoint['name']] = {
            'r': model_hist_r,
            'g': model_hist_g,
            'b': model_hist_b,
            'combined': model_hist_combined
        }

    # Normalize input histograms
    input_avg_histograms['r'] /= num_images
    input_avg_histograms['g'] /= num_images
    input_avg_histograms['b'] /= num_images
    input_avg_histograms['combined'] /= num_images

    # Add input histograms to the comparison
    avg_histograms['Input Images'] = input_avg_histograms

    # Plot average histograms for comparison
    plt.figure(figsize=(15, 10))

    # Plot for red channel
    plt.subplot(3, 1, 1)
    for name, hist in avg_histograms.items():
        if name == 'Input Images':
            plt.plot(np.linspace(0, 1, 256), hist['r'], label=name, linestyle='--', linewidth=2, color='black')
        else:
            plt.plot(np.linspace(0, 1, 256), hist['r'], label=name)
    plt.title("Average Red Channel Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    # Plot for green channel
    plt.subplot(3, 1, 2)
    for name, hist in avg_histograms.items():
        if name == 'Input Images':
            plt.plot(np.linspace(0, 1, 256), hist['g'], label=name, linestyle='--', linewidth=2, color='black')
        else:
            plt.plot(np.linspace(0, 1, 256), hist['g'], label=name)
    plt.title("Average Green Channel Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    # Plot for blue channel
    plt.subplot(3, 1, 3)
    for name, hist in avg_histograms.items():
        if name == 'Input Images':
            plt.plot(np.linspace(0, 1, 256), hist['b'], label=name, linestyle='--', linewidth=2, color='black')
        else:
            plt.plot(np.linspace(0, 1, 256), hist['b'], label=name)
    plt.title("Average Blue Channel Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig("histogram_comparison/average_histograms_comparison.png")
    plt.close()

    # Plot combined average histograms (all channels in one plot)
    plt.figure(figsize=(15, 8))

    for name, hist in avg_histograms.items():
        if name == 'Input Images':
            plt.plot(np.linspace(0, 1, 256), hist['r'], label=f"{name} (Red)", linestyle='-', linewidth=2, color='darkred')
            plt.plot(np.linspace(0, 1, 256), hist['g'], label=f"{name} (Green)", linestyle='-', linewidth=2, color='darkgreen')
            plt.plot(np.linspace(0, 1, 256), hist['b'], label=f"{name} (Blue)", linestyle='-', linewidth=2, color='darkblue')
        else:
            plt.plot(np.linspace(0, 1, 256), hist['r'], label=f"{name} (Red)", linestyle='-')
            plt.plot(np.linspace(0, 1, 256), hist['g'], label=f"{name} (Green)", linestyle='--')
            plt.plot(np.linspace(0, 1, 256), hist['b'], label=f"{name} (Blue)", linestyle=':')

    plt.title("Average RGB Histograms Comparison")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("histogram_comparison/combined_average_histograms.png")
    plt.close()

    # Create a separate plot for input image histograms only
    plt.figure(figsize=(12, 6))

    bins = np.linspace(0, 1, 256)
    plt.plot(bins, input_avg_histograms['r'], label="Red Channel", color='red', linewidth=2)
    plt.plot(bins, input_avg_histograms['g'], label="Green Channel", color='green', linewidth=2)
    plt.plot(bins, input_avg_histograms['b'], label="Blue Channel", color='blue', linewidth=2)

    plt.title("Average Input Image Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("histogram_comparison/input_images_histograms.png")
    plt.close()

    # Create a plot for combined RGB histograms
    plt.figure(figsize=(15, 8))

    for name, hist in avg_histograms.items():
        if name == 'Input Images':
            plt.plot(np.linspace(0, 1, 256), hist['combined'], label=name, linestyle='--', linewidth=2, color='black')
        else:
            plt.plot(np.linspace(0, 1, 256), hist['combined'], label=name)

    plt.title("Combined RGB Channels Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("histogram_comparison/combined_rgb_histograms.png")
    plt.close()

    print("Histogram comparison completed. Results saved in 'histogram_comparison' directory.")

if __name__ == "__main__":
    compare_checkpoints()
