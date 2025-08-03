# USAGE
# python compare_stcgan_predictions.py
# import the necessary packages
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from collections import OrderedDict

from ST_CGAN import Generator

# Define datasets and their corresponding model checkpoints
DATASETS = [
    {
        "name": "ISTD",
        "model_path_G1": "trained_models/st-cgan/istd_results/checkpoints/ST-CGAN_G1_1000.pth",
        "model_path_G2": "trained_models/st-cgan/istd_results/checkpoints/ST-CGAN_G2_1000.pth",
        "test_shadow_path": "trained_models/st-cgan/istd_results/dataset/test/test_A",
        "test_shadow_mask_path": "trained_models/st-cgan/istd_results/dataset/test/test_B",
        "test_shadow_free_path": "trained_models/st-cgan/istd_results/dataset/test/test_C"
    },
    {
        "name": "SRD",
        "model_path_G1": "trained_models/st-cgan/srd_results/checkpoints/ST-CGAN_G1_1000.pth",
        "model_path_G2": "trained_models/st-cgan/srd_results/checkpoints/ST-CGAN_G2_1000.pth",
        "test_shadow_path": "trained_models/st-cgan/srd_results/dataset/test/test_A",
        "test_shadow_mask_path": "trained_models/st-cgan/srd_results/dataset/test/test_B",
        "test_shadow_free_path": "trained_models/st-cgan/srd_results/dataset/test/test_C"
    },
    {
        "name": "USOS",
        "model_path_G1": "trained_models/st-cgan/usos_results/checkpoints/ST-CGAN_G1_200.pth",
        "model_path_G2": "trained_models/st-cgan/usos_results/checkpoints/ST-CGAN_G2_200.pth",
        "test_shadow_path": "trained_models/st-cgan/usos_results/dataset/test/test_A",
        "test_shadow_mask_path": "trained_models/st-cgan/usos_results/dataset/test/test_B",
        "test_shadow_free_path": "trained_models/st-cgan/usos_results/dataset/test/test_C"
    },
]

# Input image size
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fix_model_state_dict(state_dict):
    """
    Remove 'module.' prefix from dataparallel models
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def prepare_plot(origImage, gtImage, predMask, title=None):
    """
    Prepare a plot with original shadowed image, ground truth shadow-free image, and predicted shadow-free image
    """
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # plot the original image, ground truth, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(gtImage)
    ax[2].imshow(predMask)

    # set the titles of the subplots
    ax[0].set_title("Shadowed Image")
    ax[1].set_title("Ground Truth")
    ax[2].set_title("Prediction")

    # set the main title if provided
    if title:
        figure.suptitle(title)

    # set the layout of the figure and display it
    figure.tight_layout()
    return figure

def calculate_metrics(gt_image, pred_image):
    """
    Calculate SSIM, PSNR, and RMSE metrics between ground truth and predicted images
    """
    # Convert images to numpy arrays if they're not already
    if isinstance(gt_image, torch.Tensor):
        gt_image = gt_image.cpu().numpy()
    if isinstance(pred_image, torch.Tensor):
        pred_image = pred_image.cpu().numpy()

    # Calculate SSIM
    ssim_value = ssim(gt_image, pred_image, data_range=1.0, multichannel=True, channel_axis=2)

    # Calculate PSNR
    psnr_value = psnr(gt_image, pred_image, data_range=1.0)

    # Calculate RMSE (Root Mean Square Error)
    mse_value = mse(gt_image, pred_image)
    rmse_value = np.sqrt(mse_value)

    return ssim_value, psnr_value, rmse_value

def unnormalize(x):
    """
    Unnormalize the image from [-1, 1] to [0, 1]
    """
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x

def make_predictions(G1, G2, path_t, path_gt):
    """
    Make predictions using the ST-CGAN model and return the original image, ground truth, and prediction
    """
    # set models to evaluation mode
    G1.eval()
    G2.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, convert to RGB
        image = Image.open(path_t).convert('RGB')

        # resize the image and make a copy of it for visualization
        image = image.resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        # Normalize to [-1, 1]
        image = (image - 0.5) / 0.5
        og_image = (image * 0.5 + 0.5).copy()  # Convert back to [0, 1] for visualization

        # load the ground-truth shadow-free image
        gt_image = None
        if path_gt is not None:
            gt_image = Image.open(path_gt).convert('RGB')
            gt_image = gt_image.resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
            gt_image = np.array(gt_image) / 255.0
            gt_image = gt_image.astype(np.float32)
        else:
            print('No GT image')
            gt_image = np.ones_like(og_image)

        # prepare the image for the model
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image_tensor = torch.from_numpy(image).to(DEVICE)

        # make the prediction using G1 (shadow detection)
        detected_shadow = G1(image_tensor)

        # concatenate the input image with the detected shadow mask
        concat = torch.cat([image_tensor, detected_shadow], dim=1)

        # make the prediction using G2 (shadow removal)
        shadow_removal_image = G2(concat)

        # convert to numpy array
        shadow_removal_image = shadow_removal_image.cpu().numpy()
        shadow_removal_image = np.transpose(shadow_removal_image[0], (1, 2, 0))

        # unnormalize the prediction
        shadow_removal_image = shadow_removal_image * 0.5 + 0.5

        return og_image, gt_image, shadow_removal_image

def compare_datasets():
    """
    Compare model performance across different datasets
    """
    # Create a directory to save comparison results
    os.makedirs("prediction_comparison_stcgan", exist_ok=True)

    # Dictionary to store metrics for each dataset
    metrics = {}

    # Process each dataset
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset['name']}")

        # Load the models
        model_path_G1 = dataset["model_path_G1"]
        model_path_G2 = dataset["model_path_G2"]

        print(f"Loading G1 model from: {model_path_G1}")
        G1_weights = torch.load(model_path_G1, map_location=DEVICE)
        G1 = Generator(input_channels=3, output_channels=1)
        G1.load_state_dict(fix_model_state_dict(G1_weights))
        G1 = G1.to(DEVICE)

        print(f"Loading G2 model from: {model_path_G2}")
        G2_weights = torch.load(model_path_G2, map_location=DEVICE)
        G2 = Generator(input_channels=4, output_channels=3)
        G2.load_state_dict(fix_model_state_dict(G2_weights))
        G2 = G2.to(DEVICE)

        # Get test images
        shadow_images = glob.glob(os.path.join(dataset["test_shadow_path"], "*.jpg")) + \
                        glob.glob(os.path.join(dataset["test_shadow_path"], "*.png"))
        shadow_free_images = glob.glob(os.path.join(dataset["test_shadow_free_path"], "*.jpg")) + \
                             glob.glob(os.path.join(dataset["test_shadow_free_path"], "*.png"))

        # Sort to ensure matching pairs
        shadow_images.sort()
        shadow_free_images.sort()

        # Ensure we have matching pairs
        num_images = min(len(shadow_images), len(shadow_free_images))
        if num_images == 0:
            print(f"No test images found for dataset {dataset['name']}")
            continue

        print(f"Found {num_images} test image pairs")

        # Initialize metrics for this dataset
        dataset_ssim = []
        dataset_psnr = []
        dataset_rmse = []

        # Process a subset of images (up to 10) for visualization
        vis_count = min(10, num_images)
        fig, axes = plt.subplots(vis_count, 3, figsize=(15, 5 * vis_count))

        # Process each image
        for i in range(num_images):
            shadow_path = shadow_images[i]
            shadow_free_path = shadow_free_images[i]

            # Make prediction
            orig_image, gt_image, pred_image = make_predictions(G1, G2, shadow_path, shadow_free_path)

            # Calculate metrics
            ssim_value, psnr_value, rmse_value = calculate_metrics(gt_image, pred_image)
            dataset_ssim.append(ssim_value)
            dataset_psnr.append(psnr_value)
            dataset_rmse.append(rmse_value)

            # Save visualization for a subset of images
            if i < vis_count:
                if vis_count == 1:
                    ax = axes
                else:
                    ax = axes[i]

                ax[0].imshow(orig_image)
                ax[1].imshow(gt_image)
                ax[2].imshow(pred_image)

                ax[0].set_title("Shadowed")
                ax[1].set_title("Ground Truth")
                ax[2].set_title(f"Prediction (SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f}, RMSE: {rmse_value:.4f})")

                for a in ax:
                    a.axis('off')

        # Save the visualization
        plt.tight_layout()
        plt.savefig(f"prediction_comparison_stcgan/{dataset['name']}_samples.png")
        plt.close()

        # Calculate average metrics
        avg_ssim = np.mean(dataset_ssim)
        avg_psnr = np.mean(dataset_psnr)
        avg_rmse = np.mean(dataset_rmse)

        print(f"Dataset: {dataset['name']}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print("-" * 50)

        # Store metrics
        metrics[dataset['name']] = {
            "ssim": avg_ssim,
            "psnr": avg_psnr,
            "rmse": avg_rmse
        }

    # Create comparison bar chart
    if metrics:
        dataset_names = list(metrics.keys())
        ssim_values = [metrics[name]["ssim"] for name in dataset_names]
        psnr_values = [metrics[name]["psnr"] for name in dataset_names]
        rmse_values = [metrics[name]["rmse"] for name in dataset_names]

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # SSIM comparison
        ax1.bar(dataset_names, ssim_values)
        ax1.set_title("SSIM Comparison")
        ax1.set_ylim(0, 1)  # SSIM ranges from 0 to 1

        # PSNR comparison
        ax2.bar(dataset_names, psnr_values)
        ax2.set_title("PSNR Comparison")

        # RMSE comparison
        ax3.bar(dataset_names, rmse_values)
        ax3.set_title("RMSE Comparison")

        plt.tight_layout()
        plt.savefig("prediction_comparison_stcgan/metrics_comparison.png")
        plt.close()

if __name__ == "__main__":
    compare_datasets()
