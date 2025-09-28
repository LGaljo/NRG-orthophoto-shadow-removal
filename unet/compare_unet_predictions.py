# USAGE
# python compare_predictions.py
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
import lpips

import config

loss_fn_alex = lpips.LPIPS(net='alex').to(config.DEVICE) # best forward scores

# Define datasets and their corresponding model checkpoints
DATASETS = [
    {
        "name": "ISTD",
        "model_path": "output/output_istd_20250925214442/unet_shadow_20250925214442_e100.pth",
        "test_shadow_path": "../dataset/ISTD_Dataset/test/test_A",
        "test_shadow_free_path": "../dataset/ISTD_Dataset/test/test_C"
    },
    {
        "name": "SRD",
        "model_path": "output/output_srd_20250926072011/unet_shadow_20250926072011_e100.pth",
        "test_shadow_path": "../dataset/SRD/SRD_Test/SRD/shadow",
        "test_shadow_free_path": "../dataset/SRD/SRD_Test/SRD/shadow_free"
    },
    {
        "name": "USOS",
        # "model_path": "output/output_usos_20250906081803/unet_shadow_20250906081803_e165.pth",
        "model_path": "output/output_usos_20250923063528/unet_shadow_20250923063528_e75.pth",
        "test_shadow_path": "../dataset/unity_dataset/mixed_visibility_dataset_320/test/train_A",
        "test_shadow_free_path": "../dataset/unity_dataset/mixed_visibility_dataset_320/test/train_C"
    },
    # Add more datasets here as needed
]

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
    Calculate SSIM, PSNR, RMSE, LPIPS metrics between ground truth and predicted images
    """
    # Convert images to numpy arrays if they're not already
    if isinstance(gt_image, torch.Tensor):
        gt_image = gt_image.cpu().numpy()
    if isinstance(pred_image, torch.Tensor):
        pred_image = pred_image.cpu().numpy()

    # Ensure inputs are in [0,1]
    gt_image = np.clip(gt_image, 0.0, 1.0)
    pred_image = np.clip(pred_image, 0.0, 1.0)

    # Calculate SSIM
    ssim_value = ssim(gt_image, pred_image, data_range=1.0, multichannel=True, channel_axis=2)

    # Calculate PSNR
    psnr_value = psnr(gt_image, pred_image, data_range=1.0)

    # Calculate RMSE (Root Mean Square Error)
    mse_value = mse(gt_image, pred_image)
    rmse_value = np.sqrt(mse_value)

    # Calculate LPIPS: expects tensors in [-1, 1]
    gt_t = torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).float()
    pred_t = torch.from_numpy(pred_image).permute(2, 0, 1).unsqueeze(0).float()
    gt_t = gt_t * 2.0 - 1.0
    pred_t = pred_t * 2.0 - 1.0
    with torch.no_grad():
        lpips_value = loss_fn_alex(gt_t.to(config.DEVICE), pred_t.to(config.DEVICE)).item()

    return ssim_value, psnr_value, rmse_value, lpips_value

def make_predictions(model, path_t, path_gt):
    """
    Make predictions using the model and return the original image, ground truth, and prediction
    """
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, convert to RGB
        image = Image.open(path_t).convert('RGB')

        # resize the image and make a copy of it for visualization
        image = image.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        image = np.array(image) / 255
        image = image.astype(np.float32)
        og_image = image.copy()

        # load the ground-truth shadow-free image
        gt_image = None
        if path_gt is not None:
            gt_image = Image.open(path_gt).convert('RGB')
            gt_image = gt_image.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
            gt_image = np.array(gt_image) / 255
            gt_image = gt_image.astype(np.float32)
        else:
            print('No GT image')
            gt_image = np.ones_like(image)

        # prepare the image for the model
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image_tensor = torch.from_numpy(image).to(config.DEVICE)

        # make the prediction and convert to numpy array
        predMask = model(image_tensor).squeeze()
        predMask = predMask.cpu().numpy()
        predMask = np.transpose(predMask, (1, 2, 0))

        return og_image, gt_image, predMask

def compare_datasets():
    """
    Compare model performance across different datasets
    """
    # Create a directory to save comparison results
    os.makedirs("prediction_comparison", exist_ok=True)

    # Dictionary to store metrics for each dataset
    metrics = {}

    # Process each dataset
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset['name']}")

        # Load the model
        model_path = dataset["model_path"]
        print(f"Loading model from: {model_path}")
        model = torch.load(model_path, map_location=config.DEVICE, weights_only=False).to(config.DEVICE)

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
        dataset_lpips = []

        # Process a subset of images (up to 10) for visualization
        vis_count = min(10, num_images)
        fig, axes = plt.subplots(vis_count, 3, figsize=(15, 5 * vis_count))

        # Process each image
        for i in range(num_images):
            shadow_path = shadow_images[i]
            shadow_free_path = shadow_free_images[i]

            # Make prediction
            orig_image, gt_image, pred_image = make_predictions(model, shadow_path, shadow_free_path)

            # Calculate metrics
            ssim_value, psnr_value, rmse_value, lpips_value = calculate_metrics(gt_image, pred_image)
            dataset_ssim.append(ssim_value)
            dataset_psnr.append(psnr_value)
            dataset_rmse.append(rmse_value)
            dataset_lpips.append(lpips_value)

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
                ax[2].set_title(f"Prediction (SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f}, RMSE: {rmse_value:.4f}, LPIPS: {lpips_value:.4f})")

                for a in ax:
                    a.axis('off')

        # Save the visualization
        plt.tight_layout()
        plt.savefig(f"prediction_comparison/{dataset['name']}_samples.png")
        plt.close()

        # Calculate average metrics
        avg_ssim = np.mean(dataset_ssim)
        avg_psnr = np.mean(dataset_psnr)
        avg_rmse = np.mean(dataset_rmse)
        avg_lpips = np.mean(dataset_lpips)

        print(f"Dataset: {dataset['name']}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        print("-" * 50)

        # Store metrics
        metrics[dataset['name']] = {
            "ssim": avg_ssim,
            "psnr": avg_psnr,
            "rmse": avg_rmse,
            "lpips": avg_lpips
        }

    # Create comparison bar chart
    if metrics:
        dataset_names = list(metrics.keys())
        ssim_values = [metrics[name]["ssim"] for name in dataset_names]
        psnr_values = [metrics[name]["psnr"] for name in dataset_names]
        rmse_values = [metrics[name]["rmse"] for name in dataset_names]
        lpips_values = [metrics[name]["lpips"] for name in dataset_names]

        # Create figure with four subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))

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

        # LPIPS comparison
        ax4.bar(dataset_names, lpips_values)
        ax4.set_title("LPIPS Comparison (lower is better)")

        plt.tight_layout()
        plt.savefig("prediction_comparison/metrics_comparison.png")
        plt.close()

if __name__ == "__main__":
    compare_datasets()
