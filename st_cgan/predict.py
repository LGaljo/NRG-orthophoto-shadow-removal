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
import lpips

from ST_CGAN import Generator
from unet import config

loss_fn_alex = lpips.LPIPS(net='alex').to(config.DEVICE)

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

def unnormalize(x):
    """
    Unnormalize the image from [-1, 1] to [0, 1]
    """
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x

def make_predictions(G1, G2, path_t):
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

        return og_image, shadow_removal_image

def compare_datasets():
    """
    Compare model performance across different datasets
    """
    # Create a directory to save comparison results
    os.makedirs("predictions", exist_ok=True)

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
        shadow_images = []
        # shadow_images = np.append(shadow_images, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-27.png")
        # shadow_images = np.append(shadow_images, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-B0830-278.png")
        # shadow_images = np.append(shadow_images, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0644-86.png")
        # shadow_images = np.append(shadow_images, "../dataset/ortophoto_pretraining/train_C/DOF5-20240624-D0223-258.png")
        # shadow_images = np.append(shadow_images, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K0910-61.png")
        # shadow_images = np.append(shadow_images,
        #                         "../dataset/unity_dataset/mixed_visibility_dataset_320/train/train_A/DOF5-20240602-D0717_shadowClear_12e28f16b79320f8-x255-z255-Hard-39.png")
        # shadow_images = np.append(shadow_images,
        #                         "../dataset/unity_dataset/mixed_visibility_dataset_320/test/train_A/DOF5-20240620-D0722_shadowClear_e20c040c821fef9e-x255-z255-Hard-28.png")

        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-74.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-209.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-269.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0738-66.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0738-123.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0504-81.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0504-240.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0637-251.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0644-129.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240620-D0722-239.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-B0108-210.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-E0835-238.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-I0901-113.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0515-219.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0332-289.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0449-181.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0529-22.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0529-230.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0211-26.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0211-283.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-J0824-63.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K1111-182.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K1111-366.png")
        # TimagePaths = np.append(TimagePaths, "../dataset/unity_dataset/mixed_visibility_dataset_320/train/train_A/DOF5-20240602-D0717_shadowClear_63b003c5119215b3-x255-z195-Hard-2.png")

        shadow_images = np.append(shadow_images, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0504-240.png")
        shadow_images = np.append(shadow_images, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0449-181.png")

        # shadow_images = np.append(shadow_images, "../dataset/ISTD_Dataset/test/test_A/95-4.png")
        # shadow_images = np.append(shadow_images, "../dataset/ISTD_Dataset/test/test_A/120-13.png")
        #
        # shadow_images = np.append(shadow_images, "../dataset/SRD/SRD_Test/SRD/shadow/_MG_6317.jpg")
        # shadow_images = np.append(shadow_images, "../dataset/SRD/SRD_Test/SRD/shadow/IMG_6803.jpg")

        # Process each image
        for i in range(shadow_images.size):
            shadow_path = shadow_images[i]

            # Make prediction
            orig_image, pred_image = make_predictions(G1, G2, shadow_path)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig_image)
            axes[1].imshow(pred_image)

            axes[0].set_title("Input")
            axes[1].set_title("Prediction")

            for a in axes:
                a.axis('off')

            # prepare a plot for visualization
            Image.fromarray((pred_image * 255).astype(np.uint8)).save(
                os.path.basename(shadow_path).split('.')[0] + f"-stcgan-{dataset['name']}.png")

        # Save the visualization
        plt.tight_layout()
        plt.savefig(f"predictions/{dataset['name']}_samples.png")
        plt.close()

if __name__ == "__main__":
    compare_datasets()
