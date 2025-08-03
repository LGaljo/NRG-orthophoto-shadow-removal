import glob
from PIL import Image
import config
import numpy as np
import torch
import os

TimagePaths = []
GTimagePaths = []
model = None
model_name = None

DATASETS = [
    {
        "name": "ISTD",
        "model_path": "output/unet_shadow_20250626223044_istd_e500.pth",
    },
    {
        "name": "SRD",
        "model_path": "output/unet_shadow_20250629075948_srd_e500.pth",
    },
    {
        "name": "USOS",
        "model_path": "output/unet_shadow_20250703063322_usos_e250.pth",
    },
    {
        "name": "USOS2024",
        "model_path": "output/output_20241122083203/unet_shadow_20241122083203_e200.pth",
    },
]


def save_prediction(predMask, output_path):
    """
    Save the prediction image to the specified path
    """
    # Convert the prediction to a PIL Image
    # Scale the prediction to 0-255 range
    pred_image = (predMask * 255).astype(np.uint8)

    # Create a PIL Image from the numpy array
    pred_pil = Image.fromarray(pred_image)

    # Save the image
    pred_pil.save(output_path)
    print(f"Saved prediction to {output_path}")


def make_predictions(path_t, output_dir):
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = Image.open(path_t).convert('RGB')

        # Get the image filename without extension
        img_name = os.path.splitext(os.path.basename(path_t))[0]

        # Create output path with model name
        output_path = os.path.join(output_dir, f"{img_name}_{model_name}.png")

        # resize the image and make a copy of it for visualization
        image = image.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        image = np.array(image) / 255
        image = image.astype(np.float32)

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image_tensor = torch.from_numpy(image).to(config.DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image_tensor).squeeze()
        predMask = predMask.cpu().numpy()
        predMask = np.transpose(predMask, (1, 2, 0))

        # Save the prediction image
        save_prediction(predMask, output_path)


if __name__ == '__main__':

    TimagePaths = glob.glob(os.path.join(config.TESTSET_T_PATH))
    TimagePaths = np.random.choice(TimagePaths, size=10)

    # load the image paths in our testing file
    print("[INFO] loading up test image paths...")
    print(f"Found {len(TimagePaths)} test images")

    for dataset in DATASETS:
        # Create output directory for predictions
        output_dir = f"predictions/{dataset['name']}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Saving predictions to {output_dir}")

        # load our model from disk and flash it to the current device
        print("[INFO] load up model...")

        # Extract model name from path
        model_name = os.path.basename(dataset['model_path']).split('.')[0]
        print(f"[INFO] Using model: {model_name}")

        # Load the model
        model = torch.load(dataset['model_path']).to(config.DEVICE)
        # set model to evaluation mode
        model.eval()

        # iterate over all test image paths
        for t_path in TimagePaths:
            # make predictions and save the results
            make_predictions(t_path, output_dir)
