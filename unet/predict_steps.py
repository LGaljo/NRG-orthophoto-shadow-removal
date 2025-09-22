import glob

import cv2
from PIL import Image
from torchvision.transforms.v2 import Compose, ToImage, ToDtype

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from unet.predict import predict

TimagePaths = []
GTimagePaths = []


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    # ax[1].imshow(origMask)
    ax[1].imshow(predMask)

    # set the titles of the subplots
    ax[0].set_title("Image")
    # ax[1].set_title("Ground truth")
    ax[1].set_title("Prediction")

    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()
    plt.close()


def make_predictions(model, path_t, path_gt, iteration=0):
    prediction = predict(model, path_t)

    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = Image.open(path_t).convert('RGB')

        # resize the image and make a copy of it for visualization
        image = image.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        og_image = np.array(image).astype(np.float32)
        og_image = og_image / 255

        # find the filename and generate the path to ground truth
        # mask
        gtMask = np.ones_like(image)
        if path_gt is not None:
            # load the ground-truth segmentation mask in grayscale mode
            # and resize it
            gtMask = Image.open(path_gt)
            gtMask = gtMask.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        else:
            print('No GT image')

        # Apply the same transformation pipeline as in dataset.py
        # to_tensor = Compose([
        #     ToImage(),
        #     ToDtype(torch.float32, scale=True),
        #     # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        # image_tensor = to_tensor(image).unsqueeze(0).to(config.DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        # predMask = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # prepare a plot for visualization
        prepare_plot(og_image, gtMask, prediction)
        cv2.imwrite(f'./prediction_path/image_{iteration}.jpg', cv2.cvtColor(prediction * 255, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # TimagePaths = glob.glob(os.path.join(config.TESTSET_T_PATH))
    # GTimagePaths = glob.glob(os.path.join(config.TESTSET_GT_PATH))

    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    # imagePaths = np.random.choice(imagePaths, size=10)
    TimagePaths = []
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-269.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-27.png")

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    # iterate over the randomly selected test image paths
    for epoch in range(5, 175, 5):
        print("test " + str(epoch))
        # model = glob.glob(f"output/output_usos_20250801222705/unet_shadow_20250801222705_e{epoch}.pth")
        # model = glob.glob(f"output/output_pretraining_20250805183113/unet_shadow_20250805183113_e{epoch}.pth")
        # model = glob.glob(f"output/output_20241122083203/unet_shadow_20241122083203_e{epoch}.pth")
        # model = glob.glob(f"output/output_pretraining_20250805193927/unet_shadow_20250805193927_e{epoch}.pth")
        # model = glob.glob(f"output/output_usos_20250906081803/unet_shadow_20250906081803_e{epoch}.pth")
        # model = glob.glob(f"output/output_usos_20250703063322/unet_shadow_20250703063322_e{epoch}.pth")
        model = glob.glob(f'output/output_usos_20250921214439/unet_shadow_20250921214439_e{epoch}.pth')  # usos s l1ssim
        i = 0
        unet = torch.load(model[i], map_location=config.DEVICE).to(config.DEVICE)

        # make predictions and visualize the results
        make_predictions(unet, TimagePaths[0], None, epoch)
