import glob

import cv2
from PIL import Image
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

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
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = Image.open(path_t).convert('RGB')

        # resize the image and make a copy of it for visualization
        image = image.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        image = np.array(image) / 255
        image = image.astype(np.float32)
        og_image = image.copy()

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
        # predMask = (predMask-np.min(predMask))/(np.max(predMask)-np.min(predMask))

        # prepare a plot for visualization
        prepare_plot(og_image, gtMask, predMask)
        cv2.imwrite(f'./prediction_path/image_{iteration}.jpg', cv2.cvtColor(predMask * 255, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    TimagePaths = glob.glob(os.path.join(config.TESTSET_PATH, "train_A", "*.png"))
    GTimagePaths = glob.glob(os.path.join(config.TESTSET_PATH, "train_C", "*.png"))

    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    # imagePaths = np.random.choice(imagePaths, size=10)
    TimagePaths = []
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-269.png")

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    # iterate over the randomly selected test image paths
    for epoch in range(5, 91, 5):
        print("test " + str(epoch))
        model = glob.glob(f"output/output_20241108171754/unet_shadow_20241108171754_e{epoch}.pth")
        i = 0
        unet = torch.load(model[i]).to(config.DEVICE)

        # make predictions and visualize the results
        make_predictions(unet, TimagePaths[0], None, epoch)

    for epoch in range(95, 201, 5):
        print("test " + str(epoch))
        model = glob.glob(f"output/output_20241111072901/unet_shadow_20241111072901_e{epoch}.pth")
        unet = torch.load(model[0]).to(config.DEVICE)

        # make predictions and visualize the results
        make_predictions(unet, TimagePaths[0], None, epoch)
