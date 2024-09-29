# USAGE
# python predict.py
# import the necessary packages
import glob

from PIL import Image

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")

    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = Image.open(imagePath).convert('RGB')

        # resize the image and make a copy of it for visualization
        image = image.resize((256, 256))
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        og_image = image.copy()

        # find the filename and generate the path to ground truth
        # mask
        gtMask = np.ones_like(image)
        try:
            filename = imagePath.split(os.path.sep)[-1]
            groundTruthPath = os.path.join(config.GT_DATASET_PATH, filename)

            # load the ground-truth segmentation mask in grayscale mode
            # and resize it
            gtMask = Image.open(groundTruthPath)
            gtMask = gtMask.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        except:
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


if __name__ == '__main__':
    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    imagePaths = []
    # imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
    # imagePaths = np.random.choice(imagePaths, size=10)
    imagePaths = np.append(imagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-27.png")
    imagePaths = np.append(imagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-74.png")
    imagePaths = np.append(imagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-B0830-278.png")
    imagePaths = np.append(imagePaths, "../dataset/ortophoto_pretraining/train_A/DOF5-20240602-D0717-27.png")
    imagePaths = np.append(imagePaths, "../dataset/ortophoto_pretraining/train_A/DOF5-20240602-D0717-74.png")
    imagePaths = np.append(imagePaths, "../dataset/ortophoto_pretraining/train_A/DOF5-20240602-B0830-278.png")

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    model = glob.glob(config.BASE_OUTPUT + "/*.pth")
    unet = torch.load(model[0]).to(config.DEVICE)

    # iterate over the randomly selected test image paths
    for path in imagePaths:
        # make predictions and visualize the results
        make_predictions(unet, path)
