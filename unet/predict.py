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


def make_predictions(model, path_t, path_gt):
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


if __name__ == '__main__':
    TimagePaths = glob.glob(os.path.join(config.TESTSET_PATH, "train_A", "*.png"))
    GTimagePaths = glob.glob(os.path.join(config.TESTSET_PATH, "train_C", "*.png"))

    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    # imagePaths = np.random.choice(imagePaths, size=10)
    TimagePaths = []
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-27.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-74.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-B0830-278.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-27.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-74.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-B0830-278.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-209.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-269.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0738-66.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0738-123.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0504-81.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0504-240.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0637-251.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0644-86.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0644-129.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240620-D0722-239.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-B0108-210.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-E0835-238.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-I0901-113.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240624-D0223-258.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0515-219.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0332-289.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0449-181.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0529-22.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0529-230.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0211-26.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0211-283.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-J0824-63.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K0910-61.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K1111-182.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K1111-366.png")
    TimagePaths = np.append(TimagePaths, "../dataset/unity_dataset/mixed_visibility_dataset_320/train/train_A/DOF5-20240602-D0717_shadowClear_12e28f16b79320f8-x255-z255-Hard-39.png")
    TimagePaths = np.append(TimagePaths, "../dataset/unity_dataset/mixed_visibility_dataset_320/train/train_A/DOF5-20240602-D0717_shadowClear_63b003c5119215b3-x255-z195-Hard-2.png")

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    # model = glob.glob("output/output_20240930215622/unet_shadow_20240930215622.pth")
    # model = glob.glob("output/output_20240630001817/unet_shadow_20240630001817_e100.pth")
    # model = glob.glob("output/output_20240929220403/unet_shadow_20240929220403_e25.pth")
    # model = glob.glob("output/output_20240930225008/unet_shadow_20240930225008_e30.pth")
    # model = glob.glob("output/output_20241001074431/unet_shadow_20241001074431_e50.pth")
    # model = glob.glob("output/output_20241001231452/unet_shadow_20241001231452_e200.pth")
    # model = glob.glob("output/output_20241002213808/unet_shadow_20241002213808_e30.pth")
    # model = glob.glob("output/output_20241019231437/unet_shadow_20241019231437_e70.pth")
    # model = glob.glob("output/output_20241024223406/unet_shadow_20241024223406_e100.pth")
    # model = glob.glob("output/output_20241024223406/unet_shadow_20241024223406_e40.pth")
    # model = glob.glob("output/output_20241111072901/unet_shadow_20241111072901_e200.pth")
    # model = glob.glob("output/output_20241120173728/unet_shadow_20241120173728.pth")
    model = glob.glob("output/output_20241120220442/unet_shadow_20241120220442.pth")
    i = 0
    unet = torch.load(model[i]).to(config.DEVICE)

    # iterate over the randomly selected test image paths
    for (t_path, gt_path) in zip(TimagePaths, GTimagePaths):
        # make predictions and visualize the results
        make_predictions(unet, t_path, gt_path)
