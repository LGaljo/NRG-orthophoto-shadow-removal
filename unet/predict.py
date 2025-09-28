# USAGE
# python predict.py
# import the necessary packages
import glob

from IPython.utils import path
from PIL import Image

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision.transforms.v2 import functional as F, Compose, ToImage, ToDtype, Normalize, PILToTensor

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


def predict(model, image_path):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = Image.open(image_path).convert('RGB')
        image = image.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))

        # Apply the same transformation pipeline as in dataset.py
        to_tensor = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            # Normalize(mean=config.MEAN, std=config.STD),
        ])

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image_tensor = to_tensor(image).unsqueeze(0).to(config.DEVICE)
        prediction = model(image_tensor).squeeze().cpu().permute(1, 2, 0).numpy()

        return prediction


def make_predictions(model, path_t, path_gt):
    prediction = predict(model, path_t)

    image = Image.open(path_t).convert('RGB')
    image = image.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))

    # find the filename and generate the path to ground truth mask
    ground_truth = np.ones_like(image)
    if path_gt is not None:
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        ground_truth = Image.open(path_gt)
        ground_truth = ground_truth.resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))

    # prepare a plot for visualization
    Image.fromarray((prediction * 255).astype(np.uint8)).save(os.path.basename(path_t).split('.')[0] + "-unet-usos.png")

    prepare_plot(image, ground_truth, prediction)


if __name__ == '__main__':
    # TimagePaths = glob.glob(os.path.join(config.TESTSET_T_PATH))
    # GTimagePaths = glob.glob(os.path.join(config.TESTSET_GT_PATH))

    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    # imagePaths = np.random.choice(imagePaths, size=10)
    TimagePaths = []
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-27.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-B0830-278.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0644-86.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240624-D0223-258.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K0910-61.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/unity_dataset/mixed_visibility_dataset_320/train/train_A/DOF5-20240602-D0717_shadowClear_12e28f16b79320f8-x255-z255-Hard-39.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/unity_dataset/mixed_visibility_dataset_320/test/train_A/DOF5-20240620-D0722_shadowClear_e20c040c821fef9e-x255-z255-Hard-28.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0504-240.png")
    TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0449-181.png")
    #
    # TimagePaths = np.append(TimagePaths, "../dataset/ISTD_Dataset/test/test_A/95-4.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ISTD_Dataset/test/test_A/120-13.png")
    #
    # TimagePaths = np.append(TimagePaths, "../dataset/SRD/SRD_Test/SRD/shadow/_MG_6317.jpg")
    # TimagePaths = np.append(TimagePaths, "../dataset/SRD/SRD_Test/SRD/shadow/IMG_6803.jpg")

    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-74.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-209.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0717-269.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0738-66.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-D0738-123.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0504-81.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0637-251.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240602-E0644-129.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240620-D0722-239.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-B0108-210.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-E0835-238.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240623-I0901-113.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0515-219.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0332-289.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0529-22.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-D0529-230.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0211-26.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-E0211-283.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-J0824-63.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K1111-182.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/ortophoto_pretraining/train_C/DOF5-20240920-K1111-366.png")
    # TimagePaths = np.append(TimagePaths, "../dataset/unity_dataset/mixed_visibility_dataset_320/train/train_A/DOF5-20240602-D0717_shadowClear_63b003c5119215b3-x255-z195-Hard-2.png")

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
    # model = glob.glob("output/output_20241120220442/unet_shadow_20241120220442.pth")
    # model = glob.glob("output/output_20241121220913/unet_shadow_20241121220913_e20.pth")
    # model = glob.glob("output/unet_shadow_20241122083203_e100.pth")
    # model = glob.glob("output/output_20250801070442/unet_shadow_20250801070442.pth")
    # model = glob.glob("output/output_20241211224201/unet_shadow_20241211224201_e100.pth")
    # model = glob.glob("output/output_20241213175748/unet_shadow_20241213175748_e100.pth")
    # model = glob.glob("output/output_20241122083203/unet_shadow_20241122083203_e200.pth")
    # model = glob.glob("output/output_20241209223619/unet_shadow_20241209223619.pth")
    # model = glob.glob("output/output_20241210193215/unet_shadow_20241210193215.pth")
    # model = glob.glob("output/output_20241210210910/unet_shadow_20241210210910.pth")
    # model = glob.glob("output/output_20241211182133/unet_shadow_20241211182133_e10.pth")
    # model = glob.glob("output/unet_shadow_2025072/5192114_pretraining_e10.pth")
    # model = glob.glob("output/output_20250731150744/unet_shadow_20250731150744_e100.pth")
    # model = glob.glob("output/output_usos_20250801222705/unet_shadow_20250801222705_e200.pth")
    # model = glob.glob("output/output_usos_20250802224220/unet_shadow_20250802224220_e15.pth")
    # model = glob.glob("output/output_pretraining_20250803092013/unet_shadow_20250803092013_e50.pth")
    # model = glob.glob("output/output_pretraining_20250805193927/unet_shadow_20250805193927_e175.pth")
    # model = glob.glob("output/output_pretraining_20250811165205/unet_shadow_20250811165205.pth")
    # model = glob.glob("output/unet_shadow_20250806211308_usos_e40.pth")
    # model = glob.glob("output/unet_shadow_20250725192114_pretraining_e15.pth")
    # model = glob.glob("output/unet_shadow_20250811152315_e30.pth")
    # model = glob.glob("output/unet_shadow_20250626223044_istd_e500.pth")
    # model = glob.glob("output/unet_shadow_20250629075948_srd_e500.pth")
    # model = glob.glob("output/output_pretraining_20250901222915/unet_shadow_20250901222915_e40.pth")
    # model = glob.glob("output/output_pretraining_20250902193310/unet_shadow_20250902193310_e105.pth")
    # model = glob.glob("output/output_usos_20250903225233/unet_shadow_20250903225233_e25.pth")
    # model = glob.glob("output/output_usos_20250905181321/unet_shadow_20250905181321.pth")
    # model = glob.glob("output/output_usos_20250905233425/unet_shadow_20250905233425.pth")
    # model = glob.glob("output/output_usos_20250906081803/unet_shadow_20250906081803_e165.pth")
    # model = glob.glob("output/output_usos_20250905233425/unet_shadow_20250905233425_e65.pth")
    # model = glob.glob("output/unet_shadow_20250918181839_e5.pth")
    # model = glob.glob("output/unet_shadow_20250919045059.pth")
    # model = glob.glob("output/unet_shadow_20250919141649_e20.pth")
    # model = glob.glob("output/unet_shadow_20250919173118.pth")
    # model = glob.glob("output/unet_shadow_20250920121910_e50.pth") # latest pretraining
    # model = glob.glob("output/unet_shadow_20250920193451_e60.pth") # usos - fail
    # model = glob.glob("output/unet_shadow_20250921085812_e20.pth") # usos - fail
    # model = glob.glob("output/output_usos_20250921155209/unet_shadow_20250921155209_e20.pth") # usos xs mse
    # model = glob.glob("output/output_usos_20250921161043/unet_shadow_20250921161043_e20.pth") # usos xs mse
    # model = glob.glob("output/output_usos_20250921163709/unet_shadow_20250921163709.pth") # usos xs sml
    # model = glob.glob("output/output_usos_20250921171843/unet_shadow_20250921171843.pth") # usos xs sml
    # model = glob.glob("output/output_usos_20250921184850/unet_shadow_20250921184850.pth") # usos xs sml
    # model = glob.glob("output/output_usos_20250921203036/unet_shadow_20250921203036.pth") # usos xs l1ssim
    # model = glob.glob("output/output_usos_20250922195221/unet_shadow_20250922195221.pth") # usos s l1ssim
    # model = glob.glob("output/output_usos_20250922230121/unet_shadow_20250922230121.pth")
    # model = glob.glob("output/output_usos_20250922211950/unet_shadow_20250922211950_e20.pth")
    # model = glob.glob("output/output_usos_20250923074529/unet_shadow_20250923074529.pth")
    # model = glob.glob("output/unet_shadow_20250923061157.pth")
    # model = glob.glob("output/output_usos_20250923063528/unet_shadow_20250923063528_e75.pth")
    model = glob.glob("output/output_usos_20250921214439/unet_shadow_20250921214439.pth") # usos s l1ssim tale dela super
    # model = glob.glob("output/output_istd_20250925214442/unet_shadow_20250925214442_e100.pth")
    # model = glob.glob("output/output_srd_20250926072011/unet_shadow_20250926072011_e100.pth")
    i = 0
    unet = torch.load(model[i], weights_only=False).to(config.DEVICE)

    # iterate over the randomly selected test image paths
    for t_path in TimagePaths:
        # make predictions and visualize the results
        make_predictions(unet, t_path, None)
