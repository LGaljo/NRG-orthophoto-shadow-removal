import math

from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from skimage import io, img_as_float

from PIL import Image, ImageChops, ImageEnhance, ImageCms


def rgb2labProfile():
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    return ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")


def to_lab(image):
    return ImageCms.applyTransform(image, rgb2labProfile())


if __name__ == '__main__':
    image_gt = img_as_float(io.imread('./evalimg/image_gt.png'))
    image_pd = img_as_float(io.imread('./evalimg/image_pd.png'))
    mask_gt = img_as_float(io.imread('./evalimg/mask_gt.png'))
    mask_pd = img_as_float(io.imread('./evalimg/mask_pd.png'))

    # RMSE
    print("rmse gt/pd", mean_squared_error(image_gt, image_pd))
    print("rmse gt/gt", mean_squared_error(image_gt, image_gt))

    # SSIM
    ssim, _ = structural_similarity(image_gt, image_pd, full=True, channel_axis=2, multichannel=True, data_range=1)
    print("ssim gt/pd", ssim)
    ssim, _ = structural_similarity(image_gt, image_gt, full=True, channel_axis=2, multichannel=True, data_range=1)
    print("ssim gt/gt", ssim)

    # PSNR
    psnr = peak_signal_noise_ratio(image_gt, image_pd, data_range=1)
    print("psnr gt/pd", psnr)
    psnr = peak_signal_noise_ratio(image_gt, image_gt, data_range=1)
    print("psnr gt/gt", psnr)

    print('mask')
    print(mean_squared_error(mask_gt, mask_pd))
    ssim, img_ssim = structural_similarity(mask_gt, mask_pd, full=True, data_range=1)
    print(ssim)
