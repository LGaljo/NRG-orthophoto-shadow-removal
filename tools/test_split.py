from glob import glob
from os import path, mkdir
from PIL import Image
import argparse

from st_cgan.main import ST_CGAN


"""
This script takes all the images from the defined folder and converts them using the st-cgan model.

Refer to command parameters to parser code.
"""


def main(args):
    st_cgan = ST_CGAN(args.g1, args.g2)

    images = glob(args.input + "/*.jpg")
    for image_path in images:
        image = Image.open(image_path)
        image, mask = st_cgan.convert_image(image)
        if args.save_masks:
            pathB = path.join(path.join(args.output, 'masks'), path.basename(image_path))
            mask.save(pathB)
        pathC = path.join(path.join(args.output, 'converted'), path.basename(image_path))
        image.save(pathC)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remove shadows from given images",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', default="../test_images", required=True, help="Input path of images")
    parser.add_argument('-o', '--output', default="../test_images_out", help="Output path for images")
    parser.add_argument('-m', '--save_masks', default=False, help="Saves the masks if desired")
    parser.add_argument('-g1', default="../st_cgan/model/ST-CGAN_G1.pth", help="Location of G1 Generator model")
    parser.add_argument('-g2', default="../st_cgan/model/ST-CGAN_G2.pth", help="Location of G2 Generator model")
    args = parser.parse_args()

    if not path.exists(path.join(args.output)):
        mkdir(path.join(args.output))
    if not path.exists(path.join(args.output, 'masks')):
        mkdir(path.join(args.output, 'masks'))
    if not path.exists(path.join(args.output, 'converted')):
        mkdir(path.join(args.output, 'converted'))
    if not path.exists(args.input):
        print('No input folder')
        exit(1)

    main(args)
