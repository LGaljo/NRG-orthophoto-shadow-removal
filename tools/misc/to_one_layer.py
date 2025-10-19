from glob import glob

from PIL import Image

in_path = './img'

"""
This short code converts any image to single color grayscale image (useful for converting masks).
"""


def main():
    images = glob(in_path + "/*.jpg")
    for image_path in images:
        image = Image.open(image_path).convert('L')
        image.save(image_path)


if __name__ == '__main__':
    main()
